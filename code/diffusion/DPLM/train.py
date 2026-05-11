# --steps 100000 --batch 128 --lr 2e-4 --amp bf16
import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .model import DPLM, DPLMConfig
from .diffusion import DPLMDiffusion, DPLMDiffusionConfig
from ..LD4LG.data import PairedAntibodyDataset, make_collate_fn
from ..LD4LG.tokenizer import AATokenizer


def cosine_decay(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    frac = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=1_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--cfg-weight", type=float, default=2.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--max-len", type=int, default=288)
    ap.add_argument("--resume", type=Path, default=None)
    # Modify and add joint cfg for joint CFG ablation May 9th, 2026 Y.L.
    ap.add_argument("--joint-cfg", action="store_true",
                help="use joint CFG dropout (all 3 conditions dropped together) instead of independent")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    ds_train = PairedAntibodyDataset(args.data, "train", max_len=args.max_len)
    tok = AATokenizer()
    collate = make_collate_fn(tok.pad_id, tok.bos_id)
    dl = DataLoader(
        ds_train, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    cfg_model = DPLMConfig(
        max_len=args.max_len,
        num_isotypes=ds_train.num_isotypes,
        num_v_families=ds_train.num_v_families,
        num_light_loci=ds_train.num_light_loci,
    )
    # Modify for joint CFG ablation May 9th, 2026 Y.L.
    #model = DPLM(cfg_model, cond_drop_prob=args.cond_drop).to(device)
    model = DPLM(cfg_model, cond_drop_prob=args.cond_drop, joint_cfg=args.joint_cfg).to(device)
    diff_cfg = DPLMDiffusionConfig(cfg_weight=args.cfg_weight)
    diffusion = DPLMDiffusion(model=model, cfg=diff_cfg).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters")

    # no weight decay on norms / biases / embeddings
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim < 2 or "norm" in n.lower() or "bias" in n.lower() or "emb" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.999),
    )

    scaler = torch.amp.GradScaler(enabled=(args.amp == "fp16"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": torch.float32}[args.amp]

    # resume
    step = 0
    if args.resume is not None and args.resume.exists():
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        if "optim" in sd:
            optim.load_state_dict(sd["optim"])
        step = sd.get("step", 0)
        print(f"[resume] from {args.resume} @ step={step}")

    t0 = time.time()
    data_iter = iter(dl)
    model.train()
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        tokens = batch["source_tokens"].to(device, non_blocking=True)
        iso    = batch["iso"].to(device,  non_blocking=True)
        vfam   = batch["vfam"].to(device, non_blocking=True)
        loc    = batch["loc"].to(device,  non_blocking=True)

        lr_scale = cosine_decay(step, args.warmup, args.steps)
        for g in optim.param_groups:
            g["lr"] = args.lr * lr_scale

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(args.amp != "off")):
            loss = diffusion.loss(tokens, iso, vfam, loc).float()

        if args.amp == "fp16":
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:>6d} | loss {loss.item():.4f} | lr {args.lr*lr_scale:.2e} | {dt:.1f}s")

        step += 1

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "cfg_model": cfg_model.__dict__,
                "cfg_diffusion": diff_cfg.__dict__,
                "args": vars(args),
            }
            torch.save(ckpt, args.out / f"dplm_{step}.pt")
            torch.save(ckpt, args.out / "dplm_latest.pt")


if __name__ == "__main__":
    main()