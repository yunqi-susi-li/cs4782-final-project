import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from .autoencoder import AutoencoderConfig, LanguageAutoencoder
from .data import PairedAntibodyDataset, make_collate_fn
from .ddp_utils import ddp_setup, ddp_teardown, is_main, all_reduce_mean, unwrap
from .tokenizer import AATokenizer


def linear_warmup_linear_decay(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    frac = (step - warmup) / max(1, total - warmup)
    return max(0.0, 1.0 - frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="processed memmap folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--warmup", type=int, default=1_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max-len", type=int, default=288)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--encoder-layers", type=int, default=6)
    ap.add_argument("--decoder-layers", type=int, default=6)
    ap.add_argument("--compress-layers", type=int, default=3)
    ap.add_argument("--reconstruct-layers", type=int, default=3)
    ap.add_argument("--latent-len", type=int, default=32)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from")
    args = ap.parse_args()

    state = ddp_setup()
    device = state.device
    if is_main(state):
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"[ddp] world={state.world_size} rank={state.rank} device={device}")
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = AATokenizer()
    cfg = AutoencoderConfig(
        vocab_size=tok.vocab_size,
        pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id,
        max_source_len=args.max_len, max_target_len=args.max_len,
        dim=args.dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        compress_layers=args.compress_layers,
        reconstruct_layers=args.reconstruct_layers,
        latent_len=args.latent_len,
        latent_dim=args.latent_dim,
    )
    model = LanguageAutoencoder(cfg).to(device)
    if is_main(state):
        print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters")

    # wrap with DDP before torch.compile so unwrap() works the same in both branches
    if state.is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[state.local_rank], find_unused_parameters=True,
        )
    if args.compile:
        model = torch.compile(model)

    ds_train = PairedAntibodyDataset(args.data, "train", max_len=args.max_len)
    val_meta = args.data / "val.meta.json"
    ds_val = PairedAntibodyDataset(args.data, "val", max_len=args.max_len) if val_meta.exists() else None
    if is_main(state) and ds_val is None:
        print("[note] no val split found -- skipping validation passes")
    collate = make_collate_fn(tok.pad_id, tok.bos_id)

    if state.is_distributed:
        sampler = DistributedSampler(ds_train, shuffle=True, drop_last=True)
        dl_train = DataLoader(
            ds_train, batch_size=args.batch, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate, drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        sampler = None
        dl_train = DataLoader(
            ds_train, batch_size=args.batch, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate, drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
    ) if ds_val is not None else None

    # no weight decay on biases / LayerNorm
    decay, no_decay = [], []
    for n, p in unwrap(model).named_parameters():
        if p.ndim < 2 or "norm" in n.lower() or "bias" in n.lower():
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
        unwrap(model).load_state_dict(sd["model"])
        if "optim" in sd:
            optim.load_state_dict(sd["optim"])
        step = sd.get("step", 0)
        if is_main(state):
            print(f"[resume] from {args.resume} @ step={step}")

    t0 = time.time()
    epoch = 0
    data_iter = iter(dl_train)
    model.train()
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(dl_train)
            batch = next(data_iter)

        src = batch["source_tokens"].to(device, non_blocking=True)
        di  = batch["decoder_input"].to(device, non_blocking=True)
        tgt = batch["target_tokens"].to(device, non_blocking=True)

        lr_scale = linear_warmup_linear_decay(step, args.warmup, args.steps)
        for g in optim.param_groups:
            g["lr"] = args.lr * lr_scale

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(args.amp != "off")):
            loss, _ = model(src, di, tgt)
        if args.amp == "fp16":
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), args.grad_clip)
            scaler.step(optim); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), args.grad_clip)
            optim.step()

        if step % args.log_every == 0 and is_main(state):
            loss_r = all_reduce_mean(loss.detach(), state)
            dt = time.time() - t0
            print(f"step {step:>6d} | loss {loss_r.item():.4f} | lr {args.lr*lr_scale:.2e} | {dt:.1f}s")

        step += 1

        if (step % args.ckpt_every == 0 or step == args.steps) and is_main(state):
            v_mean = float("nan")
            if dl_val is not None:
                model.eval()
                v_losses = []
                with torch.no_grad():
                    for i, vb in enumerate(dl_val):
                        if i >= 50:
                            break
                        src = vb["source_tokens"].to(device)
                        di  = vb["decoder_input"].to(device)
                        tgt = vb["target_tokens"].to(device)
                        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(args.amp != "off")):
                            vl, _ = model(src, di, tgt)
                        v_losses.append(vl.item())
                v_mean = sum(v_losses) / max(1, len(v_losses))
                print(f"  val loss @ step {step} = {v_mean:.4f}")
                model.train()
            else:
                print(f"  step {step}: no val split -- saving ckpt only")

            ckpt = {
                "step": step,
                "model": unwrap(model).state_dict(),
                "optim": optim.state_dict(),
                "cfg": cfg.__dict__,
                "val_loss": v_mean,
                "args": vars(args),
            }
            torch.save(ckpt, args.out / f"autoencoder_{step}.pt")
            torch.save(ckpt, args.out / "autoencoder_latest.pt")
            with open(args.out / "train_log.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "val_loss": v_mean}) + "\n")

    ddp_teardown(state)


if __name__ == "__main__":
    main()