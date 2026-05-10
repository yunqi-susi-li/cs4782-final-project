import argparse
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from .autoencoder import AutoencoderConfig, LanguageAutoencoder
from .data import PairedAntibodyDataset, make_collate_fn
from .ddp_utils import ddp_setup, ddp_teardown, is_main, all_reduce_mean, unwrap
from .denoiser import Denoiser, DenoiserConfig
from .diffusion import DiffusionConfig, GaussianDiffusion
from .ema import EMA
from .tokenizer import AATokenizer


def cosine_decay_schedule(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    frac = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))


def build_autoencoder(ae_ckpt, device):
    # weights_only=False: our own ckpts contain a Path in args
    sd = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
    cfg = AutoencoderConfig(**sd["cfg"])
    model = LanguageAutoencoder(cfg).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--warmup", type=int, default=1_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--self-cond-prob", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=2.0)
    ap.add_argument("--loss-type", default="l2")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=10_000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    state = ddp_setup()
    device = state.device
    if is_main(state):
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"[ddp] world={state.world_size} rank={state.rank} device={device}")
    torch.backends.cuda.matmul.allow_tf32 = True

    # frozen AE
    ae = build_autoencoder(args.ae_ckpt, device)
    if is_main(state):
        print(f"[ae] latent shape = ({ae.cfg.latent_len}, {ae.cfg.latent_dim})")

    ds_train = PairedAntibodyDataset(args.data, "train", max_len=ae.cfg.max_source_len)
    tok = AATokenizer()
    collate = make_collate_fn(tok.pad_id, tok.bos_id)
    if state.is_distributed:
        sampler = DistributedSampler(ds_train, shuffle=True, drop_last=True)
        dl = DataLoader(
            ds_train, batch_size=args.batch, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate, drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        sampler = None
        dl = DataLoader(
            ds_train, batch_size=args.batch, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate, drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )

    d_cfg = DenoiserConfig(
        latent_len=ae.cfg.latent_len,
        latent_dim=ae.cfg.latent_dim,
        num_isotypes=ds_train.num_isotypes,
        num_v_families=ds_train.num_v_families,
        num_light_loci=ds_train.num_light_loci,
    )
    denoiser = Denoiser(d_cfg, cond_drop_prob=args.cond_drop).to(device)
    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        cfg=DiffusionConfig(
            latent_dim=ae.cfg.latent_dim,
            loss_type=args.loss_type,
            self_cond_prob=args.self_cond_prob,
            cfg_weight=args.cfg_weight,
        ),
    ).to(device)

    if state.is_distributed:
        # find_unused_parameters=True: self-cond branch and CFG null-class
        # embeddings are touched inconsistently across iterations
        denoiser = nn.parallel.DistributedDataParallel(
            denoiser, device_ids=[state.local_rank], find_unused_parameters=True,
        )
        diffusion.denoiser = denoiser
    if args.compile:
        denoiser = torch.compile(denoiser)
        diffusion.denoiser = denoiser

    if is_main(state):
        print(f"[denoiser] {sum(p.numel() for p in unwrap(denoiser).parameters()):,} parameters")

    # optimizer (no weight decay on norms / biases / embeddings)
    decay, no_decay = [], []
    for n, p in unwrap(denoiser).named_parameters():
        if p.ndim < 2 or "norm" in n.lower() or "bias" in n.lower() or "emb" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.999),
    )

    ema = EMA(unwrap(denoiser), decay=args.ema_decay)

    scaler = torch.amp.GradScaler(enabled=(args.amp == "fp16"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": torch.float32}[args.amp]

    # resume
    step = 0
    if args.resume is not None and args.resume.exists():
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrap(denoiser).load_state_dict(sd["denoiser"])
        if "ema" in sd:
            ema.load_state_dict(sd["ema"])
        if "optim" in sd:
            optim.load_state_dict(sd["optim"])
        step = sd.get("step", 0)
        if is_main(state):
            print(f"[resume] from {args.resume} @ step={step}")

    t0 = time.time()
    epoch = 0
    data_iter = iter(dl)
    denoiser.train()
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(dl)
            batch = next(data_iter)

        src  = batch["source_tokens"].to(device, non_blocking=True)
        iso  = batch["iso"].to(device,  non_blocking=True)
        vfam = batch["vfam"].to(device, non_blocking=True)
        loc  = batch["loc"].to(device,  non_blocking=True)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(args.amp != "off")):
            x0 = ae.encode(src)

        lr_scale = cosine_decay_schedule(step, args.warmup, args.steps)
        for g in optim.param_groups:
            g["lr"] = args.lr * lr_scale

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(args.amp != "off")):
            loss = diffusion.loss(x0.to(amp_dtype), iso, vfam, loc)
            loss = loss.float()

        if args.amp == "fp16":
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(unwrap(denoiser).parameters(), args.grad_clip)
            scaler.step(optim); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unwrap(denoiser).parameters(), args.grad_clip)
            optim.step()

        ema.update(unwrap(denoiser))

        if step % args.log_every == 0 and is_main(state):
            loss_r = all_reduce_mean(loss.detach(), state)
            dt = time.time() - t0
            print(f"step {step:>6d} | loss {loss_r.item():.4f} | lr {args.lr*lr_scale:.2e} | {dt:.1f}s")

        step += 1

        if (step % args.ckpt_every == 0 or step == args.steps) and is_main(state):
            ckpt = {
                "step": step,
                "denoiser": unwrap(denoiser).state_dict(),
                "ema": ema.state_dict(),
                "optim": optim.state_dict(),
                "denoiser_cfg": d_cfg.__dict__,
                "diffusion_cfg": diffusion.cfg.__dict__,
                "ae_ckpt": str(args.ae_ckpt),
                "args": vars(args),
            }
            torch.save(ckpt, args.out / f"diffusion_{step}.pt")
            torch.save(ckpt, args.out / "diffusion_latest.pt")

    ddp_teardown(state)


if __name__ == "__main__":
    main()