# Sample from a trained latent diffusion model and decode to AA sequences
import argparse
from pathlib import Path
import torch
from .autoencoder import AutoencoderConfig, LanguageAutoencoder
from .denoiser import Denoiser, DenoiserConfig
from .diffusion import DiffusionConfig, GaussianDiffusion
from .data import ISOTYPES, V_FAMILIES, LIGHT_LOCI
from .tokenizer import AATokenizer


def _label_to_idx(label, table):
    """Real label -> integer id; falls back to 'Other' if unknown.
    The CFG null index is table length and is never returned"""
    return table.index(label) if label in table else table.index("Other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--diff-ckpt", type=Path, required=True)
    ap.add_argument("--iso", default="IGHG", help="isotype: IGHG, IGHM, IGHA, IGHD, IGHE, Bulk")
    ap.add_argument("--vfam", default="IGHV3", help="V-family: IGHV1..IGHV7")
    ap.add_argument("--loc", default="K", help="light-chain locus: K or L")
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--max-decode", type=int, default=288)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # autoencoder
    sd_ae = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    ae = LanguageAutoencoder(AutoencoderConfig(**sd_ae["cfg"])).to(device)
    ae.load_state_dict(sd_ae["model"])
    ae.eval()

    # denoiser + diffusion. EMA weights preferred at sample time.
    sd_d = torch.load(args.diff_ckpt, map_location="cpu", weights_only=False)
    d_cfg = DenoiserConfig(**sd_d["denoiser_cfg"])
    denoiser = Denoiser(d_cfg, cond_drop_prob=0.0).to(device)
    denoiser.load_state_dict(sd_d["ema"] if "ema" in sd_d else sd_d["denoiser"])
    denoiser.eval()

    diff_cfg = DiffusionConfig(**sd_d["diffusion_cfg"])
    diff_cfg.cfg_weight = args.cfg
    diff_cfg.num_sampling_steps = args.steps
    diffusion = GaussianDiffusion(denoiser=denoiser, cfg=diff_cfg).to(device)

    # resolve string labels to ids
    iso_idx  = _label_to_idx(args.iso,  ISOTYPES)
    vfam_idx = _label_to_idx(args.vfam, V_FAMILIES)
    loc_idx  = _label_to_idx(args.loc,  LIGHT_LOCI)
    print(f"[cond] iso={args.iso}({iso_idx}) vfam={args.vfam}({vfam_idx}) loc={args.loc}({loc_idx})")

    iso  = torch.full((args.num,), iso_idx,  dtype=torch.long, device=device)
    vfam = torch.full((args.num,), vfam_idx, dtype=torch.long, device=device)
    loc  = torch.full((args.num,), loc_idx,  dtype=torch.long, device=device)

    # sample latents -> decode
    with torch.no_grad():
        x = diffusion.sample(
            batch_size=args.num,
            latent_len=d_cfg.latent_len,
            latent_dim=d_cfg.latent_dim,
            iso=iso, vfam=vfam, loc=loc,
            device=device,
            num_steps=args.steps,
            cfg_weight=args.cfg,
        )
        ids = ae.generate_from_latent(x, max_len=args.max_decode)

    tok = AATokenizer()
    seqs = [tok.decode(row.tolist()) for row in ids]

    with open(args.out, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">sample_{i}|iso={args.iso}|vfam={args.vfam}|loc={args.loc}|cfg={args.cfg}\n")
            f.write(s + "\n")
    print(f"[done] wrote {args.num} sequences to {args.out}")


if __name__ == "__main__":
    main()