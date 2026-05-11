"""
LD4LG CFG-weight sweep. Mirror of DPLM/sampling_sweep.py but for the
continuous-latent track. Traces LD4LG's quality-diversity Pareto curve
by varying classifier-free guidance weight w at sample time (no retraining).

    python -m code.diffusion.LD4LG.sampling_cfg_sweep \\
        --ae-ckpt   runs/ae/autoencoder_latest.pt \\
        --diff-ckpt runs/ld4lg/diffusion_latest.pt \\
        --out       results/ld4lg_cfg_sweep.json \\
        --n-per-config 128

Cost on H100: 5 w x 3 cells x 128 seqs x 250 DDPM steps ~ 6 min.
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch

from .autoencoder import AutoencoderConfig, LanguageAutoencoder
from .denoiser import Denoiser, DenoiserConfig
from .diffusion import DiffusionConfig, GaussianDiffusion
from .tokenizer import AATokenizer
from .data import ISOTYPES, V_FAMILIES, LIGHT_LOCI


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
LINKER_RE = re.compile(r"GGGGSGGGGS")


def metrics_for_seqs(seqs, n_gram=4):
    """Lightweight inline metrics: validity, linker, n-gram diversity."""
    n = len(seqs)
    n_valid = sum(1 for s in seqs if s and set(s).issubset(CANONICAL_AA))
    n_linker = sum(1 for s in seqs if LINKER_RE.search(s))
    counter = Counter()
    total = 0
    for s in seqs:
        if len(s) < n_gram:
            continue
        for i in range(len(s) - n_gram + 1):
            counter[s[i:i + n_gram]] += 1
            total += 1
    div = (len(counter) / total) if total > 0 else 0.0
    return {
        "n": n,
        "share_valid_aa": n_valid / max(1, n),
        "share_linker":   n_linker / max(1, n),
        "div_4gram":      div,
    }


def parse_cell(cell):
    parts = cell.split("_")
    if len(parts) != 3:
        raise ValueError(f"bad cell name {cell}: expected ISO_VFAM_LOC")
    return parts[0], parts[1], parts[2]


def label_to_idx(label, table):
    return table.index(label) if label in table else table.index("Other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--diff-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", default="IGHM_IGHV1_K,IGHG_IGHV1_K,IGHA_IGHV1_K")
    ap.add_argument("--weights", default="1.0,1.5,2.0,3.0,5.0")
    ap.add_argument("--n-per-config", type=int, default=128)
    ap.add_argument("--num-steps", type=int, default=250)
    ap.add_argument("--max-decode", type=int, default=288)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    ws = [float(x) for x in args.weights.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[grid]   {len(ws)} CFG x {len(cells)} cells = {len(ws)*len(cells)} configs")

    # load AE
    sd_ae = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    ae = LanguageAutoencoder(AutoencoderConfig(**sd_ae["cfg"])).to(device)
    ae.load_state_dict(sd_ae["model"])
    ae.eval()
    print(f"[ae]   loaded {args.ae_ckpt}")

    # load denoiser + diffusion
    sd_d = torch.load(args.diff_ckpt, map_location="cpu", weights_only=False)
    d_cfg = DenoiserConfig(**sd_d["denoiser_cfg"])
    denoiser = Denoiser(d_cfg, cond_drop_prob=0.0).to(device)
    denoiser.load_state_dict(sd_d["ema"] if "ema" in sd_d else sd_d["denoiser"])
    denoiser.eval()
    diff_cfg = DiffusionConfig(**sd_d["diffusion_cfg"])
    diff_cfg.num_sampling_steps = args.num_steps
    diffusion = GaussianDiffusion(denoiser=denoiser, cfg=diff_cfg).to(device)
    tok = AATokenizer()
    print(f"[diff] loaded {args.diff_ckpt}")

    results = []
    t0 = time.time()
    for wi, w in enumerate(ws):
        print(f"\n--- w={w} ({wi+1}/{len(ws)}) ---", flush=True)
        per_cell = {}
        for cell in cells:
            iso, vfam, loc = parse_cell(cell)
            n = args.n_per_config
            iso_t  = torch.full((n,), label_to_idx(iso,  ISOTYPES),    dtype=torch.long, device=device)
            vfam_t = torch.full((n,), label_to_idx(vfam, V_FAMILIES),  dtype=torch.long, device=device)
            loc_t  = torch.full((n,), label_to_idx(loc,  LIGHT_LOCI),  dtype=torch.long, device=device)
            torch.manual_seed(args.seed)
            with torch.no_grad():
                x = diffusion.sample(
                    batch_size=n,
                    latent_len=ae.cfg.latent_len,
                    latent_dim=ae.cfg.latent_dim,
                    iso=iso_t, vfam=vfam_t, loc=loc_t,
                    device=device,
                    num_steps=args.num_steps,
                    cfg_weight=w,
                )
                ids = ae.generate_from_latent(x, max_len=args.max_decode)
            seqs = [tok.decode(row.tolist()) for row in ids]
            m = metrics_for_seqs(seqs)
            per_cell[cell] = m
            print(f"  {cell}: div_4gram={m['div_4gram']:.4f}  "
                  f"valid={m['share_valid_aa']:.3f}  linker={m['share_linker']:.3f}", flush=True)

        mean_div    = sum(per_cell[c]["div_4gram"]      for c in cells) / len(cells)
        mean_valid  = sum(per_cell[c]["share_valid_aa"] for c in cells) / len(cells)
        mean_linker = sum(per_cell[c]["share_linker"]   for c in cells) / len(cells)

        results.append({
            "cfg_weight": w,
            "mean_div_4gram": mean_div,
            "mean_valid": mean_valid,
            "mean_linker": mean_linker,
            "per_cell": per_cell,
            "elapsed_s": time.time() - t0,
        })

        with open(args.out, "w") as f:
            json.dump({
                "configs_tested": wi + 1,
                "configs_total":  len(ws),
                "cells": cells,
                "weights": ws,
                "n_per_config": args.n_per_config,
                "results": results,
            }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{'w':<6} {'div':<10} {'valid':<10} {'linker':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['cfg_weight']:<6} {r['mean_div_4gram']:.4f}    {r['mean_valid']:.3f}    {r['mean_linker']:.3f}")
    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
