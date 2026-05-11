"""
Grid sweep over DPLM sampling hyperparameters (temperature x top-p).
Find a (T, top_p) combo that maximizes per-cell 4-gram diversity while
preserving validity (>=99% canonical AA + linker present). The DPLM-1
default (T=1.0, top_p=0.95) is a generic protein recommendation and
turns out to be suboptimal on the antibody domain. 
"""

import argparse
import itertools
import json
import re
import time
from collections import Counter
from pathlib import Path
import torch
from .diffusion import DPLMDiffusion, DPLMDiffusionConfig
from .model import DPLM, DPLMConfig
from .tokenizer import DPLMTokenizer
from ..LD4LG.data import ISOTYPES, V_FAMILIES, LIGHT_LOCI


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
LINKER_RE = re.compile(r"GGGGSGGGGS")


def metrics_for_seqs(seqs, n_gram=4):
    """Lightweight inline metrics: validity, linker, n-gram diversity."""
    n = len(seqs)
    n_valid = sum(1 for s in seqs if s and set(s).issubset(CANONICAL_AA))
    n_linker = sum(1 for s in seqs if LINKER_RE.search(s))

    # n-gram diversity pooled across all seqs in this cell
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
    """'IGHM_IGHV1_K' -> ('IGHM', 'IGHV1', 'K')"""
    parts = cell.split("_")
    if len(parts) != 3:
        raise ValueError(f"bad cell name {cell}: expected ISO_VFAM_LOC")
    return parts[0], parts[1], parts[2]


def label_to_idx(label, table):
    return table.index(label) if label in table else table.index("Other")


def sample_one_cell(diffusion, tok, cell, n, seq_len, cfg_w, steps,
                    temperature, top_p, sample_mode, device, seed):
    iso, vfam, loc = parse_cell(cell)
    iso_t  = torch.full((n,), label_to_idx(iso,  ISOTYPES),    dtype=torch.long, device=device)
    vfam_t = torch.full((n,), label_to_idx(vfam, V_FAMILIES),  dtype=torch.long, device=device)
    loc_t  = torch.full((n,), label_to_idx(loc,  LIGHT_LOCI),  dtype=torch.long, device=device)

    torch.manual_seed(seed)
    with torch.no_grad():
        tokens = diffusion.sample(
            batch_size=n, seq_len=seq_len,
            iso=iso_t, vfam=vfam_t, loc=loc_t,
            device=device,
            num_steps=steps, cfg_weight=cfg_w,
            temperature=temperature, top_p=top_p,
            sample_mode=sample_mode,
        )
    return [tok.decode(tokens[i].tolist()) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="path to dplm_latest.pt")
    ap.add_argument("--out", type=Path, required=True, help="path to write sweep JSON")
    ap.add_argument("--cells", type=str,
                    default="IGHM_IGHV1_K,IGHG_IGHV1_K,IGHA_IGHV1_K",
                    help="comma-sep cells (one per isotype is enough)")
    ap.add_argument("--n-per-config", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=243)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperatures", type=str, default="0.7,0.85,1.0,1.15,1.3")
    ap.add_argument("--top-ps", type=str, default="0.85,0.9,0.95,0.99")
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    Ts = [float(x) for x in args.temperatures.split(",")]
    top_ps = [float(x) for x in args.top_ps.split(",")]
    configs = list(itertools.product(Ts, top_ps))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[grid]   {len(Ts)} T x {len(top_ps)} top_p = {len(configs)} configs")
    print(f"[cells]  {cells}")
    print(f"[total]  {len(configs) * len(cells) * args.n_per_config} sequences\n")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_model = DPLMConfig(**sd["cfg_model"])
    cfg_diff = DPLMDiffusionConfig(**sd["cfg_diffusion"])
    cfg_diff.cfg_weight = args.cfg
    cfg_diff.num_sampling_steps = args.steps
    model = DPLM(cfg_model, cond_drop_prob=0.0).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    diffusion = DPLMDiffusion(model=model, cfg=cfg_diff).to(device)
    tok = DPLMTokenizer()
    print(f"[ckpt]   loaded {args.ckpt}")

    results = []
    t0 = time.time()
    for ci, (T, top_p) in enumerate(configs):
        print(f"\n--- config {ci+1}/{len(configs)}: T={T}  top_p={top_p} ---")
        per_cell = {}
        for cell in cells:
            seqs = sample_one_cell(
                diffusion, tok, cell,
                n=args.n_per_config, seq_len=args.seq_len,
                cfg_w=args.cfg, steps=args.steps,
                temperature=T, top_p=top_p,
                sample_mode="stochastic",
                device=device, seed=args.seed,
            )
            m = metrics_for_seqs(seqs)
            per_cell[cell] = m
            print(f"  {cell}: div_4gram={m['div_4gram']:.4f}  "
                  f"valid={m['share_valid_aa']:.3f}  linker={m['share_linker']:.3f}")

        mean_div    = sum(per_cell[c]["div_4gram"]      for c in cells) / len(cells)
        mean_valid  = sum(per_cell[c]["share_valid_aa"] for c in cells) / len(cells)
        mean_linker = sum(per_cell[c]["share_linker"]   for c in cells) / len(cells)
        score = mean_div * mean_valid * mean_linker

        # per-isotype split (for diagnosing whether IGHM is being collapsed harder than IGHG/A)
        def iso_mean(prefix):
            xs = [per_cell[c]["div_4gram"] for c in cells if c.startswith(prefix)]
            return sum(xs) / len(xs) if xs else None

        results.append({
            "temperature": T, "top_p": top_p,
            "mean_div_4gram": mean_div,
            "mean_valid": mean_valid,
            "mean_linker": mean_linker,
            "score": score,
            "ighm_div": iso_mean("IGHM"),
            "ighg_div": iso_mean("IGHG"),
            "igha_div": iso_mean("IGHA"),
            "per_cell": per_cell,
            "elapsed_s": time.time() - t0,
        })

        # checkpoint after every config so a crash doesn't lose the sweep
        with open(args.out, "w") as f:
            json.dump({
                "configs_tested": ci + 1,
                "configs_total":  len(configs),
                "cells": cells,
                "n_per_config": args.n_per_config,
                "results": results,
            }, f, indent=2)

    # final ranked table
    print("\n" + "=" * 78)
    print(f"{'rank':<5} {'T':<6} {'top_p':<7} {'div':<8} {'valid':<8} "
          f"{'link':<8} {'score':<10} {'IGHM/IGHG ratio':<18}")
    print("-" * 78)
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    for rank, r in enumerate(ranked, 1):
        ratio = "n/a"
        if r["ighm_div"] is not None and r["ighg_div"] not in (None, 0):
            ratio = f"{r['ighm_div'] / r['ighg_div']:.2f}"
        print(f"{rank:<5} {r['temperature']:<6} {r['top_p']:<7} "
              f"{r['mean_div_4gram']:.4f}  {r['mean_valid']:.3f}    "
              f"{r['mean_linker']:.3f}    {r['score']:.5f}    {ratio:<18}")
    print("=" * 78)
    print(f"\n[done] {len(configs)} configs in {time.time() - t0:.1f}s")
    print(f"[done] full results written to {args.out}")


if __name__ == "__main__":
    main()