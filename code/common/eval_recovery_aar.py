"""
Tier-1.5 evaluation: autoencoder reconstruction Amino Acid Recovery (AAR).

The closest analogue in our setup to a DPLM-style "recovery rate":

    held-out test sequence  →  AE.encode  →  latent x  →  AE.decode  →  predicted tokens
                                                                              │
                                                       compute per-position match
                                                                              │
                                                                       AAR = correct / total

Unlike DPLM-1 (which fills in masked tokens), our model has no masking.
Recovery here means "encode-decode round-trip fidelity", which is the
relevant measure for a latent-diffusion autoencoder: it tells us how much
information the (32, 64) latent retains about the input sequence.

This script uses TEACHER-FORCED forward passes (consistent with how
val cross-entropy is reported during training). Greedy autoregressive
decoding gives a lower number due to error compounding; teacher-forcing
is the standard reporting convention and matches val_loss.

Usage (CPU is fine, takes ~2 min for 5000 sequences):
    python scripts/eval_recovery_aar.py \
        --data <data-dir>/ab_ld4lg/processed \
        --ae-ckpt <data-dir>/runs/ae/autoencoder_latest.pt \
        --out  <data-dir>/eval_reports/recovery_aar.json \
        --split test \
        --n-samples 5000
"""


import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from code.diffusion.LD4LG.autoencoder import AutoencoderConfig, LanguageAutoencoder
from code.diffusion.LD4LG.data import PairedAntibodyDataset, make_collate_fn
from code.diffusion.LD4LG.tokenizer import AATokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-samples", type=int, default=5000,
                    help="how many sequences to evaluate; 0 = all")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"[device] {device}")

    # Load AE
    sd = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    cfg = AutoencoderConfig(**sd["cfg"])
    ae = LanguageAutoencoder(cfg).to(device)
    ae.load_state_dict(sd["model"])
    ae.eval()
    print(f"[model] loaded from {args.ae_ckpt}")
    print(f"[model] latent shape = ({cfg.latent_len}, {cfg.latent_dim})")

    # Dataset
    ds = PairedAntibodyDataset(args.data, args.split, max_len=cfg.max_source_len)
    n = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))
    print(f"[data] {args.split} split: using {n}/{len(ds)} sequences")

    tok = AATokenizer()
    collate = make_collate_fn(tok.pad_id, tok.bos_id)

    # Subsample
    indices = list(range(n))
    sub = torch.utils.data.Subset(ds, indices)
    dl = DataLoader(sub, batch_size=args.batch, shuffle=False, collate_fn=collate)

    # Per-cell accumulators (keyed by (iso, vfam, loc) tuple)
    overall_correct = 0
    overall_total = 0
    per_cell = {}                      # cell_key -> (correct, total)
    per_position_correct = [0] * cfg.max_target_len
    per_position_total = [0] * cfg.max_target_len

    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            src = batch["source_tokens"].to(device)
            di  = batch["decoder_input"].to(device)
            tgt = batch["target_tokens"].to(device)
            iso = batch["iso"].tolist()
            vfam = batch["vfam"].tolist()
            loc = batch["loc"].tolist()

            # Forward (teacher-forced) — same path as training val_loss
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                _, logits = ae(src, di, tgt)        # logits (B, L, V)
            preds = logits.argmax(dim=-1)           # (B, L)

            mask = tgt.ne(tok.pad_id)               # (B, L) — ignore pad positions
            correct = (preds == tgt) & mask         # (B, L) bool

            # Per-position
            cor_pos = correct.sum(dim=0).cpu().tolist()
            tot_pos = mask.sum(dim=0).cpu().tolist()
            for i in range(cfg.max_target_len):
                per_position_correct[i] += cor_pos[i]
                per_position_total[i] += tot_pos[i]

            # Per-sequence + per-cell
            n_correct_per_seq = correct.sum(dim=1).cpu().tolist()
            n_total_per_seq = mask.sum(dim=1).cpu().tolist()
            for k in range(len(n_correct_per_seq)):
                key = (int(iso[k]), int(vfam[k]), int(loc[k]))
                if key not in per_cell:
                    per_cell[key] = [0, 0]
                per_cell[key][0] += n_correct_per_seq[k]
                per_cell[key][1] += n_total_per_seq[k]
                overall_correct += n_correct_per_seq[k]
                overall_total += n_total_per_seq[k]

            if (bi + 1) % 20 == 0:
                running = overall_correct / max(1, overall_total)
                print(f"  batch {bi+1}/{len(dl)} -- running AAR {running:.4f}")

    overall_aar = overall_correct / max(1, overall_total)

    # Per-cell summary
    from code.diffusion.LD4LG.data import ISOTYPES, V_FAMILIES, LIGHT_LOCI
    per_cell_named = {}
    for (iso_idx, vfam_idx, loc_idx), (cor, tot) in per_cell.items():
        iso_name = ISOTYPES[iso_idx] if iso_idx < len(ISOTYPES) else f"iso_{iso_idx}"
        vfam_name = V_FAMILIES[vfam_idx] if vfam_idx < len(V_FAMILIES) else f"vfam_{vfam_idx}"
        loc_name = LIGHT_LOCI[loc_idx] if loc_idx < len(LIGHT_LOCI) else f"loc_{loc_idx}"
        cell_name = f"{iso_name}_{vfam_name}_{loc_name}"
        per_cell_named[cell_name] = {
            "n_correct": cor,
            "n_total": tot,
            "aar": cor / max(1, tot),
        }

    # Per-position curve
    per_position_aar = [
        (per_position_correct[i] / per_position_total[i] if per_position_total[i] > 0 else None)
        for i in range(cfg.max_target_len)
    ]

    out = {
        "split": args.split,
        "n_sequences": n,
        "overall_aar": overall_aar,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "per_cell_aar": per_cell_named,
        "per_position_aar_first_50": per_position_aar[:50],
        "elapsed_seconds": time.time() - t0,
        "ae_ckpt": str(args.ae_ckpt),
        "comment": "Teacher-forced AAR; matches val cross-entropy convention. "
                   "Greedy autoregressive AAR (with error compounding) would be slightly lower.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== Overall test-set AAR: {overall_aar*100:.2f}% "
          f"({overall_correct}/{overall_total}) ===")
    print(f"Elapsed: {time.time()-t0:.1f}s")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
