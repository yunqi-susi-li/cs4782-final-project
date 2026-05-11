"""
Tier-1.6 — Perplexity comparison: LD4LG (AE recon) vs DPLM (masked-token).

For LD4LG: we report autoencoder reconstruction perplexity, defined as
exp(val_cross_entropy) of the trained AE on a held-out val set. This is
the natural perplexity for an encoder-decoder reconstruction task.

For DPLM: we report masked-token perplexity, defined as exp(CE) when the
model predicts randomly-masked tokens at p=0.15 (BERT-style) on val set.

These are different evaluation protocols (reconstruction vs MLM) and
their absolute values are not directly comparable, but both measure how
well each model fits the held-out antibody distribution. We report both
in the spirit of the original LD4LG Table 2 (which itself uses a
different reference protocol — GPT-2-Large embedding — that we don't
have set up for the protein domain).

Usage:
    python scripts/eval_perplexity.py \
        --data /mnt/.../ab_ld4lg/processed \
        --ae-ckpt   /mnt/.../runs/ae/autoencoder_latest.pt \
        --dplm-ckpt /mnt/.../runs/dplm/dplm_latest.pt \
        --out       /mnt/.../eval_reports/perplexity.json \
        --n-samples 5000
"""


import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from code.diffusion.LD4LG.autoencoder import AutoencoderConfig, LanguageAutoencoder
from code.diffusion.LD4LG.data import PairedAntibodyDataset, make_collate_fn
from code.diffusion.LD4LG.tokenizer import AATokenizer
from code.diffusion.DPLM.model import DPLM, DPLMConfig


def evaluate_ae_perplexity(args, device, dl):
    print("\n=== LD4LG Autoencoder reconstruction perplexity ===")
    sd = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    cfg = AutoencoderConfig(**sd["cfg"])
    ae = LanguageAutoencoder(cfg).to(device)
    ae.load_state_dict(sd["model"])
    ae.eval()
    print(f"[ae] loaded {args.ae_ckpt}")

    total_ce = 0.0
    total_count = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            src = batch["source_tokens"].to(device, non_blocking=True)
            di = batch["decoder_input"].to(device, non_blocking=True)
            tgt = batch["target_tokens"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                _, logits = ae(src, di, tgt)
            # Per-token CE, ignore pad
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                tgt.view(-1),
                reduction="none",
            ).view(tgt.shape)
            mask = tgt.ne(0)  # not pad
            total_ce += (ce * mask.float()).sum().item()
            total_count += mask.sum().item()
            if (bi + 1) % 20 == 0:
                running = total_ce / max(1, total_count)
                print(f"  batch {bi+1}/{len(dl)}  running CE = {running:.4f}")

    avg_ce = total_ce / max(1, total_count)
    ppl = math.exp(avg_ce)
    print(f"[ae] {total_count} tokens scored in {time.time()-t0:.1f}s")
    print(f"[ae] avg CE = {avg_ce:.4f}   →   reconstruction perplexity = {ppl:.4f}")
    return {"ae_recon_ce": avg_ce, "ae_recon_perplexity": ppl,
            "ae_n_tokens": total_count}


def evaluate_dplm_perplexity(args, device, dl, mask_prob=0.15, mask_id=24):
    print(f"\n=== DPLM masked-token perplexity (mask p={mask_prob}) ===")
    sd = torch.load(args.dplm_ckpt, map_location="cpu", weights_only=False)
    cfg = DPLMConfig(**sd["cfg_model"])
    model = DPLM(cfg, cond_drop_prob=0.0).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    print(f"[dplm] loaded {args.dplm_ckpt}")

    total_ce = 0.0
    total_count = 0
    rng = torch.Generator(device=device).manual_seed(0)
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            tokens = batch["source_tokens"].to(device, non_blocking=True)
            iso = batch["iso"].to(device, non_blocking=True)
            vfam = batch["vfam"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            B, L = tokens.shape

            # Randomly mask 15% of non-special-token positions
            keep = (
                tokens.eq(0)            # pad
                | tokens.eq(1)          # bos
                | tokens.eq(2)          # eos
            )
            rand = torch.rand(B, L, device=device, generator=rng)
            mask = (rand < mask_prob) & ~keep
            # ensure at least one masked per row
            if (mask.sum(dim=1) == 0).any():
                # force first non-special position
                for b in (mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]:
                    cand = (~keep[b]).nonzero(as_tuple=True)[0]
                    if cand.numel() > 0:
                        mask[b, cand[0]] = True

            corrupted = torch.where(mask, torch.full_like(tokens, mask_id), tokens)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                logits = model(corrupted, iso, vfam, loc)
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                tokens.view(-1),
                reduction="none",
            ).view(tokens.shape)
            total_ce += (ce * mask.float()).sum().item()
            total_count += mask.sum().item()
            if (bi + 1) % 20 == 0:
                running = total_ce / max(1, total_count)
                print(f"  batch {bi+1}/{len(dl)}  running CE = {running:.4f}")

    avg_ce = total_ce / max(1, total_count)
    ppl = math.exp(avg_ce)
    print(f"[dplm] {total_count} masked tokens in {time.time()-t0:.1f}s")
    print(f"[dplm] avg CE = {avg_ce:.4f}   →   masked perplexity = {ppl:.4f}")
    return {"dplm_masked_ce": avg_ce, "dplm_masked_perplexity": ppl,
            "dplm_n_tokens": total_count, "mask_prob": mask_prob}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--dplm-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"[device] {device}")

    ds = PairedAntibodyDataset(args.data, args.split)
    n = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))
    sub = Subset(ds, list(range(n)))
    tok = AATokenizer()
    collate = make_collate_fn(tok.pad_id, tok.bos_id)
    dl = DataLoader(sub, batch_size=args.batch, shuffle=False, collate_fn=collate)
    print(f"[data] {args.split} split: {n}/{len(ds)} sequences, {len(dl)} batches")

    ae_result = evaluate_ae_perplexity(args, device, dl)
    dplm_result = evaluate_dplm_perplexity(args, device, dl)

    out_data = {
        "split": args.split,
        "n_sequences": n,
        **ae_result,
        **dplm_result,
        "comment": "AE recon perplexity (LD4LG) and masked-token perplexity (DPLM) "
                   "are different evaluation protocols (full reconstruction vs MLM "
                   "with p=0.15 masking) and should not be directly numerically "
                   "compared. Both indicate how well the trained model fits the "
                   "held-out antibody distribution.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)

    print("\n" + "="*60)
    print(f"=== SUMMARY ===")
    print(f"LD4LG AE   reconstruction perplexity:  {ae_result['ae_recon_perplexity']:.4f}")
    print(f"DPLM       masked-token  perplexity:  {dplm_result['dplm_masked_perplexity']:.4f}")
    print("="*60)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
