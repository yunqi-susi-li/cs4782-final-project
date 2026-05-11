"""
Sample from a trained DPLM. Same CLI shape as LD4LG.sample so the
existing eval pipeline (eval.py, eval_foldability.py, eval_hmmer.py)
works without changes.
"""

import argparse
from pathlib import Path
import torch
from .model import DPLM, DPLMConfig
from .diffusion import DPLMDiffusion, DPLMDiffusionConfig
from .tokenizer import DPLMTokenizer
from ..LD4LG.data import ISOTYPES, V_FAMILIES, LIGHT_LOCI


def label_to_idx(label, table):
    return table.index(label) if label in table else table.index("Other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--iso", default="IGHG")
    ap.add_argument("--vfam", default="IGHV3")
    ap.add_argument("--loc", default="K")
    ap.add_argument("--num", type=int, default=512)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=243)   # median paired length
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    # Decoding controls. Released DPLM-1 codebase defaults to greedy, which
    # collapses on antibody data; the paper specifies stochastic. Y.L.
    ap.add_argument("--sample-mode", choices=["stochastic", "greedy"],
                    default="stochastic",
                    help="'stochastic' (paper recipe) or 'greedy' (ablation only)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="nucleus / top-p filter; 1.0 disables")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_model = DPLMConfig(**sd["cfg_model"])
    cfg_diff = DPLMDiffusionConfig(**sd["cfg_diffusion"])
    cfg_diff.cfg_weight = args.cfg
    cfg_diff.num_sampling_steps = args.steps

    model = DPLM(cfg_model, cond_drop_prob=0.0).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    diffusion = DPLMDiffusion(model=model, cfg=cfg_diff).to(device)

    iso_idx  = label_to_idx(args.iso,  ISOTYPES)
    vfam_idx = label_to_idx(args.vfam, V_FAMILIES)
    loc_idx  = label_to_idx(args.loc,  LIGHT_LOCI)
    print(f"[cond]   iso={args.iso}({iso_idx}) vfam={args.vfam}({vfam_idx}) loc={args.loc}({loc_idx})")
    print(f"[sample] mode={args.sample_mode}  T={args.temperature}  top_p={args.top_p}")

    tok = DPLMTokenizer()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_seqs = []
    n_remaining = args.num
    seq_idx = 0
    while n_remaining > 0:
        bs = min(args.batch, n_remaining)
        iso_t  = torch.full((bs,), iso_idx,  dtype=torch.long, device=device)
        vfam_t = torch.full((bs,), vfam_idx, dtype=torch.long, device=device)
        loc_t  = torch.full((bs,), loc_idx,  dtype=torch.long, device=device)

        with torch.no_grad():
            tokens = diffusion.sample(
                batch_size=bs, seq_len=args.seq_len,
                iso=iso_t, vfam=vfam_t, loc=loc_t,
                device=device,
                num_steps=args.steps, cfg_weight=args.cfg,
                temperature=args.temperature, top_p=args.top_p,
                sample_mode=args.sample_mode,
            )
        for i in range(bs):
            all_seqs.append(tok.decode(tokens[i].tolist()))
            seq_idx += 1
        n_remaining -= bs
        print(f"  generated {seq_idx}/{args.num}")

    with open(args.out, "w") as f:
        for i, seq in enumerate(all_seqs):
            f.write(
                f">sample_{i}|iso={args.iso}|vfam={args.vfam}|loc={args.loc}"
                f"|cfg={args.cfg}|mode={args.sample_mode}"
                f"|T={args.temperature}|topp={args.top_p}|model=DPLM\n"
            )
            f.write(seq + "\n")
    print(f"[done] wrote {args.num} sequences to {args.out}")


if __name__ == "__main__":
    main()