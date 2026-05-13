"""
Tier-2 evaluation: foldability via IgFold pLDDT (DPLM-2 style metric).

We fold each generated antibody (or a subsample of them) using IgFold and
report:
  - mean pLDDT across all residues
  - mean pLDDT in CDR regions (where novelty matters)
  - %  of structures with mean pLDDT > 70 (threshold from DPLM-2 paper)

For the antibody domain IgFold is ~25 sec/structure on H100 and is more
accurate than ESMFold on antibodies because it was trained on PDB
antibody-only structures with paired heavy + light input.

Subsample: by default 50 sequences per cell × 18 cells = 900 sequences
total. ~6-7 hours on 1× H100.

Install:
    pip install igfold

Usage:
    python scripts/eval_foldability.py \
        --samples-dir <data-dir>/samples \
        --out-dir     <data-dir>/foldability \
        --n-per-cell  50 \
        --report      <data-dir>/eval_reports/foldability.json
"""


import argparse
import json
import re
import time
from pathlib import Path

LINKER_RE = re.compile(r"GGGGSGGGGS")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    out, header, body = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    out.append((header, "".join(body)))
                header = line[1:]
                body = []
            else:
                body.append(line)
        if header is not None:
            out.append((header, "".join(body)))
    return out


def split_pair(seq: str) -> tuple[str, str] | None:
    """Split VH+linker+VL by GGGGSGGGGS. Returns (VH, VL) or None."""
    m = LINKER_RE.search(seq)
    if m is None:
        return None
    return seq[: m.start()], seq[m.end():]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--n-per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cells", type=int, default=18)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # ----- Monkey-patch torch.load BEFORE importing igfold -----
    # PyTorch 2.6+ defaults to weights_only=True which blocks IgFold's
    # checkpoint loading (it pickles BertConfig from antiberty). We trust
    # IgFold's own packaged checkpoints, so disable the safety check.
    import torch
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    print("[setup] patched torch.load to weights_only=False (for IgFold ckpts)")

    # Lazy import of IgFold so the script can be inspected without it
    print("[setup] importing igfold ...")
    try:
        from igfold import IgFoldRunner
    except ImportError:
        print("[setup] IgFold not installed. Run: pip install igfold")
        return

    print("[setup] initializing IgFold (downloads weights on first run)...")
    igfold = IgFoldRunner()
    print("[setup] ready.")

    import random
    rng = random.Random(args.seed)

    per_cell_results = {}
    total_folded = 0
    fastas = sorted(args.samples_dir.glob("*.fasta"))[: args.max_cells]

    t0 = time.time()
    for fasta in fastas:
        cell = fasta.stem
        cell_outdir = args.out_dir / cell
        cell_outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== cell {cell} ===")
        recs = read_fasta(fasta)

        # Filter to ones we can split into VH/VL
        usable = []
        for header, seq in recs:
            split = split_pair(seq)
            if split is None:
                continue
            vh, vl = split
            if 80 <= len(vh) <= 140 and 80 <= len(vl) <= 130:
                usable.append((header, vh, vl))
        rng.shuffle(usable)
        sample = usable[: args.n_per_cell]
        print(f"  usable: {len(usable)}/{len(recs)}, folding {len(sample)}")

        plddts_all = []
        plddts_cdr = []
        for i, (header, vh, vl) in enumerate(sample):
            try:
                out_pdb = cell_outdir / f"{i:03d}.pdb"
                # IgFold accepts a dict of {chain_name: seq}
                result = igfold.fold(
                    str(out_pdb),
                    sequences={"H": vh, "L": vl},
                    do_renum=False,
                    do_refine=False,
                )
                # `result.prmsd` is per-residue predicted RMSD; convert to pLDDT-like 0-100.
                # IgFold doesn't output pLDDT directly; we use 100*exp(-prmsd) as proxy
                # following common practice (see IgFold paper / DPLM-2 supplementary).
                import math
                prmsd = result.prmsd.cpu().numpy().flatten()
                plddt_est = [100.0 * math.exp(-r) for r in prmsd]
                mean_plddt = sum(plddt_est) / len(plddt_est) if plddt_est else float("nan")
                plddts_all.append(mean_plddt)

                # Heuristic CDR identification: positions 23-32, 50-65, 95-110
                # (Kabat-style approximate). We don't run ANARCI here; this is
                # rough for monitoring purposes.
                cdr_indices = list(range(23, 33)) + list(range(50, 66)) + list(range(95, 111))
                cdr_indices = [i for i in cdr_indices if i < len(plddt_est)]
                if cdr_indices:
                    cdr_mean = sum(plddt_est[i] for i in cdr_indices) / len(cdr_indices)
                    plddts_cdr.append(cdr_mean)
            except Exception as e:
                print(f"    fold failed for {header}: {e}")
                continue

            if (i + 1) % 10 == 0:
                dt = time.time() - t0
                print(f"  [{cell}] {i+1}/{len(sample)} done  "
                      f"latest plddt={mean_plddt:.1f}  cum {dt:.0f}s")

        n = len(plddts_all)
        mean = sum(plddts_all) / max(1, n)
        share70 = sum(1 for p in plddts_all if p > 70) / max(1, n)
        cdr_mean = sum(plddts_cdr) / max(1, len(plddts_cdr)) if plddts_cdr else float("nan")
        per_cell_results[cell] = {
            "n_folded": n,
            "mean_plddt": mean,
            "share_plddt_above_70": share70,
            "mean_cdr_plddt": cdr_mean,
            "plddt_distribution": plddts_all,
        }
        total_folded += n
        print(f"  cell summary  mean pLDDT={mean:.1f}  >70={share70*100:.0f}%  CDR mean={cdr_mean:.1f}")

    # Aggregate
    all_plddt = [p for c in per_cell_results.values() for p in c["plddt_distribution"]]
    all_cdr = [c["mean_cdr_plddt"] for c in per_cell_results.values()
               if c["mean_cdr_plddt"] == c["mean_cdr_plddt"]]   # filter NaN

    aggregate = {
        "total_folded": total_folded,
        "n_per_cell_target": args.n_per_cell,
        "overall_mean_plddt": sum(all_plddt) / max(1, len(all_plddt)),
        "overall_share_above_70": sum(1 for p in all_plddt if p > 70) / max(1, len(all_plddt)),
        "overall_mean_cdr_plddt": sum(all_cdr) / max(1, len(all_cdr)) if all_cdr else float("nan"),
        "elapsed_seconds": time.time() - t0,
        "per_cell": per_cell_results,
    }
    with open(args.report, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n=== overall mean pLDDT: {aggregate['overall_mean_plddt']:.1f}  "
          f"share>70: {aggregate['overall_share_above_70']*100:.0f}% ===")
    print(f"[done] {total_folded} folded structures in {time.time()-t0:.0f}s")
    print(f"[done] wrote {args.report}")


if __name__ == "__main__":
    main()
