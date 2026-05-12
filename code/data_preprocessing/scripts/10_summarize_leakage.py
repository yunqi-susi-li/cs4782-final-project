#!/usr/bin/env python3
"""Identity-percentile summary of an MMseqs2 audit m8 file."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLS = [
    "query", "target", "pident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_m8", type=Path)
    ap.add_argument("output_prefix", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.input_m8, sep="\t", header=None, names=COLS)
    if df.empty:
        raise SystemExit("Input m8 is empty")

    top = (
        df.sort_values(["query", "bits", "evalue", "pident"],
                       ascending=[True, False, True, False])
        .groupby("query", as_index=False)
        .head(1)
        .copy()
    )
    top.to_csv(f"{args.output_prefix}.top_hits.tsv", sep="\t", index=False)

    summary = {
        "n_queries_with_hit": int(top.shape[0]),
        "median_pident": float(top["pident"].median()),
        "p90_pident": float(top["pident"].quantile(0.9)),
        "p95_pident": float(top["pident"].quantile(0.95)),
        "p99_pident": float(top["pident"].quantile(0.99)),
        "frac_ge_100": float((top["pident"] >= 100.0).mean()),
        "frac_ge_99": float((top["pident"] >= 99.0).mean()),
        "frac_ge_95": float((top["pident"] >= 95.0).mean()),
        "frac_ge_90": float((top["pident"] >= 90.0).mean()),
        "frac_ge_80": float((top["pident"] >= 80.0).mean()),
    }
    pd.Series(summary).to_csv(f"{args.output_prefix}.summary.tsv", sep="\t", header=False)
    print(pd.Series(summary).to_string())
    print(f"\nWrote {args.output_prefix}.top_hits.tsv and .summary.tsv")


if __name__ == "__main__":
    main()
