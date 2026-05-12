#!/usr/bin/env python3
"""Pick the top non-self hit per query from an MMseqs2 m8 file."""
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
    ap.add_argument("output_tsv", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.input_m8, sep="\t", header=None, names=COLS)
    df = df[df["query"] != df["target"]].copy()
    df = df.sort_values(["query", "bits", "evalue", "pident"],
                        ascending=[True, False, True, False])
    top = df.groupby("query", as_index=False).head(1).copy()
    top.to_csv(args.output_tsv, sep="\t", index=False)

    if top.empty:
        print("No non-self hits found.")
        return

    desc = top["pident"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(desc.to_string())
    print(f"\nWrote {args.output_tsv}")


if __name__ == "__main__":
    main()
