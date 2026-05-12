#!/usr/bin/env python3
"""Export final train/val/test tables and FASTA files from split assignments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def write_fasta(ids: Iterable[str], seqs: Iterable[str], path: Path) -> int:
    n = 0
    with path.open("w") as f:
        for seq_id, seq in zip(ids, seqs):
            if not seq:
                continue
            f.write(f">{seq_id}\n{seq}\n")
            n += 1
    return n


def build_pair_seq(df: pd.DataFrame, linker: str) -> pd.Series:
    return df["heavy_full_seq"].astype(str) + linker + df["light_full_seq"].astype(str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("assignments_tsv", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--collapse-level", choices=["none", "tuple", "component"],
                    default="tuple")
    ap.add_argument("--input-pickle", type=Path, default=None)
    ap.add_argument("--linker", default="GGGGSGGGGS")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.assignments_tsv, sep="\t")

    required = ["pair_id", "row_index", "split", "heavy_full_seq", "light_full_seq"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in assignments TSV: {missing}")

    if args.collapse_level == "tuple":
        if "is_tuple_representative" not in df.columns:
            raise SystemExit("assignments TSV lacks is_tuple_representative")
        out = df[df["is_tuple_representative"]].copy()
    elif args.collapse_level == "component":
        if "is_component_representative" not in df.columns:
            raise SystemExit("assignments TSV lacks is_component_representative")
        out = df[df["is_component_representative"]].copy()
    else:
        out = df.copy()

    out["pair_input_seq"] = build_pair_seq(out, args.linker)

    for split, sub in out.groupby("split"):
        split_dir = args.outdir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        sub = sub.sort_values("row_index").copy()
        sub.to_csv(split_dir / f"{split}.metadata.tsv", sep="\t", index=False)
        write_fasta(sub["pair_id"], sub["heavy_full_seq"],   split_dir / f"{split}.heavy.fasta")
        write_fasta(sub["pair_id"], sub["light_full_seq"],   split_dir / f"{split}.light.fasta")
        write_fasta(sub["pair_id"], sub["pair_input_seq"],   split_dir / f"{split}.pair.fasta")

    summary = pd.DataFrame([
        {
            "split": split,
            "n_rows": int(sub.shape[0]),
            "sum_exact_dup_count": (
                int(sub["exact_dup_count"].sum())
                if "exact_dup_count" in sub.columns else int(sub.shape[0])
            ),
        }
        for split, sub in out.groupby("split")
    ])
    summary.to_csv(args.outdir / "export.summary.tsv", sep="\t", index=False)

    if args.input_pickle is not None:
        full_df = pd.read_pickle(args.input_pickle).copy()
        full_df["row_index"] = full_df.index.astype(int)
        keep_cols = [c for c in out.columns if c not in full_df.columns or c == "row_index"]
        annot = out[keep_cols].copy()
        merged = full_df.merge(annot, on="row_index", how="inner", validate="1:1")
        for split, sub in merged.groupby("split"):
            split_dir = args.outdir / split
            sub.to_pickle(split_dir / f"{split}.pkl")

    print(summary.to_string(index=False))
    print(f"\nWrote split exports to: {args.outdir}")


if __name__ == "__main__":
    main()
