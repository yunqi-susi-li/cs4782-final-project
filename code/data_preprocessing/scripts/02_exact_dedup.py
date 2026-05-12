#!/usr/bin/env python3
"""Exact deduplication of paired rows on (heavy_full_seq, light_full_seq).

Keeps one representative per unique pair (lowest row_index) and records the
exact-duplicate count.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def write_fasta(ids: Iterable[str], seqs: Iterable[str], path: Path) -> int:
    n = 0
    with path.open("w") as f:
        for seq_id, seq in zip(ids, seqs):
            if not seq:
                continue
            f.write(f">{seq_id}\n{seq}\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata_tsv", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--sort-by", choices=["pair_id", "row_index", "exact_dup_count"],
                    default="row_index")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metadata_tsv, sep="\t")

    required = ["pair_id", "row_index", "heavy_full_seq", "light_full_seq", "pair_full_seq"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    pair_concat = df["heavy_full_seq"].astype(str) + "\t" + df["light_full_seq"].astype(str)
    df["exact_pair_sha1"] = pair_concat.map(sha1_text)

    grp = df.groupby("exact_pair_sha1", sort=False, dropna=False)
    counts = grp.size().rename("exact_dup_count")
    rep_idx = grp["row_index"].idxmin()
    reps = df.loc[rep_idx].copy()
    reps = reps.merge(counts, left_on="exact_pair_sha1", right_index=True,
                      how="left", validate="1:1")
    reps["exact_rep_pair_id"] = reps["pair_id"]

    map_df = df[["pair_id", "row_index", "exact_pair_sha1"]].merge(
        reps[["exact_pair_sha1", "exact_rep_pair_id", "exact_dup_count"]],
        on="exact_pair_sha1",
        how="left",
        validate="m:1",
    )

    reps = reps.sort_values(["row_index", "pair_id"]).copy()

    reps.to_csv(args.outdir / "metadata.exact.tsv", sep="\t", index=False)
    map_df.to_csv(args.outdir / "exact_pair_map.tsv", sep="\t", index=False)

    write_fasta(reps["pair_id"], reps["heavy_full_seq"], args.outdir / "heavy_full.exact.fasta")
    write_fasta(reps["pair_id"], reps["light_full_seq"], args.outdir / "light_full.exact.fasta")
    write_fasta(reps["pair_id"], reps["pair_full_seq"],  args.outdir / "pair_full.exact.fasta")

    if "heavy_cdr3_seq" in reps.columns:
        write_fasta(reps["pair_id"], reps["heavy_cdr3_seq"].fillna(""),
                    args.outdir / "heavy_cdr3.exact.fasta")
    if "light_cdr3_seq" in reps.columns:
        write_fasta(reps["pair_id"], reps["light_cdr3_seq"].fillna(""),
                    args.outdir / "light_cdr3.exact.fasta")

    summary = pd.Series({
        "n_input_rows": int(df.shape[0]),
        "n_exact_unique_pairs": int(reps.shape[0]),
        "exact_reduction_fraction": float(1.0 - reps.shape[0] / max(df.shape[0], 1)),
        "max_exact_dup_count": int(reps["exact_dup_count"].max()),
        "median_exact_dup_count": float(reps["exact_dup_count"].median()),
    })
    summary.to_csv(args.outdir / "exact_dedup.summary.tsv", sep="\t", header=False)
    print(summary.to_string())
    print(f"\nWrote outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
