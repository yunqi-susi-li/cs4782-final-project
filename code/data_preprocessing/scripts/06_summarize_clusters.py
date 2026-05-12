#!/usr/bin/env python3
"""Per-cluster size + purity summary for an MMseqs2 cluster TSV.

The columns inspected for purity are auto-detected: any metadata column whose
name appears in a default list (gene calls, isotype, locus, run id, CDR3 length).
Additional columns can be passed via --inspect.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INSPECT_COLS = [
    "v_call_heavy", "j_call_heavy", "v_call_light", "j_call_light",
    "v_call_beta",  "j_call_beta",  "v_call_alpha", "j_call_alpha",
    "Isotype_heavy", "Isotype_light", "locus_heavy", "locus_light",
    "run_id", "species",
    "heavy_cdr3_len", "light_cdr3_len",
]


def purity(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    vc = series.fillna("<NA>").value_counts(dropna=False)
    return vc.iloc[0] / vc.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata_tsv", type=Path)
    ap.add_argument("cluster_tsv", type=Path)
    ap.add_argument("out_prefix", type=Path)
    ap.add_argument("--inspect", nargs="*", default=None,
                    help="Override the list of columns to inspect for purity.")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    clu = pd.read_csv(args.cluster_tsv, sep="\t", header=None, names=["rep", "member"])

    merged = clu.merge(meta, left_on="member", right_on="pair_id",
                       how="left", validate="m:1")
    grp = merged.groupby("rep", dropna=False)

    stats = pd.DataFrame({
        "cluster_id": list(grp.groups.keys()),
        "cluster_size": grp.size().values,
    })

    inspect_cols = args.inspect if args.inspect is not None else DEFAULT_INSPECT_COLS
    for col in inspect_cols:
        if col in merged.columns:
            stats[f"{col}_purity"] = grp[col].apply(purity).values
            stats[f"{col}_n_unique"] = grp[col].nunique(dropna=False).values

    stats = stats.sort_values("cluster_size", ascending=False)
    stats.to_csv(f"{args.out_prefix}.cluster_stats.tsv", sep="\t", index=False)

    summary = {
        "n_clusters": int(stats.shape[0]),
        "n_members": int(stats["cluster_size"].sum()),
        "reduction_fraction": float(1.0 - stats.shape[0] / max(stats["cluster_size"].sum(), 1)),
        "singleton_fraction": float((stats["cluster_size"] == 1).mean()),
        "median_cluster_size": float(stats["cluster_size"].median()),
        "p90_cluster_size": float(stats["cluster_size"].quantile(0.9)),
        "p99_cluster_size": float(stats["cluster_size"].quantile(0.99)),
        "max_cluster_size": int(stats["cluster_size"].max()),
    }
    for col in [c for c in stats.columns if c.endswith("_purity")]:
        summary[f"median_{col}"] = float(stats[col].median())
        summary[f"p10_{col}"] = float(stats[col].quantile(0.1))

    pd.Series(summary).to_csv(f"{args.out_prefix}.summary.tsv", sep="\t", header=False)
    print(pd.Series(summary).to_string())
    print(f"\nWrote {args.out_prefix}.cluster_stats.tsv and {args.out_prefix}.summary.tsv")


if __name__ == "__main__":
    main()
