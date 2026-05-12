#!/usr/bin/env python3
"""Group rows by chain-cluster pairs and split into train / val / test.

Two grouping modes:
  --group-mode tuple   group key = (chain1_cluster, chain2_cluster)
  --group-mode union   union-find across both chains (transitive closure)

The split unit is the group; entire groups are assigned to one of
train/val/test. Greedy assignment by largest deficit fills each split toward
its target fraction.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

SUMMARY_INSPECT_COLS = [
    "run_id",
    "Isotype_heavy", "Isotype_light",
    "locus_heavy", "locus_light",
    "v_call_heavy", "v_call_light",
    "v_call_beta", "v_call_alpha",
    "species",
]


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def load_member_to_rep(cluster_tsv: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with cluster_tsv.open() as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            rep, member = row[0], row[1]
            mapping[member] = rep
    return mapping


def greedy_group_split(
    group_sizes: Dict[str, int],
    train_frac: float, val_frac: float, test_frac: float,
    seed: int,
) -> Dict[str, str]:
    rng = random.Random(seed)
    items = list(group_sizes.items())
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True)

    total = sum(group_sizes.values())
    targets = {"train": train_frac * total, "val": val_frac * total, "test": test_frac * total}
    current = {"train": 0, "val": 0, "test": 0}
    assign: Dict[str, str] = {}

    for gid, gsize in items:
        deficits = {
            split: (targets[split] - current[split]) / max(targets[split], 1e-8)
            for split in ["train", "val", "test"]
        }
        best_split = sorted(deficits.items(),
                            key=lambda kv: (-kv[1], current[kv[0]], kv[0]))[0][0]
        assign[gid] = best_split
        current[best_split] += gsize
    return assign


def summarize_splits(df: pd.DataFrame, out_prefix: Path) -> None:
    rows = []
    for split, sub in df.groupby("split"):
        row = {
            "split": split,
            "n_rows": int(sub.shape[0]),
            "n_tuple_groups": int(sub["tuple_group_id"].nunique()),
            "n_component_groups": int(sub["component_group_id"].nunique()),
            "sum_exact_dup_count": int(
                sub.get("exact_dup_count", pd.Series([1] * len(sub))).sum()
            ),
        }
        for col in SUMMARY_INSPECT_COLS:
            if col in sub.columns:
                vc = sub[col].fillna("<NA>").value_counts(dropna=False)
                row[f"top_{col}"] = vc.index[0]
                row[f"top_{col}_fraction"] = float(vc.iloc[0] / vc.sum())
        rows.append(row)
    pd.DataFrame(rows).sort_values("split").to_csv(
        f"{out_prefix}.split_summary.tsv", sep="\t", index=False
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata_exact_tsv", type=Path)
    ap.add_argument("heavy_cluster_tsv", type=Path)
    ap.add_argument("light_cluster_tsv", type=Path)
    ap.add_argument("out_prefix", type=Path)
    ap.add_argument("--group-mode", choices=["tuple", "union"], default="union")
    ap.add_argument("--train-frac", type=float, default=0.96)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--test-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frac_sum = args.train_frac + args.val_frac + args.test_frac
    if abs(frac_sum - 1.0) > 1e-8:
        raise SystemExit(f"Split fractions must sum to 1.0, got {frac_sum}")

    meta = pd.read_csv(args.metadata_exact_tsv, sep="\t")
    required = ["pair_id", "row_index"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise SystemExit(f"Missing required metadata columns: {missing}")

    heavy_map = load_member_to_rep(args.heavy_cluster_tsv)
    light_map = load_member_to_rep(args.light_cluster_tsv)

    meta["heavy_cluster_rep"] = meta["pair_id"].map(lambda x: heavy_map.get(x, x))
    meta["light_cluster_rep"] = meta["pair_id"].map(lambda x: light_map.get(x, x))
    meta["tuple_group_id"] = (
        meta["heavy_cluster_rep"].astype(str)
        + "||"
        + meta["light_cluster_rep"].astype(str)
    )

    pair_ids: List[str] = meta["pair_id"].astype(str).tolist()
    uf = UnionFind(len(pair_ids))

    heavy_owner: Dict[str, int] = {}
    light_owner: Dict[str, int] = {}
    for i, (hrep, lrep) in enumerate(zip(meta["heavy_cluster_rep"],
                                          meta["light_cluster_rep"])):
        hrep = str(hrep)
        lrep = str(lrep)
        if hrep in heavy_owner:
            uf.union(i, heavy_owner[hrep])
        else:
            heavy_owner[hrep] = i
        if lrep in light_owner:
            uf.union(i, light_owner[lrep])
        else:
            light_owner[lrep] = i

    root_to_members: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(pair_ids)):
        root_to_members[uf.find(i)].append(i)

    component_id_by_idx: Dict[int, str] = {}
    for root, members in root_to_members.items():
        rep_pair_id = pair_ids[root]
        cid = f"CC|{rep_pair_id}"
        for i in members:
            component_id_by_idx[i] = cid

    meta["component_group_id"] = [component_id_by_idx[i] for i in range(len(pair_ids))]
    meta["tuple_group_size"] = meta.groupby("tuple_group_id")["pair_id"].transform("size")
    meta["component_group_size"] = meta.groupby("component_group_id")["pair_id"].transform("size")

    split_unit_col = "component_group_id" if args.group_mode == "union" else "tuple_group_id"
    meta["split_unit_id"] = meta[split_unit_col]
    group_sizes = meta.groupby("split_unit_id")["pair_id"].size().to_dict()
    split_assign = greedy_group_split(
        group_sizes, args.train_frac, args.val_frac, args.test_frac, args.seed
    )
    meta["split"] = meta["split_unit_id"].map(split_assign)

    tuple_rep_idx = meta.groupby("tuple_group_id")["row_index"].idxmin()
    component_rep_idx = meta.groupby("component_group_id")["row_index"].idxmin()
    meta["is_tuple_representative"] = False
    meta.loc[tuple_rep_idx, "is_tuple_representative"] = True
    meta["is_component_representative"] = False
    meta.loc[component_rep_idx, "is_component_representative"] = True

    meta = meta.sort_values(["split", "row_index", "pair_id"]).copy()
    meta.to_csv(f"{args.out_prefix}.assignments.tsv", sep="\t", index=False)
    summarize_splits(meta, args.out_prefix)

    summary = pd.Series({
        "group_mode": args.group_mode,
        "n_rows": int(meta.shape[0]),
        "n_tuple_groups": int(meta["tuple_group_id"].nunique()),
        "n_component_groups": int(meta["component_group_id"].nunique()),
        "train_rows": int((meta["split"] == "train").sum()),
        "val_rows": int((meta["split"] == "val").sum()),
        "test_rows": int((meta["split"] == "test").sum()),
    })
    summary.to_csv(f"{args.out_prefix}.summary.tsv", sep="\t", header=False)
    print(summary.to_string())
    print(f"\nWrote {args.out_prefix}.assignments.tsv")


if __name__ == "__main__":
    main()
