"""
One-off preprocessing script.

Two input modes:

  (A)  --archives FILE [FILE...]        tar.gz archives like run_090_export.tar.gz
       Each is expected to contain
         06_export/{train,val,test}/{split}.metadata.tsv

  (B)  --tsv-dir DIR                    a directory that already holds
         train.metadata.tsv  [val.metadata.tsv]  [test.metadata.tsv]

Output (written to --out):
  {split}.tokens.npy   int16  (N, max_len)
  {split}.lengths.npy  int32  (N,)
  {split}.iso.npy      int8   (N,)  [0, num_isotypes-1]       (exclude null slot)
  {split}.vfam.npy     int8   (N,)  [0, num_v_families-1]
  {split}.loc.npy      int8   (N,)  [0, num_light_loci-1]
  {split}.meta.json                {n, max_len, num_*}

Usage examples:
  # Mode A
  python scripts/preprocess.py \
      --archives /path/run_090_export.tar.gz \
      --out /path/processed --max-len 288

  # Mode B (no tar.gz extraction)
  python scripts/preprocess.py \
      --tsv-dir /mnt/beegfs/.../bio-diffusion \
      --out /mnt/beegfs/.../processed --max-len 288

Notes:
  - We stream the tsv line by line and write memmaps lazily so we never
    hold the full dataset in RAM.
  - Multiple archives are concatenated per-split.
  - We drop rows where `pair_input_seq` is empty or exceeds max-len.
  - If only train.metadata.tsv is present, val/test are simply skipped --
    the diffusion training loop only ever reads `train`, so this is fine.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

# Make the sibling src/ package importable when running directly.

from .tokenizer import AATokenizer              # noqa: E402
from .data import (                             # noqa: E402
    ISOTYPES, V_FAMILIES, LIGHT_LOCI,
    _isotype_to_idx, _vfam_to_idx, _locus_to_idx,
)


def extract_metadata(archives: list[Path], workdir: Path) -> dict[str, list[Path]]:
    """Extract metadata.tsv files from one or more tar.gz archives into workdir.

    Returns a mapping {split: [list of metadata.tsv paths]}.
    """
    paths = {"train": [], "val": [], "test": []}
    for archive in archives:
        print(f"[extract] {archive}")
        with tarfile.open(archive, "r:gz") as tf:
            members = [
                m for m in tf.getmembers()
                if m.name.endswith("metadata.tsv")
            ]
            for m in members:
                tf.extract(m, path=workdir)
                # Determine split from path.
                for split in paths:
                    if f"/{split}/" in m.name:
                        out = workdir / m.name
                        paths[split].append(out)
                        print(f"  -> {split}: {out}")
                        break
    return paths


def count_valid_rows(tsv_paths: list[Path], max_len: int, seq_col: str) -> int:
    n = 0
    for p in tsv_paths:
        with open(p, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seq = row.get(seq_col, "") or ""
                if not seq:
                    continue
                if len(seq) + 2 > max_len:  # +2 for <bos>/<eos>
                    continue
                n += 1
    return n


def preprocess_split(
    tsv_paths: list[Path],
    out_dir: Path,
    split: str,
    max_len: int,
    tokenizer: AATokenizer,
    seq_col: str = "pair_input_seq",
) -> None:
    print(f"[{split}] counting valid rows...")
    n = count_valid_rows(tsv_paths, max_len, seq_col)
    if n == 0:
        print(f"[{split}] no valid rows -- skipping")
        return
    print(f"[{split}] will write {n} rows to {out_dir}")

    tokens = np.memmap(out_dir / f"{split}.tokens.npy", dtype=np.int16, mode="w+", shape=(n, max_len))
    lengths = np.memmap(out_dir / f"{split}.lengths.npy", dtype=np.int32, mode="w+", shape=(n,))
    iso = np.memmap(out_dir / f"{split}.iso.npy", dtype=np.int8, mode="w+", shape=(n,))
    vfam = np.memmap(out_dir / f"{split}.vfam.npy", dtype=np.int8, mode="w+", shape=(n,))
    loc = np.memmap(out_dir / f"{split}.loc.npy", dtype=np.int8, mode="w+", shape=(n,))

    # Pre-fill tokens with pad_id so trailing positions are valid.
    tokens[:] = tokenizer.pad_id

    i = 0
    for p in tsv_paths:
        with open(p, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seq = row.get(seq_col, "") or ""
                if not seq:
                    continue
                if len(seq) + 2 > max_len:
                    continue
                ids = tokenizer.encode(seq)  # includes <bos> .. <eos>
                L = len(ids)
                tokens[i, :L] = ids
                lengths[i] = L
                iso[i] = _isotype_to_idx(row.get("Isotype_heavy", ""))
                vfam[i] = _vfam_to_idx(row.get("v_call_heavy_gene", ""))
                loc[i] = _locus_to_idx(row.get("locus_light", ""))
                i += 1
                if i % 100_000 == 0:
                    print(f"  [{split}] {i}/{n}")

    tokens.flush(); lengths.flush(); iso.flush(); vfam.flush(); loc.flush()

    meta = {
        "n": n,
        "max_len": max_len,
        "num_isotypes": len(ISOTYPES),         # excl. null slot
        "num_v_families": len(V_FAMILIES),
        "num_light_loci": len(LIGHT_LOCI),
        "isotypes": ISOTYPES,
        "v_families": V_FAMILIES,
        "light_loci": LIGHT_LOCI,
        "seq_col": seq_col,
    }
    with open(out_dir / f"{split}.meta.json", "w") as fm:
        json.dump(meta, fm, indent=2)

    print(f"[{split}] done: {i} rows")


def collect_tsv_dir(tsv_dir: Path) -> dict[str, list[Path]]:
    """Mode B: look for {split}.metadata.tsv directly under tsv_dir."""
    out: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        p = tsv_dir / f"{split}.metadata.tsv"
        out[split] = [p] if p.exists() else []
        if not p.exists():
            print(f"[tsv-dir] note: {p.name} not found in {tsv_dir} -- {split} split will be skipped")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--archives", nargs="+", type=Path,
                     help="One or more tar.gz archives like run_090_export.tar.gz")
    grp.add_argument("--tsv-dir", type=Path,
                     help="Directory holding train/val/test.metadata.tsv directly")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=288)
    ap.add_argument("--seq-col", default="pair_input_seq",
                    help="which TSV column to tokenize")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AATokenizer()

    if args.tsv_dir is not None:
        # Mode B: tsv files are already on disk; no extraction needed.
        split2paths = collect_tsv_dir(args.tsv_dir)
        for split in args.splits:
            preprocess_split(
                split2paths.get(split, []),
                args.out,
                split,
                max_len=args.max_len,
                tokenizer=tokenizer,
                seq_col=args.seq_col,
            )
    else:
        # Mode A: extract metadata.tsv files from tar.gz first.
        with tempfile.TemporaryDirectory(prefix="ab_ld4lg_extract_") as tmp:
            workdir = Path(tmp)
            split2paths = extract_metadata(args.archives, workdir)
            for split in args.splits:
                preprocess_split(
                    split2paths.get(split, []),
                    args.out,
                    split,
                    max_len=args.max_len,
                    tokenizer=tokenizer,
                    seq_col=args.seq_col,
                )

    print(f"[done] wrote memmaps to {args.out}")


if __name__ == "__main__":
    main()
