"""
One-off preprocessing: raw OAS metadata.tsv -> int16 token memmaps.

Two input modes:
  (A) --archives FILE [FILE...]   tar.gz archives containing
                                  06_export/{train,val,test}/{split}.metadata.tsv
  (B) --tsv-dir DIR               a directory with {split}.metadata.tsv directly

Output (written to --out):
  {split}.tokens.npy   int16  (N, max_len)
  {split}.lengths.npy  int32  (N,)
  {split}.iso.npy      int8   (N,)
  {split}.vfam.npy     int8   (N,)
  {split}.loc.npy      int8   (N,)
  {split}.meta.json

Examples:
  python -m LD4LG.preprocess --archives run_090_export.tar.gz \\
      --out /path/processed --max-len 288
  python -m LD4LG.preprocess --tsv-dir /path/to/tsvs \\
      --out /path/processed --max-len 288
"""

import argparse
import csv
import json
import tarfile
import tempfile
from pathlib import Path
import numpy as np
from .tokenizer import AATokenizer
from .data import (
    ISOTYPES, V_FAMILIES, LIGHT_LOCI,
    _isotype_to_idx, _vfam_to_idx, _locus_to_idx,
)


def extract_metadata(archives, workdir):
    """Pull metadata.tsv files out of one or more tar.gz archives. Returns {split: [paths]}."""
    paths = {"train": [], "val": [], "test": []}
    for archive in archives:
        print(f"[extract] {archive}")
        with tarfile.open(archive, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith("metadata.tsv")]
            for m in members:
                tf.extract(m, path=workdir)
                for split in paths:
                    if f"/{split}/" in m.name:
                        out = workdir / m.name
                        paths[split].append(out)
                        print(f"  -> {split}: {out}")
                        break
    return paths


def count_valid_rows(tsv_paths, max_len, seq_col):
    n = 0
    for p in tsv_paths:
        with open(p) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seq = row.get(seq_col, "") or ""
                if not seq or len(seq) + 2 > max_len:   # +2 for <bos>/<eos>
                    continue
                n += 1
    return n


def preprocess_split(tsv_paths, out_dir, split, max_len, tokenizer, seq_col="pair_input_seq"):
    print(f"[{split}] counting valid rows...")
    n = count_valid_rows(tsv_paths, max_len, seq_col)
    if n == 0:
        print(f"[{split}] no valid rows -- skipping")
        return
    print(f"[{split}] will write {n} rows to {out_dir}")

    tokens  = np.memmap(out_dir / f"{split}.tokens.npy",  dtype=np.int16, mode="w+", shape=(n, max_len))
    lengths = np.memmap(out_dir / f"{split}.lengths.npy", dtype=np.int32, mode="w+", shape=(n,))
    iso     = np.memmap(out_dir / f"{split}.iso.npy",     dtype=np.int8,  mode="w+", shape=(n,))
    vfam    = np.memmap(out_dir / f"{split}.vfam.npy",    dtype=np.int8,  mode="w+", shape=(n,))
    loc     = np.memmap(out_dir / f"{split}.loc.npy",     dtype=np.int8,  mode="w+", shape=(n,))

    # pre-fill with pad_id so trailing positions are valid
    tokens[:] = tokenizer.pad_id

    i = 0
    for p in tsv_paths:
        with open(p) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seq = row.get(seq_col, "") or ""
                if not seq or len(seq) + 2 > max_len:
                    continue
                ids = tokenizer.encode(seq)   # adds <bos>/<eos>
                L = len(ids)
                tokens[i, :L] = ids
                lengths[i] = L
                iso[i]  = _isotype_to_idx(row.get("Isotype_heavy", ""))
                vfam[i] = _vfam_to_idx(row.get("v_call_heavy_gene", ""))
                loc[i]  = _locus_to_idx(row.get("locus_light", ""))
                i += 1
                if i % 100_000 == 0:
                    print(f"  [{split}] {i}/{n}")

    for arr in (tokens, lengths, iso, vfam, loc):
        arr.flush()

    meta = {
        "n": n,
        "max_len": max_len,
        "num_isotypes":   len(ISOTYPES),     # excl. null slot
        "num_v_families": len(V_FAMILIES),
        "num_light_loci": len(LIGHT_LOCI),
        "isotypes":   ISOTYPES,
        "v_families": V_FAMILIES,
        "light_loci": LIGHT_LOCI,
        "seq_col": seq_col,
    }
    with open(out_dir / f"{split}.meta.json", "w") as fm:
        json.dump(meta, fm, indent=2)

    print(f"[{split}] done: {i} rows")


def collect_tsv_dir(tsv_dir):
    """Mode B: look for {split}.metadata.tsv directly under tsv_dir."""
    out = {}
    for split in ("train", "val", "test"):
        p = tsv_dir / f"{split}.metadata.tsv"
        out[split] = [p] if p.exists() else []
        if not p.exists():
            print(f"[tsv-dir] note: {p.name} not in {tsv_dir} -- {split} skipped")
    return out


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--archives", nargs="+", type=Path,
                     help="one or more tar.gz archives, e.g. run_090_export.tar.gz")
    grp.add_argument("--tsv-dir", type=Path,
                     help="directory holding train/val/test.metadata.tsv directly")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=288)
    ap.add_argument("--seq-col", default="pair_input_seq",
                    help="TSV column to tokenize")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AATokenizer()

    if args.tsv_dir is not None:
        split2paths = collect_tsv_dir(args.tsv_dir)
        for split in args.splits:
            preprocess_split(
                split2paths.get(split, []), args.out, split,
                max_len=args.max_len, tokenizer=tokenizer, seq_col=args.seq_col,
            )
    else:
        # mode A: extract tar.gz to a temp dir first
        with tempfile.TemporaryDirectory(prefix="ab_ld4lg_extract_") as tmp:
            split2paths = extract_metadata(args.archives, Path(tmp))
            for split in args.splits:
                preprocess_split(
                    split2paths.get(split, []), args.out, split,
                    max_len=args.max_len, tokenizer=tokenizer, seq_col=args.seq_col,
                )

    print(f"[done] wrote memmaps to {args.out}")


if __name__ == "__main__":
    main()