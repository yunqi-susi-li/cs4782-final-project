#!/usr/bin/env python3
"""Extract paired-chain sequences from a pickle into per-chain FASTA + metadata TSV.

Receptor-agnostic: takes --chain1 and --chain2 column suffixes (e.g. heavy/light,
beta/alpha). Output uses fixed internal names "heavy"/"light" so the downstream
scripts can stay generic.
"""
import argparse
import pathlib
import sys

import pandas as pd


def write_fasta(ids: pd.Series, seqs: pd.Series, path: pathlib.Path) -> int:
    mask = seqs.notna() & (seqs.str.len() > 0)
    lines = ">" + ids[mask].astype(str) + "\n" + seqs[mask].astype(str) + "\n"
    with open(path, "w") as f:
        f.write("".join(lines.values))
    return int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input pickle path")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--chain1", required=True,
                    help="Chain 1 column suffix (e.g. heavy, beta)")
    ap.add_argument("--chain2", required=True,
                    help="Chain 2 column suffix (e.g. light, alpha)")
    args = ap.parse_args()

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...", flush=True)
    df = pd.read_pickle(args.input)
    print(f"  Shape: {df.shape}", flush=True)

    c1, c2 = args.chain1, args.chain2

    required = [
        f"sequence_id_{c1}", f"sequence_id_{c2}",
        f"sequence_alignment_aa_{c1}", f"sequence_alignment_aa_{c2}",
        f"cdr3_aa_{c1}", f"cdr3_aa_{c2}",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing required columns: {missing}\n"
                 f"Available columns matching '_{c1}'/'_{c2}': "
                 f"{[c for c in df.columns if c.endswith('_'+c1) or c.endswith('_'+c2)][:20]}")

    df["row_index"] = range(len(df))
    df["pair_id"] = (
        "row" + df["row_index"].astype(str)
        + "|C1:" + df[f"sequence_id_{c1}"].astype(str)
        + "|C2:" + df[f"sequence_id_{c2}"].astype(str)
    )

    # Strip allele suffix from V/D/J calls (e.g. IGHV3-23*01 -> IGHV3-23).
    for col in [f"v_call_{c1}", f"d_call_{c1}", f"j_call_{c1}",
                f"v_call_{c2}", f"d_call_{c2}", f"j_call_{c2}"]:
        if col in df.columns:
            df[col + "_gene"] = df[col].astype(str).str.split("*").str[0]

    # Normalize sequence columns; drop gap / stop characters that some
    # numbering schemes insert.
    df["heavy_full_seq"] = (
        df[f"sequence_alignment_aa_{c1}"].astype(str)
          .str.replace(r"[.\-*]", "", regex=True)
    )
    df["light_full_seq"] = (
        df[f"sequence_alignment_aa_{c2}"].astype(str)
          .str.replace(r"[.\-*]", "", regex=True)
    )
    df["pair_full_seq"] = df["heavy_full_seq"] + "GGGGSGGGGS" + df["light_full_seq"]
    df["heavy_cdr3_seq"] = (
        df[f"cdr3_aa_{c1}"].astype(str)
          .str.replace(r"[.\-*]", "", regex=True)
    )
    df["light_cdr3_seq"] = (
        df[f"cdr3_aa_{c2}"].astype(str)
          .str.replace(r"[.\-*]", "", regex=True)
    )

    df["heavy_len"] = df["heavy_full_seq"].str.len()
    df["light_len"] = df["light_full_seq"].str.len()
    df["pair_len"] = df["pair_full_seq"].str.len()
    df["heavy_cdr3_len"] = df["heavy_cdr3_seq"].str.len()
    df["light_cdr3_len"] = df["light_cdr3_seq"].str.len()

    print("Writing FASTA files ...", flush=True)
    for chain_name, col in [("heavy", "heavy_full_seq"),
                             ("light", "light_full_seq"),
                             ("pair",  "pair_full_seq")]:
        n = write_fasta(df["pair_id"], df[col], out / f"{chain_name}_full.fasta")
        print(f"  {chain_name}_full.fasta: {n} sequences", flush=True)

    for chain_name, col in [("heavy", "heavy_cdr3_seq"),
                             ("light", "light_cdr3_seq")]:
        n = write_fasta(df["pair_id"], df[col], out / f"{chain_name}_cdr3.fasta")
        print(f"  {chain_name}_cdr3.fasta: {n} sequences", flush=True)

    print("Writing metadata.tsv ...", flush=True)

    meta_cols = [
        "row_index", "pair_id",
        f"sequence_id_{c1}", f"sequence_id_{c2}",
    ]
    for c in [c1, c2]:
        if f"productive_{c}" in df.columns:
            meta_cols.append(f"productive_{c}")
    for c in [c1, c2]:
        for g in ["v_call", "d_call", "j_call"]:
            for suffix in ["", "_gene"]:
                col = f"{g}_{c}{suffix}"
                if col in df.columns:
                    meta_cols.append(col)
    # Pass through any annotation columns that happen to exist
    # (BCR: Isotype/locus/shm_rate; TCR or other: nothing extra).
    for col in ["Isotype_heavy", "Isotype_light",
                "locus_heavy", "locus_light",
                "shm_rate_heavy", "shm_rate_light", "species"]:
        if col in df.columns and col not in meta_cols:
            meta_cols.append(col)
    for c in [c1, c2]:
        for col in [f"cdr3_aa_{c}", f"junction_aa_{c}", f"junction_aa_length_{c}"]:
            if col in df.columns:
                meta_cols.append(col)

    meta_cols += [
        "heavy_full_seq", "light_full_seq", "pair_full_seq",
        "heavy_len", "light_len", "pair_len",
        "heavy_cdr3_seq", "light_cdr3_seq",
        "heavy_cdr3_len", "light_cdr3_len",
    ]
    seen = set()
    meta_cols = [c for c in meta_cols if c in df.columns and not (c in seen or seen.add(c))]

    df[meta_cols].to_csv(out / "metadata.tsv", sep="\t", index=False)

    summary = pd.DataFrame({
        "output": ["heavy_full", "light_full", "pair_full", "heavy_cdr3", "light_cdr3"],
        "n_sequences": [len(df)] * 5,
    })
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    print(f"\nDone. Output in {out}", flush=True)


if __name__ == "__main__":
    main()
