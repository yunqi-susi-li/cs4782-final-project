"""
Offline evaluation for generated antibody FASTA files.
Pure-python checks (no ANARCI / IgBLAST / pretrained models). For
rigorous V-gene assignment or CDR3 extraction, run ANARCI separately.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from .tokenizer import AATokenizer


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
LINKER_RE = re.compile(r"GGGGSGGGGS")


def read_fasta(path):
    seqs = []
    cur_header, cur_seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_header is not None:
                    seqs.append((cur_header, "".join(cur_seq_parts)))
                cur_header = line[1:]
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)
        if cur_header is not None:
            seqs.append((cur_header, "".join(cur_seq_parts)))
    return seqs


def pct(values, q):
    if not values:
        return 0
    arr = sorted(values)
    return int(arr[int(q * (len(arr) - 1))])


def ngram_diversity(seqs, n):
    total, unique = 0, set()
    for s in seqs:
        if len(s) < n:
            continue
        grams = [s[i:i + n] for i in range(len(s) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total > 0 else 0.0


def fingerprint(s):
    """SHA1 of the sequence -- cheap exact-match set membership."""
    return hashlib.sha1(s.encode()).hexdigest()


def decode_training_sequences(tokens_path, meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    N, L = meta["n"], meta["max_len"]
    tokens = np.memmap(tokens_path, dtype=np.int16, mode="r", shape=(N, L))
    tok = AATokenizer()
    return [tok.decode(tokens[i].tolist()) for i in range(N)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--train-tokens", type=Path, default=None)
    ap.add_argument("--train-meta", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--near-match-hamming", type=int, default=3,
                    help="max Hamming distance counted as near-match (same-length only)")
    ap.add_argument("--near-match-sample", type=int, default=2000,
                    help="cap training-set comparison size for speed; 0 = no cap")
    args = ap.parse_args()

    recs = read_fasta(args.fasta)
    seqs = [s for _, s in recs]
    print(f"[eval] loaded {len(seqs)} sequences from {args.fasta}")

    # per-sequence sanity + structural checks
    aa_valid   = [all(c in CANONICAL_AA for c in s) and len(s) > 0 for s in seqs]
    has_linker = [bool(LINKER_RE.search(s)) for s in seqs]

    vh_lens, vl_lens, pair_lens = [], [], []
    for s, ok_linker in zip(seqs, has_linker):
        pair_lens.append(len(s))
        if ok_linker:
            m = LINKER_RE.search(s)
            vh_lens.append(m.start())
            vl_lens.append(len(s) - m.end())

    metrics = {
        "n_total": len(seqs),
        "n_valid_aa": int(sum(aa_valid)),
        "share_valid_aa":      float(np.mean(aa_valid))   if seqs else 0.0,
        "share_linker_present": float(np.mean(has_linker)) if seqs else 0.0,
        "length_stats_pair": {
            "min": pct(pair_lens, 0.0),
            "p10": pct(pair_lens, 0.1),
            "p50": pct(pair_lens, 0.5),
            "p90": pct(pair_lens, 0.9),
            "p99": pct(pair_lens, 0.99),
            "max": pct(pair_lens, 1.0),
        },
        "length_stats_vh": {"p50": pct(vh_lens, 0.5), "p90": pct(vh_lens, 0.9)} if vh_lens else {},
        "length_stats_vl": {"p50": pct(vl_lens, 0.5), "p90": pct(vl_lens, 0.9)} if vl_lens else {},
        "div_2gram": ngram_diversity(seqs, 2),
        "div_3gram": ngram_diversity(seqs, 3),
        "div_4gram": ngram_diversity(seqs, 4),
    }

    # exact + near match against training set (memorization check)
    if args.train_tokens and args.train_meta:
        print("[eval] decoding training set for memorization check...")
        train_seqs = decode_training_sequences(args.train_tokens, args.train_meta)
        train_fp = {fingerprint(s): None for s in train_seqs}
        exact = sum(1 for s in seqs if fingerprint(s) in train_fp)
        metrics["exact_match_train"]       = exact
        metrics["share_exact_match_train"] = exact / max(1, len(seqs))

        # near-match (Hamming) -- same-length only, optionally on a subsample
        subset = train_seqs
        if args.near_match_sample > 0 and len(subset) > args.near_match_sample:
            rng = np.random.default_rng(0)
            subset = [train_seqs[i] for i in rng.choice(len(train_seqs), args.near_match_sample, replace=False)]
        by_len = {}
        for s in subset:
            by_len.setdefault(len(s), []).append(s)
        near = 0
        for s in seqs:
            for t in by_len.get(len(s), []):
                if sum(a != b for a, b in zip(s, t)) <= args.near_match_hamming:
                    near += 1
                    break
        metrics["near_match_hamming"]      = args.near_match_hamming
        metrics["near_match_sample"]       = args.near_match_sample
        metrics["share_near_match_train"]  = near / max(1, len(seqs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()