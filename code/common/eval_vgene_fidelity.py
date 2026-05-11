"""
Tier-1 evaluation: V-gene family fidelity via germline prefix matching.

Idea: each IGHV family has a highly conserved FR1 (framework region 1) — the
first ~25 amino acids of the heavy chain. We compute a consensus prefix per
family from the training set, then for each generated sequence we ask:
"which family's consensus is closest to its prefix?" and compare against the
family it was *conditioned* on.

This is a poor man's ANARCI: no install, no GPU, runs in ~30 minutes on CPU
on the full 9,216-sample evaluation set.

Usage:
    python scripts/eval_vgene_fidelity.py \
        --train-tsv /mnt/beegfs/.../bio-diffusion/train.metadata.tsv \
        --samples-dir /mnt/beegfs/.../ab_ld4lg/samples \
        --out /mnt/beegfs/.../ab_ld4lg/eval_reports/vgene_fidelity.json

Output: a JSON with per-cell fidelity + an overall fidelity number.
"""


import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PREFIX_LEN = 25                     # how many AAs of FR1 to use as fingerprint
MIN_FAMILY_SAMPLES = 100            # don't trust a consensus from <100 samples


def read_fasta(path: Path) -> list[str]:
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
        if cur:
            seqs.append("".join(cur))
    return seqs


def build_germline_consensus(train_tsv: Path, prefix_len: int = PREFIX_LEN) -> dict:
    """For each V-gene family, return a position-wise consensus sequence
    over the first `prefix_len` residues, plus a per-position frequency table
    used to score 'how typical' a prefix is."""
    print(f"[consensus] reading {train_tsv} ...")
    family_prefixes: dict[str, list[str]] = defaultdict(list)
    n = 0
    with open(train_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            v_gene = (row.get("v_call_heavy_gene", "") or "").strip()
            seq = (row.get("pair_input_seq", "") or "").strip()
            if not v_gene or not seq or len(seq) < prefix_len:
                continue
            # IGHV3-23 -> IGHV3
            family = v_gene.split("-")[0].split("*")[0].upper()
            if not family.startswith("IGHV"):
                continue
            family_prefixes[family].append(seq[:prefix_len])
            n += 1
            if n % 200_000 == 0:
                print(f"[consensus] {n} rows scanned")

    consensus = {}
    freq_tables = {}
    for fam, prefixes in family_prefixes.items():
        if len(prefixes) < MIN_FAMILY_SAMPLES:
            print(f"[consensus] skipping {fam} ({len(prefixes)} samples < {MIN_FAMILY_SAMPLES})")
            continue
        # Per-position frequency
        freqs = [Counter(p[i] for p in prefixes) for i in range(prefix_len)]
        consensus[fam] = "".join(c.most_common(1)[0][0] for c in freqs)
        # Normalize to probabilities for soft scoring
        freq_tables[fam] = [
            {aa: cnt / sum(c.values()) for aa, cnt in c.items()} for c in freqs
        ]
        print(f"[consensus] {fam}: n={len(prefixes)}  consensus={consensus[fam]}")
    return consensus, freq_tables


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def soft_log_likelihood(prefix: str, freq_table: list[dict]) -> float:
    """Sum of log-frequencies (Laplace-smoothed) at each position."""
    import math
    ll = 0.0
    for i, aa in enumerate(prefix[: len(freq_table)]):
        p = freq_table[i].get(aa, 0.0)
        ll += math.log(p + 1e-3)
    return ll


def classify(prefix: str, consensus: dict, freq_tables: dict) -> tuple[str, float, float]:
    """Return (best_family, hamming_to_best, log_likelihood_under_best)."""
    # Hard classification by hamming
    h_best, h_dist = None, 10**9
    for fam, cons in consensus.items():
        d = hamming(prefix, cons[: len(prefix)])
        if d < h_dist:
            h_dist, h_best = d, fam
    # Soft score under chosen family's freq table
    ll = soft_log_likelihood(prefix, freq_tables[h_best])
    return h_best, h_dist, ll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tsv", type=Path, required=True)
    ap.add_argument("--samples-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix-len", type=int, default=PREFIX_LEN)
    args = ap.parse_args()

    consensus, freq_tables = build_germline_consensus(args.train_tsv, args.prefix_len)

    # Optional: print family consensus side-by-side so we can sanity-check.
    print("\n=== family consensus prefixes ===")
    for fam in sorted(consensus.keys()):
        print(f"  {fam}: {consensus[fam]}")
    print()

    per_cell = {}
    confusion = defaultdict(lambda: defaultdict(int))   # target_fam -> pred_fam -> count
    overall_correct = 0
    overall_total = 0

    for fasta in sorted(args.samples_dir.glob("*.fasta")):
        cell = fasta.stem                          # e.g. IGHG_IGHV3_K
        parts = cell.split("_")
        if len(parts) < 3:
            continue
        target_family = parts[1].upper()           # IGHV3

        seqs = read_fasta(fasta)
        n = len(seqs)
        if n == 0:
            continue

        correct = 0
        hams = []
        lls = []
        for s in seqs:
            prefix = s[: args.prefix_len]
            pred_fam, h, ll = classify(prefix, consensus, freq_tables)
            confusion[target_family][pred_fam] += 1
            hams.append(h)
            lls.append(ll)
            if pred_fam == target_family:
                correct += 1

        fidelity = correct / n
        per_cell[cell] = {
            "target_family": target_family,
            "n": n,
            "n_correct": correct,
            "fidelity": fidelity,
            "mean_hamming_to_consensus": sum(hams) / n,
            "mean_log_likelihood": sum(lls) / n,
        }
        overall_correct += correct
        overall_total += n
        print(f"  {cell}: fidelity={fidelity:.3f} ({correct}/{n})  "
              f"mean Hamming={sum(hams)/n:.2f}")

    overall = overall_correct / max(1, overall_total)
    out = {
        "overall_fidelity": overall,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "prefix_len": args.prefix_len,
        "n_families": len(consensus),
        "per_cell": per_cell,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "consensus_prefixes": consensus,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== Overall V-gene fidelity: {overall:.3f} ({overall_correct}/{overall_total}) ===")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
