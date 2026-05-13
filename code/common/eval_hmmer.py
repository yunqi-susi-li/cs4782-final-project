"""
HMMER-based "biological viability" check for generated antibodies.

Each generated paired sequence (VH + linker + VL) is split at the linker, and
both chains are scanned with `hmmscan` against the Pfam Ig variable-domain
HMMs (V-set / V_Ig / Ig). A sequence "hits" if BOTH chains return at least
one significant Ig-V-domain hit at E-value <= --evalue (default 1e-5).

Outputs (--out hmmer.json):
  {
    "n_queries": ...,        # total pair sequences scanned
    "n_chains": ...,         # 2 * n_queries (VH + VL queries)
    "n_chain_hits": ...,     # chains with >=1 hit
    "n_paired_hits": ...,    # pair seqs where BOTH VH and VL hit
    "share_chain_hits": ...,
    "share_paired_hits": ...,
    "mean_evalue": ...,
    "elapsed_seconds": ...,
    "hmm_db": "...",
    "per_cell": { ... }
  }

Requires HMMER 3.x (`hmmscan` on PATH) and a pressed HMM database.
See `scripts/setup_hmmer.sh` for installation + DB preparation.

Usage:
    python scripts/eval_hmmer.py \
        --samples-dir <data-dir>/samples_dplm_stochastic \
        --hmm-db      $HOME/ab_ld4lg/hmmer_db/Ig.hmm \
        --out         <data-dir>/eval_reports_dplm_stochastic/hmmer.json \
        --tmp-dir     /tmp/hmmer_eval \
        --evalue      1e-5 \
        --cpu         4
"""


import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

LINKER_RE = re.compile(r"GGGGSGGGGS")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    out, header, body = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    out.append((header, "".join(body)))
                header = line[1:]
                body = []
            else:
                body.append(line)
        if header is not None:
            out.append((header, "".join(body)))
    return out


def split_pair(seq: str) -> tuple[str, str] | None:
    m = LINKER_RE.search(seq)
    if m is None:
        return None
    return seq[: m.start()], seq[m.end():]


def parse_domtbl(domtbl_path: Path, evalue_cutoff: float) -> dict[str, list[float]]:
    """
    Parse hmmscan --domtblout. Returns {query_name: [evalues_of_significant_hits]}.
    Header lines start with '#'.
    Domtblout columns (whitespace-separated, fixed):
        target_name target_acc tlen query_name query_acc qlen E-value score bias
        # of c-Evalue i-Evalue score bias from to from to from to acc description
    """
    hits: dict[str, list[float]] = {}
    if not domtbl_path.exists():
        return hits
    with open(domtbl_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            query = parts[3]
            try:
                full_evalue = float(parts[6])  # full-sequence E-value
            except ValueError:
                continue
            if full_evalue <= evalue_cutoff:
                hits.setdefault(query, []).append(full_evalue)
    return hits


def run_hmmscan(query_fasta: Path, hmm_db: Path, out_domtbl: Path,
                cpu: int, evalue_cutoff: float) -> None:
    cmd = [
        "hmmscan",
        "--cpu", str(cpu),
        "-E", str(evalue_cutoff),
        "--domtblout", str(out_domtbl),
        "-o", "/dev/null",
        str(hmm_db),
        str(query_fasta),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", type=Path, required=True,
                    help="Directory of *.fasta files (one per cell)")
    ap.add_argument("--hmm-db", type=Path, required=True,
                    help="Path to pressed HMM database (e.g. Ig.hmm)")
    ap.add_argument("--out", type=Path, required=True,
                    help="JSON output report")
    ap.add_argument("--tmp-dir", type=Path, default=Path("/tmp/hmmer_eval"))
    ap.add_argument("--evalue", type=float, default=1e-5)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--n-per-cell", type=int, default=0,
                    help="If >0, subsample this many sequences per cell")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if shutil.which("hmmscan") is None:
        print("[error] hmmscan not on PATH. Install HMMER (see setup_hmmer.sh).",
              file=sys.stderr)
        sys.exit(1)
    if not args.hmm_db.exists():
        print(f"[error] HMM DB not found: {args.hmm_db}", file=sys.stderr)
        sys.exit(1)

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(args.samples_dir.glob("*.fasta"))
    if not fasta_files:
        print(f"[error] no .fasta files in {args.samples_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[hmmer] {len(fasta_files)} cells found in {args.samples_dir}")
    print(f"[hmmer] HMM DB: {args.hmm_db}")
    print(f"[hmmer] E-value cutoff: {args.evalue}")
    t0 = time.time()

    per_cell: dict = {}
    grand = {
        "n_queries": 0, "n_chains": 0,
        "n_chain_hits": 0, "n_paired_hits": 0,
        "all_evalues": [],
    }

    import random
    rng = random.Random(args.seed)

    for fa in fasta_files:
        cell = fa.stem
        recs = read_fasta(fa)
        if args.n_per_cell > 0 and len(recs) > args.n_per_cell:
            recs = rng.sample(recs, args.n_per_cell)

        # Build a single FASTA with VH and VL as separate queries.
        chains: list[tuple[str, str]] = []  # (chain_name, seq)
        pair_chain_names: list[tuple[str, str]] = []  # (vh_name, vl_name)
        for i, (h, s) in enumerate(recs):
            sp = split_pair(s)
            if sp is None:
                continue
            vh, vl = sp
            vh_name = f"{cell}_{i:04d}_VH"
            vl_name = f"{cell}_{i:04d}_VL"
            chains.append((vh_name, vh))
            chains.append((vl_name, vl))
            pair_chain_names.append((vh_name, vl_name))

        if not chains:
            print(f"[hmmer] {cell}: no parseable pairs, skipping")
            continue

        query_fa = args.tmp_dir / f"{cell}.queries.fasta"
        domtbl = args.tmp_dir / f"{cell}.domtbl"
        with open(query_fa, "w") as f:
            for name, s in chains:
                f.write(f">{name}\n{s}\n")

        print(f"[hmmer] {cell}: scanning {len(chains)} chains ({len(pair_chain_names)} pairs)...")
        t_cell = time.time()
        run_hmmscan(query_fa, args.hmm_db, domtbl, cpu=args.cpu, evalue_cutoff=args.evalue)
        cell_secs = time.time() - t_cell

        hits = parse_domtbl(domtbl, args.evalue)
        n_chain_hit = sum(1 for n, _ in chains if n in hits)
        n_pair_hit = sum(1 for vh, vl in pair_chain_names if vh in hits and vl in hits)
        evalues = [e for evs in hits.values() for e in evs]

        per_cell[cell] = {
            "n_pairs": len(pair_chain_names),
            "n_chains": len(chains),
            "n_chain_hits": n_chain_hit,
            "n_paired_hits": n_pair_hit,
            "share_chain_hits": n_chain_hit / len(chains),
            "share_paired_hits": n_pair_hit / len(pair_chain_names),
            "mean_evalue": (sum(evalues) / len(evalues)) if evalues else None,
            "elapsed_s": cell_secs,
        }
        grand["n_queries"] += len(pair_chain_names)
        grand["n_chains"] += len(chains)
        grand["n_chain_hits"] += n_chain_hit
        grand["n_paired_hits"] += n_pair_hit
        grand["all_evalues"].extend(evalues)
        print(f"[hmmer] {cell}: {n_pair_hit}/{len(pair_chain_names)} paired hits "
              f"({n_chain_hit}/{len(chains)} chains) in {cell_secs:.1f}s")

    elapsed = time.time() - t0
    n_chains = grand["n_chains"]
    n_queries = grand["n_queries"]
    evalues = grand.pop("all_evalues")

    report = {
        "samples_dir": str(args.samples_dir),
        "hmm_db": str(args.hmm_db),
        "evalue_cutoff": args.evalue,
        "n_queries": n_queries,
        "n_chains": n_chains,
        "n_chain_hits": grand["n_chain_hits"],
        "n_paired_hits": grand["n_paired_hits"],
        "n_hits": grand["n_paired_hits"],  # alias for compute_poster_metrics
        "share_chain_hits": grand["n_chain_hits"] / max(1, n_chains),
        "share_paired_hits": grand["n_paired_hits"] / max(1, n_queries),
        "share_hits": grand["n_paired_hits"] / max(1, n_queries),  # alias
        "mean_evalue": (sum(evalues) / len(evalues)) if evalues else None,
        "elapsed_seconds": elapsed,
        "per_cell": per_cell,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[hmmer] wrote {args.out}")
    print(f"[hmmer] paired hits: {grand['n_paired_hits']}/{n_queries} "
          f"({100 * grand['n_paired_hits'] / max(1, n_queries):.1f}%)")
    print(f"[hmmer] total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
