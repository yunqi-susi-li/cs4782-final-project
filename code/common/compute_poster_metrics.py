"""
Aggregate every metric the poster needs from the per-cell + foldability JSONs
that already live under each `eval_reports_*` directory.

Outputs (per --run label):
  * pLDDT mean
  * pLDDT median   ← computed from `plddt_distribution` arrays
  * share > 70
  * diversity 4-gram (mean across cells)
  * distinct n-grams (1/2/3/4-gram, summed across cells)
  * runtime (folding seconds + sampling seconds if available)
  * (optional) HMMER hits — read from a sibling `hmmer.json` if present

Usage:
    python scripts/compute_poster_metrics.py \
        --runs LD4LG=eval_reports samples=samples \
               DPLM_stoch=eval_reports_dplm_stochastic samples=samples_dplm_stochastic \
               DPLM_tuned=eval_reports_dplm_tuned samples=samples_dplm_tuned \
        --out  results_summary.json \
        --csv  results_summary.csv

Or just point it at a single run:
    python scripts/compute_poster_metrics.py \
        --label DPLM_stoch \
        --reports-dir /mnt/beegfs/.../eval_reports_dplm_stochastic \
        --samples-dir /mnt/beegfs/.../samples_dplm_stochastic \
        --out poster_metrics_dplm_stoch.json
"""


import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median

LINKER_RE = re.compile(r"GGGGSGGGGS")


def read_fasta(path: Path) -> list[str]:
    seqs, body = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if body:
                    seqs.append("".join(body))
                    body = []
            else:
                body.append(line)
        if body:
            seqs.append("".join(body))
    return seqs


def ngram_set(seqs: list[str], n: int) -> tuple[int, int]:
    """Returns (n_unique_ngrams, n_total_ngrams) over the corpus."""
    total, unique = 0, set()
    for s in seqs:
        if len(s) < n:
            continue
        grams = [s[i:i + n] for i in range(len(s) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique), total


def collect_metrics(reports_dir: Path, samples_dir: Path | None) -> dict:
    metrics: dict = {
        "reports_dir": str(reports_dir),
        "samples_dir": str(samples_dir) if samples_dir else None,
    }

    # ---- 1. per-cell JSONs (eval.py output) → diversity + n-grams ----
    cell_jsons = sorted(
        p for p in reports_dir.glob("*.json")
        if p.stem not in {"foldability", "vgene_fidelity", "sweep", "hmmer"}
    )
    div4_per_cell, n_total_per_cell = [], 0
    for p in cell_jsons:
        d = json.loads(p.read_text())
        if "div_4gram" in d:
            div4_per_cell.append(d["div_4gram"])
        n_total_per_cell += d.get("n_total", 0)
    metrics["n_cells"] = len(cell_jsons)
    metrics["n_seqs_total"] = n_total_per_cell
    metrics["mean_div_4gram_across_cells"] = (
        sum(div4_per_cell) / len(div4_per_cell) if div4_per_cell else None
    )

    # Distinct n-grams: needs FASTAs (per-cell JSON only stores ratios).
    if samples_dir and samples_dir.exists():
        all_seqs: list[str] = []
        for fa in sorted(samples_dir.glob("*.fasta")):
            all_seqs.extend(read_fasta(fa))
        for n in (1, 2, 3, 4):
            uniq, total = ngram_set(all_seqs, n)
            metrics[f"distinct_{n}gram"] = uniq
            metrics[f"total_{n}gram"] = total
            metrics[f"div_{n}gram_corpus"] = (uniq / total) if total else None
        metrics["n_seqs_in_fasta"] = len(all_seqs)

    # ---- 2. foldability.json → pLDDT mean / median / share>70 / runtime ----
    fpath = reports_dir / "foldability.json"
    if fpath.exists():
        f = json.loads(fpath.read_text())
        metrics["plddt_mean"] = f.get("overall_mean_plddt")
        metrics["plddt_share_above_70"] = f.get("overall_share_above_70")
        metrics["plddt_mean_cdr"] = f.get("overall_mean_cdr_plddt")
        metrics["folding_runtime_s"] = f.get("elapsed_seconds")
        metrics["n_folded"] = f.get("total_folded")

        # Median pLDDT across all cells, computed from per-cell distributions.
        all_plddt: list[float] = []
        per_cell_summary: dict = {}
        for cell, c in (f.get("per_cell") or {}).items():
            dist = c.get("plddt_distribution") or []
            all_plddt.extend(dist)
            per_cell_summary[cell] = {
                "n_folded": c.get("n_folded"),
                "mean_plddt": c.get("mean_plddt"),
                "median_plddt": median(dist) if dist else None,
                "share_above_70": c.get("share_plddt_above_70"),
            }
        metrics["plddt_median"] = median(all_plddt) if all_plddt else None
        metrics["per_cell_plddt"] = per_cell_summary

    # ---- 3. vgene_fidelity.json → V-gene match rate ----
    vpath = reports_dir / "vgene_fidelity.json"
    if vpath.exists():
        v = json.loads(vpath.read_text())
        metrics["vgene_fidelity"] = v.get("overall_fidelity")
        metrics["vgene_n_correct"] = v.get("overall_correct")

    # ---- 4. sweep.json → best config, sweep runtime ----
    spath = reports_dir / "sweep.json"
    if spath.exists():
        s = json.loads(spath.read_text())
        results = s.get("results", [])
        if results:
            best = max(results, key=lambda r: r.get("score", 0))
            metrics["sweep_best"] = {
                "T": best.get("temperature"),
                "top_p": best.get("top_p"),
                "div_4gram": best.get("mean_div_4gram"),
                "score": best.get("score"),
            }
            metrics["sweep_runtime_s"] = sum(r.get("elapsed_s", 0) for r in results)
            metrics["sweep_n_configs"] = len(results)

    # ---- 5. hmmer.json (sibling, if eval_hmmer.py was run) ----
    hpath = reports_dir / "hmmer.json"
    if hpath.exists():
        h = json.loads(hpath.read_text())
        metrics["hmmer_n_hits"] = h.get("n_hits")
        metrics["hmmer_n_queries"] = h.get("n_queries")
        metrics["hmmer_share_hits"] = h.get("share_hits")
        metrics["hmmer_runtime_s"] = h.get("elapsed_seconds")
        metrics["hmmer_db"] = h.get("hmm_db")

    return metrics


def to_csv_row(label: str, m: dict) -> dict:
    """Flatten one run's metrics into a single CSV-friendly row."""
    return {
        "run": label,
        "n_seqs": m.get("n_seqs_in_fasta") or m.get("n_seqs_total"),
        "n_cells": m.get("n_cells"),
        "plddt_mean": m.get("plddt_mean"),
        "plddt_median": m.get("plddt_median"),
        "plddt_share_above_70": m.get("plddt_share_above_70"),
        "div_4gram_cells": m.get("mean_div_4gram_across_cells"),
        "div_4gram_corpus": m.get("div_4gram_corpus"),
        "distinct_1gram": m.get("distinct_1gram"),
        "distinct_2gram": m.get("distinct_2gram"),
        "distinct_3gram": m.get("distinct_3gram"),
        "distinct_4gram": m.get("distinct_4gram"),
        "vgene_fidelity": m.get("vgene_fidelity"),
        "hmmer_n_hits": m.get("hmmer_n_hits"),
        "hmmer_share_hits": m.get("hmmer_share_hits"),
        "folding_runtime_s": m.get("folding_runtime_s"),
        "sweep_runtime_s": m.get("sweep_runtime_s"),
    }


def parse_runs_arg(triples: list[str]) -> list[tuple[str, Path, Path | None]]:
    """
    Parse alternating tokens: LABEL=reports_dir samples=samples_dir LABEL=...

    Or simpler: a list of "label:reports[:samples]" triples (preferred).
    """
    runs: list[tuple[str, Path, Path | None]] = []
    for t in triples:
        if ":" in t:
            parts = t.split(":")
            label = parts[0]
            reports = Path(parts[1])
            samples = Path(parts[2]) if len(parts) > 2 and parts[2] else None
            runs.append((label, reports, samples))
        else:
            raise SystemExit(f"Bad --run token: {t!r}. Use LABEL:reports_dir[:samples_dir]")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="Single-run mode: a name for this run")
    ap.add_argument("--reports-dir", type=Path)
    ap.add_argument("--samples-dir", type=Path, default=None)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="Multi-run mode: LABEL:reports_dir[:samples_dir] LABEL2:...")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    runs: list[tuple[str, Path, Path | None]] = []
    if args.runs:
        runs = parse_runs_arg(args.runs)
    elif args.reports_dir:
        runs = [(args.label or args.reports_dir.name, args.reports_dir, args.samples_dir)]
    else:
        ap.error("Provide either --runs or --reports-dir")

    out: dict = {}
    rows: list[dict] = []
    for label, reports, samples in runs:
        if not reports.exists():
            print(f"[warn] skipping {label}: {reports} does not exist", file=sys.stderr)
            continue
        print(f"[metrics] processing {label}: {reports}")
        m = collect_metrics(reports, samples)
        out[label] = m
        rows.append(to_csv_row(label, m))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[metrics] wrote {args.out}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"[metrics] wrote {args.csv}")

    # Pretty summary to stdout
    print("\n=== POSTER METRICS SUMMARY ===")
    for r in rows:
        print(f"\n[{r['run']}]")
        for k, v in r.items():
            if k == "run":
                continue
            if isinstance(v, float):
                print(f"  {k:30s} {v:.4f}")
            else:
                print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
