# `code/common/` — Shared evaluation suite

Independent, additive evaluation scripts shared between the LD4LG and DPLM
tracks. Each script reads from FASTAs / checkpoints and writes its own JSON
to `results/eval_reports/`; nothing else is touched.

| Script | Input | Output | Compute |
|---|---|---|---|
| `eval_recovery_aar.py` | held-out tokens + AE ckpt | `recovery_aar.json` | CPU, ~2 min for 5k seqs |
| `eval_perplexity.py` | held-out tokens + AE + DPLM ckpts | `perplexity.json` | 1× GPU, ~5 min |
| `eval_vgene_fidelity.py` | 18 FASTA + train metadata | `vgene_fidelity.json` | CPU, ~30 min |
| `eval_foldability.py` | 18 FASTA | `foldability.json` + per-cell PDBs | 1× H100, ~3–7 h |
| `eval_hmmer.py` | 18 FASTA + Pfam HMM | `hmmer.json` | CPU, ~10–30 min |
| `compute_poster_metrics.py` | the above JSONs | `poster_metrics.{csv,json}` | CPU, seconds |

## Setup

- HMMER 3.x must be on PATH for `eval_hmmer.py`. See `setup_hmmer.sh` for
  the install + Pfam Ig V-set HMM download steps.
- `pip install igfold` for `eval_foldability.py`.

## Running

Each script is standalone. See its `--help` for arguments. Typical
post-sampling order:

```bash
python -m code.common.eval_vgene_fidelity --samples-dir samples/ld4lg/ --train-tsv ... --out results/vgene.json
python -m code.common.eval_hmmer          --samples-dir samples/ld4lg/ --hmm-db <Pfam_Ig.hmm> --out results/hmmer.json
python -m code.common.eval_foldability    --samples-dir samples/ld4lg/ --out results/foldability.json
python -m code.common.eval_perplexity     --data processed/ --ae-ckpt ... --dplm-ckpt ... --out results/ppl.json
python -m code.common.eval_recovery_aar   --data processed/ --ae-ckpt ... --out results/recovery.json
python -m code.common.compute_poster_metrics --reports-dir results/ --out results/poster_metrics.csv
```

To roll back any single tier: delete its output JSON.