# AI Tool Use Declaration

**Project:** Multi-Condition Diffusion for Paired Antibody Sequences
**Course:** CS 5782 / 4782 — Introduction to Deep Learning (Spring 2026)
**Authors:** Yunqi Li, Yonglin Zhang
**Last updated:** May 12, 2026

In accordance with course policy on transparent AI assistance, we disclose
the role of AI tools in producing this repository, the accompanying 2-page
report, and the poster.

---

## Tools used

Anthropic Claude (`claude-opus-4-7` and earlier Claude Sonnet / Opus
releases) via the claude.ai web interface and the Cowork desktop app,
used for code-drafting and writing assistance.

No other AI / LLM products were involved.

---

## Where AI assistance **was** used

- Drafting and iteratively revising the top-level `README.md` and the
  per-subdirectory README files.
- Skeleton / boilerplate drafts for:
  - the plotting scripts that produced the figures in
    [`results/figures/`](../results/figures/),
  - the shared evaluation utilities under
    [`code/common/eval_*.py`](../code/common/),
  - portions of the training and sampling scripts in
    [`code/diffusion/{LD4LG,DPLM}/`](../code/diffusion/).
- Suggestions on repository layout, file-naming conventions, and LaTeX
  typesetting (e.g., resolving two-column vs. one-column page breaks
  for the 2-page report).
- Debugging assistance on isolated Python errors, Slurm submission
  syntax, and git operations (branch-protection workflow, PR merge
  conventions).
- Copy-editing this declaration.

---

## Where AI assistance was **not** used

- **Research question and experimental design.** The decision to compare
  LD4LG and DPLM as two paradigms on the antibody-domain
  quality–diversity Pareto frontier was ours. The joint-vs.-independent
  multi-condition CFG ablation, the CFG-weight sweep, the 20-configuration
  (T × top-p) decoding grid for DPLM, the latency benchmark, and the
  per-cell SRR evaluation are all our own design choices.
- **Dataset, conditioning, metrics, hyperparameters.** Selection of OAS;
  the three conditioning labels (isotype / V-family / light-chain locus);
  the seven headline metrics (linker recovery, 4-gram diversity, V-family
  accuracy, SRR, held-out NLL, HMMER hit rate, IgFold pLDDT > 70 foldable
  share); model hyperparameters
  (12-layer pre-LN Transformer, 768-dim, GeGLU FFN, QK-RMSNorm, AdaLN,
  U-ViT-style skips, 32 × 64 AE latent geometry, 24-symbol AA
  vocabulary); decoding-grid bounds — all set by us.
- **Execution.** All experimental runs were executed by us on Cornell's
  AIDA cluster under our own account credentials. The AI tools had no
  access to the cluster, no access to the training data, and did not run
  any experiments.
- **Verification.** Every AI-drafted snippet of code was read, tested,
  and revised before being committed. Reported numbers in the report and
  the top-level README come from our own runs, which we re-verified
  prior to submission.
- **Interpretation.** The LD4LG-vs-DPLM comparison, the identification of
  decoding choice as a primary determinant of where a discrete-diffusion
  model lands on the Pareto frontier, and the Reflections / Future-work
  sections of the 2-page report are our own analysis.

---

## Responsibility

We take full responsibility for the correctness of all code, all reported
numbers, and all conclusions in this repository, the 2-page report, and
the poster.

— Yunqi Li, Yonglin Zhang
