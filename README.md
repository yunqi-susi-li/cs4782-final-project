# Multi-Condition Diffusion for Paired Antibody Sequences

**CS 5782/4782 — Introduction to Deep Learning, Final Project (Spring 2026)**
Cornell University · **Authors:** Yunqi Li, Yonglin Zhang

> **TL;DR.** We re-implement and compare two conditional-diffusion language
> models—**LD4LG** (Lovelace et al., NeurIPS 2023; continuous latent
> diffusion) and **DPLM** (Wang et al., ICLR 2024; discrete absorbing
> diffusion)—on 2.17M paired antibody sequences. We extend both with
> three-way independent classifier-free guidance and add a 20-configuration
> temperature × top-p decoding sweep for DPLM. The resulting four
> (model, decoder) configurations characterize a quality–diversity Pareto
> frontier: LD4LG dominates on sequence recovery (SRR = 0.753) and perplexity
> (1.08); decoding choice alone, with DPLM weights held fixed, spans
> foldability from 96.7% to 25.9%.

---

## 1. Introduction

This is the GitHub **re-implementation deliverable** for our CS 5782/4782
final project. The repo re-implements and compares two papers on the same
conditional sequence-generation task:

- **Latent Diffusion for Language Generation (LD4LG).** Lovelace, J., Kishore,
  V., Wan, C., Shekhtman, E., Weinberger, K. Q. NeurIPS 2023.
  <https://github.com/justinlovelace/latent-diffusion-for-language>
- **Diffusion Language Models Are Versatile Protein Learners (DPLM).**
  Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., Gu, Q. ICLR 2024.
  <https://github.com/bytedance/dplm>

The task is conditional generation of paired antibody chains
(V$_H$ ⊕ V$_L$): each training example is a $\le$288-token amino-acid string
plus three categorical labels (isotype, V-gene family, light-chain locus).
LD4LG runs continuous Gaussian diffusion in a 32 × 64 latent produced by a
language autoencoder; DPLM runs discrete absorbing diffusion directly over the
24-symbol AA alphabet. Both originally support a single class condition, but
antibodies factor naturally into three semantically independent labels,
motivating our multi-condition extension. The comparison is symmetric—neither
paradigm is a baseline; we ask where each lands on the antibody-domain
quality–diversity Pareto frontier and which decoding choices control that
position.

## 2. Chosen Result

We target LD4LG's CFG-controlled diversity–quality trade-off (Sec. 3, Fig. 3
of Lovelace et al., 2023) and DPLM's recovery performance (Tab. 3 of Wang et
al., 2024), adapted to the antibody domain. The natural-language fluency
oracle has no antibody analogue, so we substitute two structural-biology
oracles—IgFold-predicted pLDDT > 70 (foldability) and HMMER hit rate against
the Pfam Ig V-set HMM (domain validity)—and use corpus 4-gram diversity as
the diversity axis. See `report/` for the full 2-page analysis.

## 3. GitHub Contents

```
cs4782-final-project/
├── README.md                       this file
├── LICENSE                         MIT
├── requirements.txt
├── code/
│   ├── README.md
│   ├── common/                     shared evaluation suite
│   ├── data_preprocessing/         MMseqs2 similarity-reduction pipeline:
│   │                               7-step paired-chain dedup at 95% / 90%
│   │                               sequence identity, 0% train/test leakage
│   │                               audited (see PIPELINE.md for full spec)
│   └── diffusion/
│       ├── DPLM/                   discrete absorbing diffusion
│       └── LD4LG/                  continuous latent diffusion
├── data/                           OAS download + preprocessing notes
├── results/{figures,tables}        per-cell metrics, Pareto data
├── poster/                         final-presentation poster ([PDF](poster/DL5782_Final_Project_Poster.pdf))
└── report/                         2-page summary (PDF)
```

Each subdirectory has its own README.

## 4. Re-implementation Details

**Dataset.** 2.17M paired V$_H$ ⊕ GGGGSGGGGS ⊕ V$_L$ chains from the Observed
Antibody Space (OAS), MMseqs2-deduplicated at 95%/90% sequence identity for
0% train/test leakage (pipeline at [`code/data_preprocessing/`](code/data_preprocessing/)).
Right-padded to 288 tokens over a 24-symbol AA vocabulary. An 18-cell
stratified test split (3 isotypes × 3 V-families × 2 loci) provides 200
reference + 512 generated sequences per cell.

**LD4LG.** Stage 1: encoder–Perceiver-Resampler–decoder autoencoder mapping
tokens to a unit-norm 32 × 64 latent (val CE = 0.038, 97.7% reconstruction).
Stage 2: 12-layer pre-LN Transformer denoiser with QK-RMSNorm, GeGLU FFN,
AdaLN time/class conditioning, and U-ViT-style dense skips; v-prediction loss
(Salimans & Ho, 2022). Extensions vs the paper: AE trained from scratch (BART
vocabulary incompatible with the 24-AA alphabet); CFG extended to three
independent conditions, each with its own null index; self-conditioning
disabled (DDP rank divergence).

**DPLM.** 12-layer bidirectional Transformer sharing all blocks with the
LD4LG denoiser, trained with absorbing-state cross-entropy under
γ(t) = 1 − cos²(πt/2). Sampling via confidence-ranked iterative unmasking.
Extensions vs the paper: same 3-way independent CFG over per-position
logits; stochastic categorical sampling (temperature + nucleus top-p)
replaces the released codebase's greedy argmax, which we observed collapses
to a single near-template output per condition; a 20-configuration grid
sweep (T × top-p) to locate the quality–diversity sweet spot.

**Evaluation** (`code/common/`). Seven metrics: linker recovery (format),
4-gram diversity, V-family classifier accuracy (conditional fidelity),
Sequence Recovery Rate (SRR), held-out NLL perplexity, HMMER hit rate, and
IgFold pLDDT > 70 foldable share. Plus exact-match and Hamming-$\le 3$
training memorization checks across all 9,216 generated sequences.

## 5. Reproduction Steps

```bash
# 1. Clone + env (Python 3.10+, CUDA 11.8+)
git clone https://github.com/yunqi-susi-li/cs4782-final-project.git
cd cs4782-final-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Similarity-reduce paired OAS to a leakage-free split
#    (see code/data_preprocessing/README.md for the 7-step MMseqs2 pipeline)
bash code/data_preprocessing/run_pipeline.sh \
    --input    /path/to/oas_paired.pkl \
    --output   processed/dedup \
    --receptor bcr --threshold 0.95

# 3. Tokenize the dedup'd splits → int16 memmaps (see data/README.md)
python -m code.diffusion.LD4LG.preprocess \
    --archives <oas_export.tar.gz> --out processed/ --max-len 288

# 4. Train LD4LG (~9 h on a single H100)
python -m code.diffusion.LD4LG.train_autoencoder \
    --data processed/ --out runs/ae --steps 50000
python -m code.diffusion.LD4LG.train_diffusion \
    --data processed/ --ae-ckpt runs/ae/autoencoder_latest.pt \
    --out runs/ld4lg --steps 250000

# 5. Train DPLM (~5 h on a single H100)
python -m code.diffusion.DPLM.train \
    --data processed/ --out runs/dplm --steps 100000

# 6. Sample one cell from each model
python -m code.diffusion.LD4LG.sample \
    --ae-ckpt runs/ae/autoencoder_latest.pt \
    --diff-ckpt runs/ld4lg/diffusion_latest.pt \
    --iso IGHG --vfam IGHV3 --loc K --num 512 --cfg 2.0 \
    --out samples/ld4lg/IGHG_IGHV3_K.fasta

python -m code.diffusion.DPLM.sample \
    --ckpt runs/dplm/dplm_latest.pt \
    --iso IGHG --vfam IGHV3 --loc K --num 512 \
    --sample-mode stochastic --temperature 1.0 --top-p 0.95 \
    --out samples/dplm/IGHG_IGHV3_K.fasta

# 7. DPLM decoding sweep (locates Pareto-best (T, top-p))
python -m code.diffusion.DPLM.sampling_sweep \
    --ckpt runs/dplm/dplm_latest.pt \
    --out results/dplm_sweep.json --n-per-config 64

# 8. Evaluate one generated FASTA
python -m code.diffusion.LD4LG.eval \
    --fasta samples/ld4lg/IGHG_IGHV3_K.fasta \
    --train-tokens processed/train.tokens.npy \
    --train-meta processed/train.meta.json \
    --out results/eval_reports/ld4lg_IGHG_IGHV3_K.json
```

**Hardware.** Trained on a single NVIDIA H100. CPU works for the smoke tests
(`python -m code.diffusion.{LD4LG,DPLM}.smoke_test`) but is not practical for
training.

## 6. Results / Insights

Headline numbers across 9,216 generated sequences (18 cells × 512). Bold =
best per metric.

| Metric | DPLM-default (T=1.0, p=0.95) | DPLM-tuned (T=1.3, p=0.99) | LD4LG (w=2.0) |
|---|---|---|---|
| Linker recovery               | **100.0%** | 92.7%       | 99.7%       |
| 4-gram diversity              | 0.051      | **0.207**   | 0.136       |
| V-family accuracy             | **99.98%** | 93.8%       | 99.6%       |
| Sequence Recovery Rate        | 0.476      | 0.564       | **0.753**   |
| Held-out NLL perplexity       | 1.37       | 1.37        | **1.08**    |
| HMMER hit rate                | **100%**   | 99.5%       | **100%**    |
| Foldable share (pLDDT > 70)   | **96.7%**  | 25.9%       | 38.8%       |

The most surprising finding is that the **same DPLM weights** span
96.7% → 25.9% foldability and 0.051 → 0.207 diversity, varying only
(T, top-p). Decoding choice is therefore a primary determinant of where a
discrete-diffusion model lands on the quality–diversity frontier. LD4LG and
DPLM are not totally ordered—they occupy distinct frontier regions, with
LD4LG dominating on SRR and perplexity while DPLM-default dominates on
foldability and conditional fidelity.

## 7. Conclusion

Two takeaways: (1) **decoding choice is a primary determinant** of where a
discrete-diffusion model lands on the quality–diversity frontier, with
weights held fixed; (2) **continuous latent and discrete absorbing
diffusion are not totally ordered**—the choice is application-conditioned.

A natural next step is a **per-cell adaptive decoding policy** that picks
(T, top-p) conditioned on (isotype, V-family, locus): the optimal operating
point is unlikely to be cell-invariant given the heterogeneous foldability
we saw across the 18 stratification cells.

## 8. References

1. Lovelace, J., Kishore, V., Wan, C., Shekhtman, E., Weinberger, K. Q.
   *Latent Diffusion for Language Generation.* NeurIPS 2023.
   <https://github.com/justinlovelace/latent-diffusion-for-language>
2. Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., Gu, Q.
   *Diffusion Language Models Are Versatile Protein Learners (DPLM).*
   ICLR 2024. <https://github.com/bytedance/dplm>
3. Olsen, T. H., Boyles, F., Deane, C. M.
   *Observed Antibody Space: A diverse database of cleaned, annotated, and
   translated unpaired and paired antibody sequences.* Protein Science 31(1),
   2022.
4. Ruffolo, J. A., Chu, L.-S., Mahajan, S. P., Gray, J. J.
   *Fast, accurate antibody structure prediction from deep learning.*
   Nature Communications 14:2389, 2023. (IgFold)
5. Steinegger, M., Söding, J. *MMseqs2 enables sensitive protein sequence
   searching for the analysis of massive data sets.* Nature Biotechnology
   35:1026–1028, 2017.
6. Salimans, T., Ho, J. *Progressive Distillation for Fast Sampling of
   Diffusion Models (v-prediction).* ICLR 2022.
7. Ho, J., Salimans, T. *Classifier-Free Diffusion Guidance.* NeurIPS
   Workshop on Deep Generative Models, 2022.
8. Bao, F. et al. *All Are Worth Words: A ViT Backbone for Diffusion Models
   (U-ViT).* CVPR 2023.
9. Eddy, S. R. *Accelerated Profile HMM Searches.* PLoS Computational Biology
   7(10):e1002195, 2011.

## 9. Acknowledgements

This project was developed at Cornell University as coursework for
**CS 5782/4782 — Introduction to Deep Learning** (Spring 2026). We thank the
course instructors, **Prof. Kilian Weinberger** and **Prof. Wei-Chiu Ma**,
for their guidance and feedback throughout the semester, and especially
Prof. Weinberger for introducing **Latent Diffusion for Language Generation**
(LD4LG) in lecture—his broader framing of diffusion for discrete sequences
made this project possible. We also thank the De Vlaminck Lab for their
support, and especially **Shaowen Jiang** (De Vlaminck Lab, Cornell) for
antibody-domain guidance, and the curators of the **Observed Antibody
Space** database. Compute resources were provided by **NSF ACCESS /
Purdue Anvil** (data preprocessing) and **Cornell University** — including
the **AIDA cluster** — for model training, sampling, and evaluation.
