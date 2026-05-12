# `code/` — Source code

Implementation of two conditional-diffusion language models for paired
antibody sequence generation, plus shared evaluation utilities.

```
code/
├── data_preprocessing/   MMseqs2 similarity-reduction pipeline:
│                         paired-chain dedup at 95%/90% identity →
│                         leakage-free train/val/test splits. The
│                         downstream OAS → token-memmap script lives
│                         in diffusion/LD4LG/preprocess.py.
├── common/               shared evaluation suite used by both diffusion
│                         tracks: foldability (IgFold), HMMER hit rate,
│                         token recovery, V-gene fidelity, perplexity
└── diffusion/
    ├── DPLM/             discrete absorbing diffusion (Wang et al., 2024)
    └── LD4LG/            continuous latent diffusion (Lovelace et al., 2023)
```

Each subdirectory has its own `README.md` with the relevant architecture,
extensions, and reproduction commands. See `../README.md` for the project-level overview.
