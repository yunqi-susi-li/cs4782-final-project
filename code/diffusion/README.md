# `code/diffusion/` — Two diffusion paradigms for paired antibody generation

This project compares two complementary approaches to conditional antibody
sequence diffusion on the same OAS data, conditioning, and evaluation pipeline:

| Dir | Paradigm | Operates on | Our extension |
|---|---|---|---|
| `DPLM/`  | **Discrete absorbing diffusion** ([Wang et al., ICLR 2024](https://github.com/bytedance/dplm)) | AA tokens directly (BERT-style mask, multi-step) | **Stochastic categorical sampling sweep** (T × top-p) replacing the default greedy argmax |
| `LD4LG/` | **Continuous latent diffusion** ([Lovelace et al., NeurIPS 2023](https://github.com/justinlovelace/latent-diffusion-for-language)) | A learned 32 × 64 latent from a separately-trained AE | **3-way independent classifier-free guidance** over (isotype, V-family, light locus) |

Both share the same data preprocessing, the same conditioning schema, and the
same evaluation suite (validity, linker presence, n-gram diversity, IgFold
pLDDT, HMMER hit rate, V-gene fidelity, memorization checks).

## Quality vs diversity trade-off

The two paradigms land in different regions of the quality-diversity frontier
(see top-level `results/` for full numbers):

| Configuration | 4-gram diversity | Foldable share (pLDDT > 70) | HMMER hit rate |
|---|---|---|---|
| **DPLM (greedy)**             | ~0.005    | **~98%**  | 100.0% |
| **DPLM (T=1.0, top-p=0.95)**  | 0.051     | 96.7%     | 100.0% |
| **LD4LG (CFG w=2.0)**         | 0.136     | 38.8%     | 100.0% |
| **DPLM (T=1.3, top-p=0.99)**  | **0.207** | 25.9%     | 99.5%  |

DPLM with greedy decoding produces highly foldable but near-template, very
repetitive sequences. Adding stochastic temperature + nucleus sampling
dramatically increases diversity at the cost of foldability; LD4LG sits in
between. Numbers match the headline table in the top-level
[`README.md`](../../README.md) and the per-decoder breakdown in
[`DPLM/README.md`](DPLM/README.md).

## Subdirectory layout

Both tracks follow the same layout:

```
<track>/
├── __init__.py
├── tokenizer.py        # AA tokenizer (24 tokens; DPLM adds <mask>)
├── model.py            # transformer denoiser
├── diffusion.py        # forward/reverse kernel + loss
├── train.py            # training loop
├── sample.py           # sampler
├── smoke_test.py       # tiny end-to-end shape test
└── README.md           # paradigm-specific docs
```

LD4LG additionally has the autoencoder split into `autoencoder.py`,
`train_autoencoder.py`, etc. (two-stage training).