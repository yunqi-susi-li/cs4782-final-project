# `code/diffusion/` — Discrete-diffusion models for CDR3 generation

This is the **core** of the re-implementation. Each subdirectory is one
ablation on the path from "plain absorbing-state diffusion" to "full DPLM-2
+ germline-edit head."

## Ablation ladder

| Dir | Name | What's added vs previous row | Params | Val AA Recovery |
|---|---|---|---|---|
| `v1_plain/`           | **v1A / v1B** | Absorbing-state D3PM on observed-CDR3 only — no germline, no MDLM weighting, no shared stream | 7.5 M / 58.3 M | ~46 – 47 % |
| `v3_d3pm_germline/`   | **v3_prod / v3_scale** | + germline CDR3 via **cross-attention**; depth/width scan | 10.2 M / 12.5 – 22.8 M | 56.89 % / 57.69 % |
| `v4_dplm2/`           | **v4 — DPLM-2 re-impl + edit head** | + **shared-stream** layout (germline in self-attn, not cross-attn); + **MDLM importance weighting** `w(t) = -log(ᾱ)/(1-ᾱ)`; + **germline-edit auxiliary head** (our extension) | **19.0 M** | **77.00 %** |
| `v5_latent/`          | **v5 — latent diffusion** *(exploratory, [Lovelace et al. '23](https://github.com/justinlovelace/latent-diffusion-for-language))* | Continuous diffusion in a learned CDR3-autoencoder latent — adapts LD4LG (Cornell, covered in CS 4782 by Prof. Weinberger) to antibody CDR3s | TBD | TBD |

## The +20 pp jump (v3 → v4)

The story the table tells: scaling v3 from 10 M to 23 M params only buys
+0.8 pp, but moving from cross-attention to the DPLM-2 shared stream with
MDLM weighting and the edit head buys **+19 pp** at essentially the same
parameter count. **How** germline is fused — not how much capacity we throw
at the model — is the dominant factor.

For the detailed DPLM-2 re-implementation spec (shared stream, MDLM weighting,
iterative unmasking sampler, germline-edit head), see
[`v4_dplm2/README.md`](./v4_dplm2/README.md).

## Subdirectory contents (all tracks share the same file layout)

```
<track>/
├── model.py           # transformer denoiser
├── diffusion.py       # forward/reverse kernel + loss
├── train.py           # training loop
├── sample.py          # iterative unmasking sampler
└── eval.py            # recovery + generative metrics
```
