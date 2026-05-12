# DPLM — Discrete Absorbing Diffusion for Paired Antibody Sequences

Re-implementation of

> Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., Gu, Q.
> **Diffusion Language Models Are Versatile Protein Learners (DPLM).** ICLR 2024.

adapted to paired antibody (V$_H \oplus$ V$_L$) generation with multi-condition
classifier-free guidance. Direct counterpart of the continuous latent diffusion
track in `../LD4LG/`.

> **Re-implementation status.** Independent PyTorch re-implementation of the
> training objective and sampler described in Wang et al. (ICLR 2024). No
> source code from the upstream repository
> (<https://github.com/bytedance/dplm>, Apache-2.0) was copied; only the
> paper and codebase were consulted for reference. Released under the MIT
> license alongside the rest of this project. See [`../../../NOTICE.md`](../../../NOTICE.md)
> for the full third-party-attribution table.

## Architecture

12-layer bidirectional pre-LN Transformer over a 25-symbol AA vocabulary
(20 canonical + 4 specials + `<mask>`). All blocks (GeGLU FFN, QK-RMSNorm
attention, learned positional embeddings) are imported from
`../LD4LG/nn_utils.py` for apples-to-apples comparison with the LD4LG
denoiser. Conditioning is three independent class embeddings (isotype,
V-family, light-chain locus), each with a null index for CFG, summed into
a per-sample bias added to the token-embedding sequence.

## Forward / reverse process

- **Forward**: $\gamma(t) = 1 - \cos^2(\pi t / 2)$; each non-special token is
  independently replaced with `<mask>` with probability $\gamma(t)$.
- **Reverse**: confidence-ranked iterative unmasking. At each step the model
  predicts logits at every position; among currently-masked positions we
  unmask the top-$k$ by confidence, writing either an argmax (greedy) or
  categorical (stochastic) sample depending on `--sample-mode`.

## Extensions vs. the original paper

1. **Three-way independent CFG.** The paper specifies single-class CFG;
   antibody data factors into three independent labels, each with its own
   null index, dropped independently with $p = 0.1$ at training time.
2. **Stochastic categorical decoding.** The DPLM-1 manuscript specifies
   per-step categorical sampling but the released codebase defaults to
   greedy argmax, which we observed collapses to a single near-template
   output per condition on antibody data. We restored stochastic sampling
   with explicit temperature + nucleus top-$p$ knobs.
3. **20-configuration $T \times p$ sweep** (`sampling_sweep.py`). Located the
   Pareto-best decoding setting ($T{=}1.3$, $p{=}0.99$) for this domain.

## Files

```
DPLM/
├── __init__.py
├── tokenizer.py            # 25 tokens (24 from LD4LG + <mask>)
├── model.py                # bidirectional Transformer denoiser
├── diffusion.py            # absorbing forward + iterative-unmask sampler
├── train.py                # training loop
├── sample.py               # FASTA sampler (--sample-mode stochastic|greedy)
├── sampling_sweep.py       # T × top-p grid (our extension)
├── smoke_test.py           # CPU-friendly end-to-end test
└── README.md               # this file
```

## Reproducing

```bash
# Train (~5 h on a single H100)
python -m code.diffusion.DPLM.train \
    --data processed/ --out runs/dplm --steps 100000

# Sample one cell (paper-spec stochastic decoding)
python -m code.diffusion.DPLM.sample \
    --ckpt runs/dplm/dplm_latest.pt \
    --iso IGHG --vfam IGHV3 --loc K --num 512 \
    --sample-mode stochastic --temperature 1.0 --top-p 0.95 \
    --out samples/IGHG_IGHV3_K.fasta

# Reproduce mode collapse with the released codebase's default decoder
python -m code.diffusion.DPLM.sample \
    --ckpt runs/dplm/dplm_latest.pt \
    --iso IGHG --vfam IGHV3 --loc K --num 512 \
    --sample-mode greedy \
    --out samples/IGHG_IGHV3_K_greedy.fasta
# Expected: 4-gram diversity < 0.01 (single near-template per condition).

# T × top-p sweep on 3 representative cells
python -m code.diffusion.DPLM.sampling_sweep \
    --ckpt runs/dplm/dplm_latest.pt \
    --out results/dplm_sweep.json \
    --cells "IGHM_IGHV1_K,IGHG_IGHV1_K,IGHA_IGHV1_K" \
    --n-per-config 64
```

## Observed results

Across 9,216 generated sequences (18 cells × 512):

| Decoder | 4-gram diversity | pLDDT > 70 share | V-family accuracy | SRR |
|---|---|---|---|---|
| Greedy (codebase default)         | ~0.005           | ~98%             | high              | low (templates) |
| Stochastic, $T{=}1.0, p{=}0.95$    | 0.051            | **96.7%**        | **99.98%**        | 0.476           |
| Stochastic, $T{=}1.3, p{=}0.99$    | **0.207**        | 25.9%            | 93.8%             | 0.564           |

Same trained weights, different decoders. See `../../../README.md` for the
full quality–diversity Pareto framing relative to LD4LG.