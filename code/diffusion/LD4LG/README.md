# LD4LG — Latent Diffusion for Antibody Sequences (LD4LG adaptation)

Continuous latent diffusion for paired (VH ⊕ linker ⊕ VL) antibody sequences,
adapted from

> Lovelace, J., Kishore, V., Wan, C., Shekhtman, E., Weinberger, K. Q.
> **Latent Diffusion for Language Generation.** NeurIPS 2023.
> <https://github.com/justinlovelace/latent-diffusion-for-language>

with multi-conditional classifier-free guidance over (isotype, V-gene
family, light-chain locus). Compared against the discrete-diffusion DPLM
baseline in `DPLM/`.

## Why latent diffusion

Discrete diffusion (DPLM/) operates directly over the AA token alphabet.
Continuous latent diffusion instead first compresses each ~241 aa sequence
to a fixed-shape `32 × 64` latent via a learned autoencoder, then runs a
v-prediction Gaussian diffusion in the latent space. Two reasons to try this:

1. **Capacity decoupling.** The autoencoder (vocabulary modeling) and the
   diffusion model (distributional learning) train independently. The AE
   can be driven to near-zero CE without bottlenecking the diffusion.
2. **CS 4782 coverage.** LD4LG was covered in lecture; this is our
   exploratory implementation extended to multi-condition CFG.

## Architecture

```
                          ┌─────────────────────────────────────┐
   AA tokens ───encoder──>│  encoder Transformer + Perceiver    │──> latent z (32×64)
   (length L)             │  Resampler compression              │
                          └─────────────────────────────────────┘
                                            │
                                            ▼
                          ┌─────────────────────────────────────┐
   conditions: isotype ──>│   12-layer pre-LN Transformer       │──> v-pred ε̂
   V-family, locus        │   GeGLU + AdaLN(t, c)               │
   (CFG, p_drop=0.1)      │   QK-RMSNorm + self-conditioning    │
                          │   U-ViT-style dense skips           │
                          └─────────────────────────────────────┘
                                            │
                                            ▼ (250-step DDPM with CFG, w=2.0)
                          ┌─────────────────────────────────────┐
   sampled latent ──────> │  reconstruction + AR decoder        │──> AA tokens
                          │  (frozen autoencoder)               │
                          └─────────────────────────────────────┘
```

**Stage 1 — Language Autoencoder** (`autoencoder.py`, `train_autoencoder.py`)

- Encoder: 6-layer pre-LN Transformer (d_model = 768, 12 heads), reads BOS + AA tokens
- Compression: Perceiver Resampler with 32 learned latent queries → 32 × 64 latent
- Reconstruction: linear up-projection + 3-layer Transformer
- Decoder: 6-layer causal Transformer with cross-attention to reconstruction features
- Loss: cross-entropy on AA vocabulary (24 tokens)

**Stage 2 — Latent Diffusion** (`denoiser.py`, `diffusion.py`, `train_diffusion.py`)

- Forward kernel: continuous Gaussian, **cosine schedule**
- Parametrization: **v-prediction** (Salimans & Ho, 2022)
- Denoiser: 12-layer pre-LN Transformer with
  - GeGLU feedforward
  - Adaptive LayerNorm conditioned on (time + summed class) embedding
  - QK-RMSNorm
  - Dense skips from the first 3 layers into the last 3 (Bao et al., U-ViT)
  - Self-conditioning at p = 0.5 during training
- Classifier-free guidance: each of the 3 conditions has its own null slot
  and is dropped **independently** with p = 0.1 at training time
- Sampler: 250-step DDPM (ancestral) with CFG weight w = 2.0 by default
- Predictions are projected onto the unit-norm latent ball (`|x|² = d_ae`)
  during sampling, matching the AE's normalized latent

## Files

```
LD4LG/
├── __init__.py
├── tokenizer.py            # AA tokenizer (24 tokens)
├── nn_utils.py             # RMSNorm, GeGLU, MHA + QK-Norm, AdaLN,
│                           # learned + sinusoidal positional embeddings
├── data.py                 # PairedAntibodyDataset + collate fn (memmap-backed)
├── ema.py                  # EMA wrapper
├── ddp_utils.py            # torchrun-compatible DDP helpers
│
├── autoencoder.py          # stage-1 encoder + Perceiver + decoder
├── denoiser.py             # stage-2 latent denoiser
├── diffusion.py            # cosine schedule + v-prediction + DDPM sampler + CFG
│
├── train_autoencoder.py    # stage-1 entry point
├── train_diffusion.py      # stage-2 entry point
├── sample.py               # generate N sequences per (iso, vfam, locus) cell
├── eval.py                 # validity / linker / n-gram diversity / memorization
├── preprocess.py           # OAS tar.gz → int16 token memmaps
├── smoke_test.py           # ~1-min end-to-end shape test (CPU-friendly)
└── README.md               # this file
```

## Reproducing

Trained on a single H100 (single-GPU, bf16 AMP). The two stages share the
same preprocessed memmap directory.

```bash
# 0) Preprocess raw OAS export(s) into int16 memmaps
python -m code.diffusion.LD4LG.preprocess \
    --archives run_090_export.tar.gz \
    --out processed/ --max-len 288

# 1) Stage 1: language autoencoder, 50k steps
python -m code.diffusion.LD4LG.train_autoencoder \
    --data processed/ --out runs/ae --steps 50000

# 2) Stage 2: latent diffusion on the frozen AE, 250k steps
python -m code.diffusion.LD4LG.train_diffusion \
    --data    processed/ \
    --ae-ckpt runs/ae/autoencoder_latest.pt \
    --out     runs/diffusion --steps 250000

# 3) Sample one (iso, vfam, locus) cell
python -m code.diffusion.LD4LG.sample \
    --ae-ckpt   runs/ae/autoencoder_latest.pt \
    --diff-ckpt runs/diffusion/diffusion_latest.pt \
    --iso IGHG --vfam IGHV3 --loc K \
    --num 512 --cfg 2.0 \
    --out samples/IGHG_IGHV3_K.fasta

# 4) Eval
python -m code.diffusion.LD4LG.eval \
    --fasta samples/IGHG_IGHV3_K.fasta \
    --train-tokens processed/train.tokens.npy \
    --train-meta   processed/train.meta.json \
    --out          eval_reports/IGHG_IGHV3_K.json
```

For multi-GPU, add `torchrun --standalone --nproc_per_node=N` in front of
the train commands and divide `--batch` by N. See `ddp_utils.py`.

## Observed results

See top-level `results/` for full per-cell numbers. Headline figures:

| Metric | Value | Notes |
|---|---|---|
| Autoencoder val CE | 0.038 | confirms latent space is information-preserving |
| Test-set token recovery | 97.74% | encode → 32 × 64 → decode |
| Validity (canonical 20 AA) | 100.0% (9216 / 9216) | 18 cells × 512 samples |
| GGGGSGGGGS linker present | 99.7% | per-cell range 99.4–100% |
| Length p50 (paired) | 240 aa | training median: 241 aa |
| Exact match to training | 0 / 9216 | no memorization |
| Hamming ≤ 3 to training | 0 / 9216 | no near-memorization |
| 4-gram diversity (per cell) | 0.083 – 0.169 | IGHM cells consistently lowest |

## Differences from the LD4LG paper

| LD4LG paper | This implementation |
|---|---|
| BART autoencoder (pretrained) | Encoder + Perceiver + decoder, all trained from scratch |
| English text alphabet (~50k BPE) | Amino-acid alphabet (24 tokens) |
| Single class condition | **Three independent CFG conditions** (iso, V-fam, locus), each with its own null slot |
| Variable-length text inputs | Right-padded to 288 tokens |