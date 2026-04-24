# v4 — DPLM-2 Re-implementation + Germline-Edit Head (MAIN)

**This is the main re-implementation target** of the final project.

> Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., Gu, Q.
> **DPLM-2: A Multimodal Diffusion Protein Language Model.** ICLR 2025.
> Nanjing University & ByteDance Research.
> Project: <https://bytedance.github.io/dplm/dplm-2>

## Core DPLM-2 ideas re-implemented from scratch

1. **Absorbing-state discrete diffusion** (D3PM family) over the AA alphabet
2. **MDLM importance weighting**: `w(t) = -log(ᾱ_t) / (1 - ᾱ_t)` applied to the
   per-position cross-entropy loss (Sahoo et al., 2024)
3. **Shared-token-stream layout**: a *single* transformer consumes
   `[germline CDR3] ⊕ [observed CDR3]`, with distinguishing stream embeddings.
   Different tasks correspond to different masking patterns over the joint stream —
   we only mask positions in the observed-CDR3 segment, but germline is directly
   visible to self-attention (not cross-attention).
4. **Iterative unmasking sampler**: at inference, mask all observed positions,
   then iteratively unmask `k_t` positions per step by argmax (or low-temperature
   sample) over the model's categorical prediction at each remaining mask.

## Our domain adaptation

DPLM-2 uses `[sequence] ⊕ [structure]` as its two modalities. We replace the
structure modality with **germline CDR3 template**, so the two streams become
`[germline CDR3] ⊕ [observed CDR3]` with V/D/J/isotype/SHM-rate header tokens
prepended to the sequence.

AA token embedding table is re-initialized from scratch (24 tokens: 20 AAs +
special tokens). DPLM-2's structure tokenizer does not apply.

## Our extension (the "extension" slot that DoRA used to occupy in #302)

**Germline-edit-track decomposition.** An auxiliary 4-way classification head
attached to the last transformer layer predicts, for each observed-CDR3
position, the edit operation that explains it relative to the germline template:

- `TEMPLATED`    — position matches germline after Needleman-Wunsch alignment
- `SUBSTITUTED`  — position aligned but AA differs (SHM)
- `INSERTED`     — position has no germline counterpart (N/P insertions from V(D)J)
- `TRIMMED`      — germline position aligned to a gap (V/J trimming)

The auxiliary cross-entropy loss is added to the main diffusion loss with a
weight of `λ_edit`. This makes `observed = germline + V(D)J edits + SHM` an
explicit modeling target rather than an implicit latent.

## Observed results (proposal #302 follow-up)

| Variant           | Params | Val AA Recovery |
|-------------------|--------|-----------------|
| **v4_dplm2 (full)** | **19.0M** | **77.00%** |

**+20pp jump** over v3_scale, attributed jointly to:
- shared-stream layout (germline directly in self-attention)
- MDLM importance weighting
- edit-head auxiliary loss

## Files (to be added)

- `model.py`        — shared-stream transformer + edit-head
- `diffusion.py`    — absorbing-state D3PM + **MDLM `w(t)` weighting**
- `train.py`        — joint loss: `L_diff + λ_edit · L_edit`
- `sample.py`       — iterative unmasking sampler
- `edit_head.py`    — 4-way position-wise classifier
- `README.md`       — this file

## What we do **not** use

- No DPLM / DPLM-2 codebase or checkpoints.
- No HuggingFace PEFT for the LoRA baseline (see `code/common/lora.py`).
- No pretrained structure tokenizer (germline template replaces it).
