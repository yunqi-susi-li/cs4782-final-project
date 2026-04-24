# `code/baselines/` — ESM-based baselines

All three baselines here are written from scratch in PyTorch — LoRA modules and
merge logic live in `code/common/lora.py`, **not** via PEFT / HuggingFace.

| Dir | Name | Objective | Params | Val AA Recovery |
|---|---|---|---|---|
| `esm_linear/` | **P0** — frozen ESM + linear head | MLM (linear probe) | — | — (reference) |
| `esm_lora/`   | **P1** — LoRA on ESM (plain)        | MLM              | 11.9 M | 0.3636 |
| `esm_lora/`   | **P2** — LoRA on ESM (region-aware) | MLM + region tokens | 15.2 M | 0.3671 |

## Track purposes

- **P0** grounds the LM-only lower bound on OAS heavy-chain variable regions.
- **P1 / P2** are the baselines inherited from proposal #302 — kept alive in
  the revised scope as the "self-implemented PEFT" deliverable. P2 adds
  per-residue region tokens (FR1/CDR1/.../FR4) to test whether telling the
  model *which* region each masked position is in helps.

## Finding

Across P1 → P2 the recovery moves only +0.35 pp despite +3.3 M params. This
motivated the shift from full-variable-region MLM to CDR3-specific generation,
which in turn motivated the pivot to discrete-diffusion models (see
`code/diffusion/README.md`).
