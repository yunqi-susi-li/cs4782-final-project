# Ed Post Draft — Updated Project Scope (Follow-up to #302)

**Suggested title:**
> Updated project scope — pivot to re-implementing DPLM-2 for BCR (follow-up to #302)

---

Hi, this is a follow-up to #302 on our CS 4782 final project proposal.

Since the original proposal, our project has evolved substantially based on what we found while setting up the ESM baselines. I want to confirm that the updated scope is still acceptable before we lock things in for the poster/report deliverables.

## What changed since #302

In #302 we proposed **self-implemented LoRA (+ DoRA ablation) on a pretrained protein LM for one TCR task**. We started executing that plan and completed two full ESM-LoRA runs on BCR (heavy-chain) sequences (P1: plain MLM, P2: region-aware conditioned MLM). Both runs finished, but the val loss plateaued around 0.36 and the task (masked AA recovery across the full variable region) turned out not to be where the interesting signal lives for immune-receptor modeling.

This pushed us toward the **CDR3 region specifically**, where V(D)J recombination + SHM make the generative problem much richer. After reading around, we found that the question **"given germline template + V/D/J/isotype/SHM, what is the observed CDR3?"** is a natural fit for **masked discrete diffusion**, not MLM — and this is exactly the regime that DPLM / DPLM-2 (ByteDance, ICLR 2025) operates in.

So our current, updated plan is:

- **Main paper (re-implementation target):** DPLM-2 — *"DPLM-2: A Multimodal Diffusion Protein Language Model"* (Wang et al., ICLR 2025). Core ideas we re-implement: absorbing-state discrete diffusion, MDLM-style importance weighting `w(t) = -log(ᾱ)/(1-ᾱ)`, and the shared-token-stream design where different tasks correspond to different masking patterns over one joint stream.
- **Domain adaptation:** DPLM-2 uses sequence + structure as the two modalities. We replace the structure modality with the **germline CDR3 template**, so the two streams become `[germline CDR3] ⊕ [observed CDR3]`, with V/D/J/isotype/SHM header tokens.
- **Extension / novel contribution:** An explicit **germline-edit-track decomposition** — a per-position auxiliary head that classifies each observed AA as `TEMPLATED / SUBSTITUTED / INSERTED / TRIMMED` via Needleman-Wunsch alignment to the germline. This makes `observed = germline + V(D)J edits + SHM` an explicit modeling target rather than an implicit latent.
- **Task:** Conditional CDR3 generation / denoising on paired BCR data (OAS, 2.28M unique heavy-chain sequences after dedup+cluster, with <0.001% train/val/test leakage). We condition on V, D, J gene, isotype, SHM rate, and germline CDR3.
- **Baselines (all already run end-to-end):**
  1. **Frozen ESM + linear head** (the #302 baseline, kept intact)
  2. **Self-implemented LoRA on ESM** with region-aware conditioning (P2 above)
  3. **Plain masked discrete diffusion** without germline (v1A/v1B — our own ablation of "what if we drop DPLM-2's ideas")
  4. **D3PM + germline cross-attention** (v3 — our clean baseline before adding MDLM weighting / shared stream)
  5. **DPLM-2 adapted + edit head** (v4 — the full re-implementation)
- **Evaluation metrics:** masked-AA recovery on the OBS region (the DPLM-style metric), per-edit-class accuracy (TEMPLATED/SUBSTITUTED/INSERTED), and generative metrics (length distribution match, motif conservation, V/J consistency of sampled CDR3s).

We are keeping the **self-implemented LoRA from #302 alive as one of the baselines** (#2 above) — we implement the LoRA modules and merge logic in PyTorch ourselves, not via PEFT/HuggingFace, consistent with the scope you approved in #302. The shift is that the **main paper** is now DPLM-2 rather than the LoRA paper, and the **extension** is the germline-edit decomposition rather than DoRA.

## Why we think the pivot is worth it

Results so far (all on the same held-out split, same recovery metric on corrupted AA positions):

| Experiment | Type | Best AA Recovery | Params | Notes |
|---|---|---|---|---|
| P1 / P2 (ESM-LoRA) | MLM baseline | val=0.3636 / 0.3671 | 11.9M / 15.2M | Full variable region |
| v1A / v1B | Plain CDR3 diffusion | ~46-47% | 7.5M / 58.3M | No germline |
| v3_prod | +germline cross-attn | 56.89% | 10.2M | First clean baseline |
| v3_scale | +depth / width | 57.69% | 12.5-22.8M | Scaling helps little |
| **v4_dplm2** | **DPLM-2 + edit head** | **77.00%** | **19.0M** | **+20pp jump** |

The +20pp breakthrough comes from the DPLM-2 ideas (shared token stream so germline is directly visible to self-attention, plus MDLM importance weighting) combined with the edit-head auxiliary loss. This is a much more interesting story to write up than "LoRA vs DoRA on one TCR task" would have been, and it keeps the core thing you approved in #302 (self-implemented PEFT, from-scratch PyTorch) as a named baseline.

## Questions

Could you confirm whether:

1. **Scope pivot is OK:** Switching the main paper from the LoRA paper to **DPLM-2**, and switching the downstream task from TCR chain-pairing to **BCR CDR3 conditional generation**, is acceptable as long as we cite #302 and keep the self-implemented LoRA as a live baseline?
2. **Extension framing is OK:** Treating the **germline-edit-track decomposition** (our own contribution — adds an auxiliary 4-way classification head and reframes the task as explicit edit-op prediction) as the "extension / ablation" slot that DoRA used to occupy?
3. **Re-implementation depth is sufficient:** We re-implement DPLM-2's training objective (absorbing-state diffusion + MDLM weighting), shared-stream layout, and iterative unmasking sampler from scratch in PyTorch — we do not use any DPLM codebase checkpoints. We do initialize the AA token embedding table from scratch (24 tokens) since DPLM-2's structure tokenizer doesn't apply. Does this level of re-implementation meet the bar you're looking for?
4. **2-person group scope:** The pivot doesn't change group size. Does the overall scope (ESM-LoRA baselines + 4-variant diffusion ablation + DPLM-2 re-implementation + germline-edit extension + BCR-specific evaluation) still look appropriate for a 2-person project, or is it now too ambitious and we should trim (e.g. drop the LoRA baseline from the final report)?

Thank you!

---

## References to cite in the final report

- Wang et al., 2024/2025 — DPLM-2 (main re-implementation target)
- Sahoo et al., 2024 — MDLM (source of the `w(t) = -log(ᾱ)/(1-ᾱ)` weighting we adopt)
- Austin et al., 2021 — D3PM (the absorbing-state diffusion framework both DPLM-2 and MDLM build on)
- Hu et al., 2021 — LoRA (#302's original paper, now a baseline)
- OAS (Kovaltsuk et al., 2018) — data source

---

**Notes for yourself (not part of the Ed post):**

- Double-check the DPLM-2 paper title / authors / venue before posting — the memo above says ICLR 2025 but confirm on arXiv / OpenReview first.
- If the TA pushes back on #3 (re-implementation depth), the fallback is to also reproduce one numerical result from the DPLM-2 paper directly (e.g. their sequence-only benchmark on UniRef or antibody data if reported) before pivoting to CDR3. That's the standard "reproduce a specific result from the paper" interpretation of the project brief.
- If the TA pushes back on #1 (scope shift too large), the fallback is to keep the TCR task as a secondary benchmark — our v4 code works on TCR data with no changes because the token stream is chain-agnostic.
- Keep the tone consistent with #302 — bulleted, concrete, end with numbered questions.
