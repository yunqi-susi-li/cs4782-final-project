# Re-implementing DPLM-2 for BCR CDR3 Conditional Generation

**CS 4782 — Introduction to Deep Learning, Final Project (Spring 2026)**
Cornell University

**Authors:** Yunqi Li, Yonglin Zhang

> TL;DR — We re-implement **DPLM-2** (Wang et al., ICLR 2025) from scratch in
> PyTorch and adapt it to antibody heavy-chain CDR3 generation, replacing
> DPLM-2's `[sequence] ⊕ [structure]` shared stream with
> `[germline CDR3] ⊕ [observed CDR3]`. We add a novel **germline-edit-track
> auxiliary head** that classifies each observed AA as
> `TEMPLATED / SUBSTITUTED / INSERTED / TRIMMED`, making
> `observed = germline + V(D)J edits + SHM` an explicit modeling target.
> Against our own pre-DPLM-2 baselines, adding DPLM-2's ideas yields a **+20 pp**
> jump in masked-AA recovery on held-out OAS heavy-chain data (57.7% → 77.0%).

---

## 1. Introduction

This repository re-implements and extends

> **DPLM-2: A Multimodal Diffusion Protein Language Model.**
> Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., Gu, Q.
> Nanjing University & ByteDance Research, ICLR 2025.
> Project page: <https://bytedance.github.io/dplm/dplm-2>

DPLM-2 is an absorbing-state discrete-diffusion protein language model that
operates over a shared stream of `[sequence] ⊕ [structure]` tokens, trained with
MDLM-style importance weighting `w(t) = -log(ᾱ_t)/(1 - ᾱ_t)` (Sahoo et al., 2024)
on top of the D3PM framework (Austin et al., 2021). DPLM-2 extends the original
DPLM with a lookup-free-quantized structure tokenizer so that one joint
transformer learns the joint distribution of sequence and structure, as well
as their marginals and conditionals.

Our work additionally explores latent diffusion for sequences via

> **Latent Diffusion for Language Generation (LD4LG).**
> Lovelace, J., Kishore, V., Wan, C., Shekhtman, E., Weinberger, K. Q.
> Cornell University, NeurIPS 2023.
> Code: <https://github.com/justinlovelace/latent-diffusion-for-language>

LD4LG learns a continuous diffusion model in the latent space of a fixed
pretrained encoder-decoder language model. It was covered in CS 4782 lectures
by Prof. Weinberger; our exploratory `v5_latent` track (see
[`code/diffusion/v5_latent/`](./code/diffusion/v5_latent/)) adapts it to
antibody CDR3s.

We are a 2-person group (Yunqi Li, Yonglin Zhang). Our work picks up from the
baseline approved in proposal #302 (self-implemented LoRA on ESM-2 for a TCR
task) and pivots the main re-implementation target to **DPLM-2** and the
downstream task to **BCR heavy-chain CDR3 conditional generation** on OAS
data. The pivot was approved by the course staff; the pivot memo is kept at
[`report/ED_POST_DRAFT_v4.md`](./report/ED_POST_DRAFT_v4.md) for the record.

## 2. Chosen Result

We reproduce DPLM-2's **masked-AA recovery** metric on a corrupted-positions
denoising task — the headline metric DPLM-2 uses to evaluate how well its
shared-stream + MDLM-weighted training recovers masked residues given visible
context. *(Original paper: reported for the sequence-only and sequence+structure
settings; we reproduce the spirit of that metric, adapted to the
observed-vs-germline CDR3 setting described above.)*

Why this result: it is the most direct probe of DPLM-2's central claim that
absorbing-state diffusion with MDLM weighting and a shared token stream is a
better training recipe than vanilla MLM / plain D3PM, and it is the same metric
we had already instrumented for the #302 ESM-LoRA baselines — making the
before/after comparison apples-to-apples.

## 3. GitHub Contents

```
cs4782-final-project/
├── README.md                 ← this file
├── LICENSE                   ← MIT
├── requirements.txt
├── .gitignore
├── code/                     ← all source
│   ├── baselines/{esm_linear, esm_lora}
│   ├── diffusion/{v1_plain, v3_d3pm_germline, v4_dplm2, v5_latent}
│   ├── common/               ← data, alignment, LoRA, metrics, diffusion utils
│   └── configs/              ← one YAML per run
├── data/                     ← OAS download + preprocessing pipeline (raw/processed gitignored)
├── results/{figures,tables,logs}
├── poster/                   ← poster.pdf (due Apr 30 / May 5)
└── report/                   ← group_topic_2page_report.pdf (due May 12)
```

Each subdirectory has its own `README.md` with details.

## 4. Re-implementation Details

- **Model:** shared-stream Transformer denoiser consuming
  `[V/D/J/isotype/SHM header tokens] ⊕ [germline CDR3] ⊕ [observed CDR3]`.
  ~19 M params for v4.
- **Tokenizer:** 24 AA-level tokens (20 AA + `<PAD>` + `<MASK>` + `<BOS>` + `<EOS>`),
  re-initialized from scratch — DPLM-2's structure tokenizer does not apply.
- **Training objective:** absorbing-state discrete diffusion cross-entropy over
  the observed-CDR3 segment, weighted by `w(t) = -log(ᾱ_t)/(1 - ᾱ_t)` (MDLM),
  plus the **edit-head auxiliary loss** `λ_edit · CE(edit_label, edit_pred)`.
- **Sampler:** iterative unmasking — start with all observed positions masked,
  unmask top-`k_t` by argmax (or low-temperature sample) at each step.
- **Dataset:** OAS paired heavy chains, deduped + clustered at 95% identity →
  **2.28 M unique sequences**; 80/10/10 cluster-level split;
  **< 0.001%** sequence-level leakage.
- **Metrics:** masked-AA recovery on corrupted OBS positions;
  per-edit-class accuracy (4-way confusion);
  generative metrics — length distribution match, motif conservation,
  V/J consistency of sampled CDR3s.
- **Baselines (all implemented from scratch in PyTorch, no PEFT/HuggingFace
  LoRA, no DPLM codebase):** P0 (frozen ESM + linear), P1/P2 (self-implemented
  LoRA on ESM, plain + region-aware), v1A/v1B (plain masked discrete diffusion
  without germline), v3_prod/v3_scale (D3PM + germline cross-attention).
- **Modifications vs the DPLM-2 paper:** structure modality replaced by
  germline CDR3 template; V/D/J/isotype/SHM conditioning injected as header
  tokens; edit-track auxiliary head added on top of the last transformer layer
  (our extension).

See [`code/diffusion/v4_dplm2/README.md`](./code/diffusion/v4_dplm2/README.md)
for the full spec.

## 5. Reproduction Steps

```bash
# 1. clone
git clone https://github.com/<org-or-user>/cs4782-final-project.git
cd cs4782-final-project

# 2. environment (Python 3.10+; CUDA 11.8+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. data (downloads ~40 GB OAS → dedups + clusters → parquet shards)
python -m code.common.data --stage download  --out data/raw/
python -m code.common.data --stage preprocess --in data/raw/ --out data/processed/ \
    --cluster-identity 0.95 --min-cdr3-len 5 --max-cdr3-len 30

# 4. train the main model (v4 — DPLM-2 + edit head)
python -m code.diffusion.v4_dplm2.train --config code/configs/v4_dplm2.yaml

# 5. evaluate
python -m code.diffusion.v4_dplm2.eval   --config code/configs/v4_dplm2.yaml \
       --ckpt results/logs/v4_dplm2/best.ckpt

# 6. sample conditional CDR3s
python -m code.diffusion.v4_dplm2.sample --config code/configs/v4_dplm2.yaml \
       --ckpt results/logs/v4_dplm2/best.ckpt --n 10000 \
       --out results/tables/v4_samples.parquet
```

**Baselines** follow the same pattern — replace `v4_dplm2` and the config path.

**Hardware:**
- Full v4 training: 1 × A100-40GB or 1 × A6000, ~8–12 hours for our setting.
- P0 / P1 / P2 baselines: 1 × consumer GPU (e.g. RTX 3090, 24 GB) is sufficient.
- CPU-only will work for the smoke tests but is not practical for training.

## 6. Results / Insights

Preliminary results on the same held-out split, same recovery metric on
corrupted AA positions:

| Experiment | Type | Best AA Recovery | Params | Notes |
|---|---|---|---|---|
| P1 / P2 (ESM-LoRA)       | MLM baseline            | val 0.3636 / 0.3671 | 11.9 M / 15.2 M | full V-region |
| v1A / v1B                | Plain CDR3 diffusion    | ~46–47%             | 7.5 M / 58.3 M  | no germline |
| v3_prod                  | + germline cross-attn   | **56.89%**          | 10.2 M          | first clean baseline |
| v3_scale                 | + depth / width         | 57.69%              | 12.5–22.8 M     | scaling ≈ flat |
| **v4_dplm2**             | **DPLM-2 + edit head**  | **77.00%**          | **19.0 M**      | **+20 pp** |

Key takeaway: the +20 pp breakthrough is not from scale — it is from DPLM-2's
training recipe (shared token stream so germline is directly visible to
self-attention, plus MDLM `w(t)` weighting), combined with the edit-head
auxiliary loss that regularizes the model toward an explicit
`germline + V(D)J edits + SHM` factorization.

*(These numbers are from our pre-report exploratory runs and may be updated in
the final report; v5 latent-diffusion numbers will be added if that track
produces publishable results.)*

## 7. Conclusion

Re-implementing DPLM-2 from scratch and adapting it to BCR CDR3 generation gave
us a concrete mechanistic lesson: **how conditioning information is fused into
the denoiser matters more than raw model capacity.** Giving the model germline
CDR3 via cross-attention (v3) buys us ~10 pp; giving it via a shared token
stream under an MDLM-weighted absorbing-state loss (v4) buys us ~20 pp more.
The edit-head extension further reframes the generation problem as explicit
edit-op prediction, producing interpretable per-position predictions
(TEMPLATED / SUBSTITUTED / INSERTED / TRIMMED) alongside the AA distribution.

## 8. References

1. Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., Gu, Q. **DPLM-2: A Multimodal Diffusion Protein Language Model.** ICLR 2025. <https://bytedance.github.io/dplm/dplm-2>
2. Lovelace, J., Kishore, V., Wan, C., Shekhtman, E., Weinberger, K. Q. **Latent Diffusion for Language Generation.** NeurIPS 2023. <https://github.com/justinlovelace/latent-diffusion-for-language>
3. Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A. M., Kuleshov, V. **Simple and Effective Masked Diffusion Language Models (MDLM).** NeurIPS 2024.
4. Austin, J., Johnson, D. D., Ho, J., Tarlow, D., van den Berg, R. **Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM).** NeurIPS 2021.
5. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR 2022.
6. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., Rives, A. **Evolutionary-scale prediction of atomic-level protein structure (ESM-2).** Science, 2023.
7. Kovaltsuk, A., Leem, J., Kelm, S., Snowden, J., Deane, C. M., Krawczyk, K. **Observed Antibody Space: A Resource for Data Mining Next-Generation Sequencing of Antibody Repertoires.** J. Immunol., 2018.

## 9. Acknowledgements

This project was developed by **Yunqi Li** and **Yonglin Zhang** as coursework
for **CS 4782 — Introduction to Deep Learning** at Cornell University
(Spring 2026), under the guidance of the course staff. We thank the
instructors and TAs for feedback on proposal #302 and for approving the scope
pivot documented in
[`report/ED_POST_DRAFT_v4.md`](./report/ED_POST_DRAFT_v4.md).

We gratefully acknowledge the authors of DPLM-2 (Wang et al., ByteDance
Research & Nanjing University) — whose paper is the primary re-implementation
target — and of MDLM, D3PM, LoRA, and ESM-2, whose work this project builds
upon. Our exploratory `v5_latent` track adapts **Latent Diffusion for
Language Generation** (Lovelace, Kishore, Wan, Shekhtman, & Weinberger,
Cornell, NeurIPS 2023), which **Prof. Weinberger — one of the instructors of
this course — covered in CS 4782 lectures**; we thank him for introducing the
method and for the broader framing of diffusion for discrete sequences that
made this project possible. We thank the curators of the Observed Antibody
Space (OAS) database for making large-scale paired BCR data available.
