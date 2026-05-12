# Third-Party Attributions

This repository is an **independent re-implementation**. No source code from
the upstream projects below was copied into this repository; they are listed
as scholarly references whose ideas and architectures we re-implemented in
PyTorch.

## Re-implemented from

- **DPLM** — Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., Gu, Q.
  *Diffusion Language Models Are Versatile Protein Learners.* ICLR 2024.
  Upstream: <https://github.com/bytedance/dplm> — **Apache License 2.0**

- **LD4LG** — Lovelace, J., Kishore, V., Wan, C., Shekhtman, E.,
  Weinberger, K. Q. *Latent Diffusion for Language Generation.*
  NeurIPS 2023.
  Upstream: <https://github.com/justinlovelace/latent-diffusion-for-language>
  — **MIT License**

## External tools invoked at runtime (not vendored)

These are installed separately by the user and called via subprocess. No
source code from these projects is included in this repository.

- **MMseqs2** — sequence similarity reduction
  <https://github.com/soedinglab/MMseqs2>
- **HMMER 3.x** — Pfam Ig V-set domain validity oracle
  <http://hmmer.org/>
- **IgFold** — antibody pLDDT foldability oracle
  <https://github.com/Graylab/IgFold>

## Data

- **Observed Antibody Space (OAS)** — Olsen, T. H., Boyles, F., Deane, C. M.
  *Observed Antibody Space: A diverse database of cleaned, annotated, and
  translated unpaired and paired antibody sequences.* Protein Science 31(1),
  2022.
  <https://opig.stats.ox.ac.uk/webapps/oas/>

---

Should any source code from the above repositories be incorporated in a
future revision, the corresponding upstream license terms (including
attribution and notice obligations) will apply to those portions in
addition to this project's MIT license.
