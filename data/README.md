# `data/` — OAS dataset and preprocessing

Raw OAS dumps are large (tens of GB) and **not** committed to this repo.
This directory holds notes on data sources and licensing.

## Source

Training and evaluation data come from the **Observed Antibody Space (OAS)**,
a public database of B-cell-receptor sequences:

- Olsen, T. H., Boyles, F., Deane, C. M. *Observed Antibody Space: A diverse
  database of cleaned, annotated, and translated unpaired and paired antibody
  sequences.* **Protein Science 31(1):141–146, 2022.**
- Database: <https://opig.stats.ox.ac.uk/webapps/oas/>

We use the **paired** subset (both heavy and light chains sequenced from
the same B cell).

## Preprocessing

After downloading OAS paired chains as `.tar.gz` exports, run:

```bash
python -m code.diffusion.LD4LG.preprocess \
    --archives <oas_export.tar.gz> \
    --out processed/ \
    --max-len 288
```

This produces the int16 token memmaps consumed by both diffusion tracks.
See [`../code/data_preprocessing/README.md`](../code/data_preprocessing/README.md)
for the full pipeline description.

## Corpus statistics

After MMseqs2 deduplication at 95% / 90% sequence identity:

- **2.17M** paired V_H ⊕ GGGGSGGGGS ⊕ V_L chains
- 0% sequence-level train/test leakage
- Held-out 18-cell stratification (3 isotypes × 3 V-families × 2 light loci):
  **200 reference + 512 generated** sequences per cell