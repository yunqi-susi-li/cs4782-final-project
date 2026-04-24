# `data/` — Datasets & preprocessing

Raw OAS dumps are large (tens of GB) and are **not** committed to this repo.
This directory holds:

1. Small, version-controlled metadata (schemas, the specific OAS studies used,
   the exact dedup/cluster parameters we ran).
2. A reproducible pipeline that, given a fresh machine, downloads and
   preprocesses the data into the split we train on.

## Obtaining the data

1. **Source:** Observed Antibody Space (OAS), paired heavy/light BCR repertoire
   subset.
   - Kovaltsuk et al., *"Observed Antibody Space: A Resource for Data Mining
     Next-Generation Sequencing of Antibody Repertoires"*, J. Immunol., 2018.
   - Project site: <https://opig.stats.ox.ac.uk/webapps/oas/>

2. Select **paired** BCR studies (heavy chain used in our experiments).

3. Run the preprocessing pipeline:
   ```bash
   python -m code.common.data --stage download --out data/raw/
   python -m code.common.data --stage preprocess \
       --in data/raw/ --out data/processed/ \
       --cluster-identity 0.95 --min-cdr3-len 5 --max-cdr3-len 30
   ```

## Final corpus

- **2.28M** unique heavy-chain sequences after dedup + 95% identity clustering.
- Train / val / test split enforced at the cluster level.
- Measured **< 0.001%** sequence-level train↔val↔test leakage.
- Per-record fields: `V_gene`, `D_gene`, `J_gene`, `isotype`, `SHM_rate`,
  `germline_CDR3` (IMGT-reconstructed), `observed_CDR3`.

## Directory layout (after preprocessing)

```
data/
├── raw/           # gitignored — OAS downloads
├── processed/     # gitignored — parquet shards + split indices
└── splits/
    ├── train.txt  # cluster IDs (small, committed)
    ├── val.txt
    └── test.txt
```
