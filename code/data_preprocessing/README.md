# `code/data_preprocessing/` — OAS → memmap pipeline

The actual preprocessing script lives at
[`../diffusion/LD4LG/preprocess.py`](../diffusion/LD4LG/preprocess.py)
and is shared between the LD4LG and DPLM tracks. This directory holds
notes on data flow and provenance.

## Input

Paired V$_H$ + V$_L$ sequences exported from the **Observed Antibody Space**
(OAS; Olsen et al., 2022) as `.tar.gz` archives containing
`06_export/{train,val,test}/{split}.metadata.tsv` files. Each row has fields
including `pair_input_seq`, `Isotype_heavy`, `v_call_heavy_gene`,
`locus_light`.

## Pipeline

1. **Filter** rows with empty `pair_input_seq` or total length (+ BOS/EOS)
   exceeding `--max-len` (default 288).
2. **MMseqs2 deduplication** at 95% / 90% sequence identity (Steinegger &
   Söding, 2017), applied externally to ensure 0% train/test leakage.
3. **Tokenize** with the 24-symbol AA vocabulary (`../diffusion/LD4LG/tokenizer.py`).
4. **Pack into memmaps** — streamed row-by-row so RAM stays bounded.

## Output

```
processed/
├── {train,val,test}.tokens.npy    int16 (N, max_len)
├── {train,val,test}.lengths.npy   int32 (N,)
├── {train,val,test}.iso.npy       int8  (N,)
├── {train,val,test}.vfam.npy      int8  (N,)
├── {train,val,test}.loc.npy       int8  (N,)
└── {train,val,test}.meta.json
```

## Running

```bash
python -m code.diffusion.LD4LG.preprocess \
    --archives <oas_export.tar.gz> \
    --out processed/ --max-len 288
```

See `--help` for additional options (`--tsv-dir`, `--splits`, `--seq-col`).