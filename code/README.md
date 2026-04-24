# `code/` — Source

```
code/
├── baselines/               # P0 / P1 / P2 — see baselines/README.md
├── diffusion/               # v1 / v3 / v4 / v5 — see diffusion/README.md
│   └── v4_dplm2/            # detailed DPLM-2 re-implementation — see its README
├── common/                  # data, alignment, LoRA module, metrics, diffusion utils
└── configs/                 # one YAML per run
```

Entry points:

```bash
python -m code.baselines.esm_lora.train   --config code/configs/p2_esm_lora_region.yaml
python -m code.diffusion.v4_dplm2.train   --config code/configs/v4_dplm2.yaml
```

To add a new experiment track: create `code/<family>/<new_track>/`, add a
matching YAML in `code/configs/`, and add a row to the ablation table in
`code/diffusion/README.md` (or `code/baselines/README.md`).
