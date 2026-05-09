"""
Antibody dataset from shared Google Drive.

Pipeline:
    raw tar.gz -> extract -> read TSV -> tokenize -> pack into int16 memmap
    +------------------------------+
    |  preprocess.py  (one-time)   |   <-- heavy, done once offline
    +------------------------------+
                 |
                 v
       memmap/{split}.tokens.npy     int16  (N, max_len)
       memmap/{split}.lengths.npy    int32  (N,)
       memmap/{split}.iso.npy        int8   (N,)   isotype class id
       memmap/{split}.vfam.npy       int8   (N,)   V-family class id
       memmap/{split}.loc.npy        int8   (N,)   light locus class id

    +------------------------------+
    |  PairedAntibodyDataset       |   <-- used by the training loops
    +------------------------------+

Keeping the on-disk format this simple (plain numpy memmaps) means the

"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from .tokenizer import AATokenizer



# Fixed label mappings to match with DenoiserConfig.num_*

ISOTYPES = ["IGHG", "IGHM", "IGHA", "IGHD", "IGHE", "Bulk", "Other"]     # 7 -> null at 6? No: null = 7-1 =6
V_FAMILIES = ["IGHV1", "IGHV2", "IGHV3", "IGHV4", "IGHV5", "IGHV6", "IGHV7", "Other"]   # 8 + null slot appended at runtime
LIGHT_LOCI = ["K", "L", "Other"]                                         # 3 + null slot

# Reserve the LAST index of each category as the null slot.
# Preprocessing returns indices in [0, K-2]; the null index is K-1.
def _isotype_to_idx(iso: str) -> int:
    iso = (iso or "").strip()
    if iso in ("IGHG", "IGHG1", "IGHG2", "IGHG3", "IGHG4"):
        iso = "IGHG"
    if iso in ISOTYPES[:-1]:
        return ISOTYPES.index(iso)
    return ISOTYPES.index("Other")


def _vfam_to_idx(v_call_heavy_gene: str) -> int:
    # "IGHV3-23" -> "IGHV3"
    v = (v_call_heavy_gene or "").strip().split("-")[0].split("*")[0].upper()
    if v in V_FAMILIES[:-1]:
        return V_FAMILIES.index(v)
    return V_FAMILIES.index("Other")


def _locus_to_idx(locus_light: str) -> int:
    loc = (locus_light or "").strip().upper()
    if loc in LIGHT_LOCI[:-1]:
        return LIGHT_LOCI.index(loc)
    return LIGHT_LOCI.index("Other")


# Dataset

@dataclass
class DatasetPaths:
    """Where the memmap files live for one split."""

    root: Path
    split: str  # 'train' / 'val' / 'test'

    @property
    def tokens(self) -> Path:
        return self.root / f"{self.split}.tokens.npy"

    @property
    def lengths(self) -> Path:
        return self.root / f"{self.split}.lengths.npy"

    @property
    def iso(self) -> Path:
        return self.root / f"{self.split}.iso.npy"

    @property
    def vfam(self) -> Path:
        return self.root / f"{self.split}.vfam.npy"

    @property
    def loc(self) -> Path:
        return self.root / f"{self.split}.loc.npy"

    @property
    def meta(self) -> Path:
        return self.root / f"{self.split}.meta.json"


class PairedAntibodyDataset(Dataset):
    """Loads pre-tokenized paired antibody latents + class labels from memmaps."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_len: Optional[int] = None,
    ):
        paths = DatasetPaths(Path(root), split)
        # Load header json to know shapes.
        import json
        with open(paths.meta, "r") as f:
            meta = json.load(f)
        self.max_len = meta["max_len"] if max_len is None else max_len
        self.n = meta["n"]

        # Memmap views. int16 is enough: vocab size << 32768.
        self.tokens = np.memmap(
            paths.tokens, dtype=np.int16, mode="r", shape=(self.n, meta["max_len"])
        )
        self.lengths = np.memmap(paths.lengths, dtype=np.int32, mode="r", shape=(self.n,))
        self.iso = np.memmap(paths.iso, dtype=np.int8, mode="r", shape=(self.n,))
        self.vfam = np.memmap(paths.vfam, dtype=np.int8, mode="r", shape=(self.n,))
        self.loc = np.memmap(paths.loc, dtype=np.int8, mode="r", shape=(self.n,))

        self.num_isotypes = meta["num_isotypes"] + 1        # +1 null slot
        self.num_v_families = meta["num_v_families"] + 1
        self.num_light_loci = meta["num_light_loci"] + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        # Trim to max_len (tokens file is already right-padded to max_len).
        tok = torch.from_numpy(np.ascontiguousarray(self.tokens[i, : self.max_len])).long()
        length = int(self.lengths[i])
        return {
            "tokens": tok,                                 # (max_len,) long
            "length": length,                              # int
            "iso": torch.tensor(int(self.iso[i]), dtype=torch.long),
            "vfam": torch.tensor(int(self.vfam[i]), dtype=torch.long),
            "loc": torch.tensor(int(self.loc[i]), dtype=torch.long),
        }



# Collate: turns a list of dicts into a batched dict with the right tensors
# for both the autoencoder (source/target/decoder_input) and the diffusion
# training loop (just tokens + labels -- latents are computed on the fly or
# precomputed).


def make_collate_fn(pad_id: int, bos_id: int):
    def collate(examples: list[dict]) -> dict:
        tokens = torch.stack([e["tokens"] for e in examples], dim=0)
        iso = torch.stack([e["iso"] for e in examples], dim=0)
        vfam = torch.stack([e["vfam"] for e in examples], dim=0)
        loc = torch.stack([e["loc"] for e in examples], dim=0)

        # Decoder input = tokens shifted right (prepend <bos>, drop last token).
        dec_in = tokens.clone()
        dec_in[:, 1:] = tokens[:, :-1]
        dec_in[:, 0] = bos_id

        # Target = tokens themselves (so the model predicts position t from
        # positions <t). PAD positions are ignored by cross_entropy.
        target = tokens.clone()

        return {
            "source_tokens": tokens,
            "decoder_input": dec_in,
            "target_tokens": target,
            "iso": iso,
            "vfam": vfam,
            "loc": loc,
        }

    return collate
