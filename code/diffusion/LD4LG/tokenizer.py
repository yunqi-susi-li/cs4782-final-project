"""
Amino-acid tokenizer for antibody sequences.

Vocabulary layout (fixed indices so checkpoints are portable):
    0  <pad>     padding
    1  <bos>     beginning of sequence (decoder prompt)
    2  <eos>     end of sequence
    3  <unk>     any non-canonical residue
    20 standard AAs: A C D E F G H I K L M N P Q R S T V W Y
    (indices 4..23)

That is 24 total tokens. We keep the size small on purpose: the paper
trains the autoencoder with a 32x64 latent, so a compact vocabulary
keeps the decoder's output projection small and easy to learn.

Everything is pure Python -- no huggingface dep -- so the tokenizer is
picklable and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

# The 20 canonical amino acids in a fixed order.
_AA = "ACDEFGHIKLMNPQRSTVWY"

# Special tokens -- order defines their IDs.
_SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")


@dataclass(frozen=True)
class AATokenizer:
    """Character-level tokenizer for antibody amino-acid sequences."""

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    @property
    def vocab_size(self) -> int:
        return len(_SPECIALS) + len(_AA)

    @property
    def itos(self) -> List[str]:
        return list(_SPECIALS) + list(_AA)

    @property
    def stoi(self) -> dict:
        return {s: i for i, s in enumerate(self.itos)}

    # ---- encoding ----

    def encode(
        self,
        seq: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        stoi = self.stoi
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in seq:
            ids.append(stoi.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(
        self,
        seqs: Sequence[str],
        max_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Encode a batch with right-padding. Returns (ids, attention_mask)."""
        ids_batch: list[list[int]] = []
        mask_batch: list[list[int]] = []
        for s in seqs:
            ids = self.encode(s, add_bos=add_bos, add_eos=add_eos)
            if len(ids) > max_len:
                ids = ids[: max_len - 1] + [self.eos_id] if add_eos else ids[:max_len]
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            ids = ids + [self.pad_id] * (max_len - len(ids))
            ids_batch.append(ids)
            mask_batch.append(mask)
        return ids_batch, mask_batch

    # ---- decoding ----

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> str:
        itos = self.itos
        chars: list[str] = []
        for i in ids:
            if i < 0 or i >= len(itos):
                continue
            tok = itos[i]
            if strip_special and tok in _SPECIALS:
                # stop at <eos>, skip <pad>/<bos>/<unk>
                if tok == "<eos>":
                    break
                continue
            chars.append(tok)
        return "".join(chars)


# Singleton for convenience (you can also just instantiate AATokenizer()).
TOKENIZER = AATokenizer()
