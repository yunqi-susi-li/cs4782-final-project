
# Indices 0..23 in this file are kept identical to v5_latent.tokenizer.AATokenizer 
# so the same pre-tokenized memmap data can be re-used without re-encoding.

from typing import List

_AA = "ACDEFGHIKLMNPQRSTVWY"
_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


# @dataclass(frozen=True)  Apr 27th, 2026 Y.L.: dropped dataclass to avoid rebuilding
# class DPLMTokenizer:    stoi/itos dicts on every @property access
#     pad_id: int = 0
#     bos_id: int = 1
#     eos_id: int = 2
#     unk_id: int = 3
#     mask_id: int = 24

class DPLMTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.mask_id = 24    # <mask> appended after the 20 AAs

        # specials, then AAs, then <mask>
        self.itos = list(_SPECIALS) + list(_AA) + ["<mask>"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, seq, add_bos=True, add_eos=True):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in seq:
            ids.append(self.stoi.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, strip_special=True):
        out = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if strip_special and (tok in _SPECIALS or tok == "<mask>"):
                if tok == "<eos>":
                    break
                continue
            out.append(tok)
        return "".join(out)


TOKENIZER = DPLMTokenizer()