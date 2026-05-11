
from dataclasses import dataclass
import torch
import torch.nn as nn

# shared transformer blocks live in the LD4LG package so the two tracks
# use literally the same primitives
from ..LD4LG.nn_utils import (
    GeGLU,
    LearnedPositionalEmbedding,
    MultiHeadAttention,
    init_transformer_weights,
)


@dataclass
class DPLMConfig:
    vocab_size: int = 25            # 4 specials + 20 AAs + 1 mask
    pad_id: int = 0
    mask_id: int = 24
    max_len: int = 288

    # transformer
    dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ffn_mult: int = 4
    dropout: float = 0.1
    qk_norm: bool = True

    # class conditioning to match dataset's num_* + 1 null slot
    num_isotypes: int = 8      
    num_v_families: int = 9    
    num_light_loci: int = 4    


class DPLMBlock(nn.Module):
    """Pre-LN bidirectional transformer block: self-attn + GeGLU FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.norm_sa = nn.LayerNorm(cfg.dim)
        self.attn    = MultiHeadAttention(cfg.dim, cfg.num_heads, cfg.dropout, qk_norm=cfg.qk_norm)
        self.norm_ff = nn.LayerNorm(cfg.dim)
        self.ff      = GeGLU(cfg.dim, cfg.ffn_mult, cfg.dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm_sa(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.norm_ff(x))
        return x


class MultiClassConditioning(nn.Module):
    """Three independent class embeddings with per-condition CFG dropout.
    Same contract as LD4LG.denoiser.MultiClassConditioning so train/inference
    CFG behaviour is identical between the two tracks. Y.L.

    Modify and add joint CFG ablation for comparison. May 9th, 2026. Y.L.
    """

    # def __init__(self, cfg, dim, cond_drop_prob=0.1):
    #     super().__init__()
    #     self.cond_drop_prob = cond_drop_prob

    def __init__(self, cfg: DPLMConfig, dim: int, cond_drop_prob: float = 0.1, joint_cfg: bool = False):
        super().__init__()
        self.cfg = cfg
        self.cond_drop_prob = cond_drop_prob
        self.joint_cfg = joint_cfg

        self.iso_emb  = nn.Embedding(cfg.num_isotypes,    dim)
        self.vfam_emb = nn.Embedding(cfg.num_v_families,  dim)
        self.loc_emb  = nn.Embedding(cfg.num_light_loci,  dim)
        for emb in (self.iso_emb, self.vfam_emb, self.loc_emb):
            nn.init.normal_(emb.weight, std=0.02)

        self.null_iso  = cfg.num_isotypes    - 1
        self.null_vfam = cfg.num_v_families  - 1
        self.null_loc  = cfg.num_light_loci  - 1

    def _maybe_drop(self, labels, null_idx):
        if not self.training or self.cond_drop_prob <= 0.0:
            return labels
        m = torch.rand(labels.shape, device=labels.device) < self.cond_drop_prob
        return torch.where(m, torch.full_like(labels, null_idx), labels)

    def forward(self, iso, vfam, loc, drop_all=False):
        if drop_all:
            iso  = torch.full_like(iso,  self.null_iso)
            vfam = torch.full_like(vfam, self.null_vfam)
            loc  = torch.full_like(loc,  self.null_loc)
        # Modify and add elif for joint CFG ablation for comparison. May 9th, 2026. Y.L.
        elif self.joint_cfg:
            # same Bernoulli mask drops all 3 conditions together
            if self.training and self.cond_drop_prob > 0.0:
                mask = torch.rand(iso.shape, device=iso.device) < self.cond_drop_prob
                iso  = torch.where(mask, torch.full_like(iso,  self.null_iso),  iso)
                vfam = torch.where(mask, torch.full_like(vfam, self.null_vfam), vfam)
                loc  = torch.where(mask, torch.full_like(loc,  self.null_loc),  loc)
        else: #indepndent CFG (drop indepenntly)
            iso  = self._maybe_drop(iso,  self.null_iso)
            vfam = self._maybe_drop(vfam, self.null_vfam)
            loc  = self._maybe_drop(loc,  self.null_loc)
        return self.iso_emb(iso) + self.vfam_emb(vfam) + self.loc_emb(loc)


class DPLM(nn.Module):
    """Discrete absorbing diffusion denoiser over AA tokens."""

    def __init__(self, cfg, cond_drop_prob=0.1):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=cfg.pad_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_len, cfg.dim)
        self.emb_drop = nn.Dropout(cfg.dropout)
        # Modify for joint CFG ablation for comparison. May 9th, 2026. Y.L.
        # self.cond = MultiClassConditioning(cfg, cfg.dim, cond_drop_prob=cond_drop_prob)
        self.cond = MultiClassConditioning(cfg, cfg.dim, cond_drop_prob=cond_drop_prob, joint_cfg=joint_cfg)

        self.layers = nn.ModuleList([DPLMBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        init_transformer_weights(self)

    def forward(self, tokens, iso, vfam, loc, force_uncond=False):
        # tokens : (B, L) int64, possibly with [MASK] at some positions
        # iso/vfam/loc : (B,) int64 class labels
        # returns logits (B, L, V)
        pad_mask = tokens.eq(self.cfg.pad_id)

        h = self.emb_drop(self.pos(self.tok_emb(tokens)))
        cond_vec = self.cond(iso, vfam, loc, drop_all=force_uncond)
        h = h + cond_vec.unsqueeze(1)

        for layer in self.layers:
            h = layer(h, key_padding_mask=pad_mask)

        return self.lm_head(self.final_norm(h))