"""
Transformer denoiser network x_theta(z_t, t, c).

This is the beast described in Section 3.2 + appendix E.2 of LD4LG.
Every feature is implemented exactly as specified:

  * Pre-LN transformer, 12 layers, dim=768, 12 heads, GeGLU FFN.
  * Query-Key RMSNorm in every attention layer (helps large LRs).
  * Absolute learned positional embeddings on the latent sequence.
  * Alpha conditioning: sinusoidal embedding of alpha_t -> MLP -> time
    embedding. The time embedding is (a) added to every position of the
    input sequence and (b) fed into adaptive layer normalization at the
    output of every feed-forward block.
  * Self-conditioning: the network is additionally conditioned on its
    own previous estimate x̃_s of x by concatenating it to the noisy
    latent along the feature dimension. When the estimate is not
    available we concatenate a learned "empty" embedding.
  * Dense connections between layers: following Bao et al. 2023 (U-ViT
    style), every late layer receives a skip from the matching early
    layer. The paper sets dense_connections=3, meaning the last 3
    layers each consume one skip from the first 3 layers.
  * Multi-condition class embeddings: separate learnable tables for
    isotype, V-gene family, and light chain locus. Each has a null slot
    for classifier-free guidance. We sum the three into one conditioning
    vector and add it to the time embedding.

The I/O is:

    x_theta(z_t, t, x_self, y_iso, y_vfam, y_loc) -> x̂ in R^(ℓ x d_ae)

We follow the paper's x-prediction interface internally (the module
returns an estimate of the clean latent). The v-prediction conversion
happens in diffusion.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .nn_utils import (
    AdaLayerNorm,
    GeGLU,
    LearnedPositionalEmbedding,
    MultiHeadAttention,
    RMSNorm,
    init_transformer_weights,
    sinusoidal_embedding,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DenoiserConfig:
    # Latent shape (must match AutoencoderConfig.latent_len / latent_dim).
    latent_len: int = 32
    latent_dim: int = 64

    # Transformer hyperparameters.
    dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ffn_mult: int = 4
    dropout: float = 0.1
    qk_norm: bool = True

    # Dense connections: the last `dense_connections` layers each receive
    # a skip from the first `dense_connections` layers.
    dense_connections: int = 3

    # Conditioning.
    # Each table has K real classes plus one "null" slot (last index)
    # that is used by classifier-free guidance. The training script
    # overrides these with ds.num_* so the defaults here are informative
    # only.
    num_isotypes: int = 8          # 7 real (IGHG, IGHM, IGHA, IGHD, IGHE, Bulk, Other) + 1 null
    num_v_families: int = 9        # 8 real (IGHV1..IGHV7, Other) + 1 null
    num_light_loci: int = 4        # 3 real (K, L, Other) + 1 null

    # Self-conditioning. If False we don't concat x_self to the input.
    self_conditioning: bool = True


# ---------------------------------------------------------------------------
# Building block: one denoiser layer (SA + FFN with AdaLN)
# ---------------------------------------------------------------------------

class DenoiserBlock(nn.Module):
    """Pre-LN transformer block used inside the denoiser."""

    def __init__(self, cfg: DenoiserConfig, cond_dim: int):
        super().__init__()
        # Self-attention sub-block.
        self.norm_sa = nn.LayerNorm(cfg.dim)
        self.attn = MultiHeadAttention(
            cfg.dim, cfg.num_heads, cfg.dropout, qk_norm=cfg.qk_norm
        )
        # FFN sub-block. AdaLN happens AFTER the FFN, before the residual add.
        self.norm_ff = nn.LayerNorm(cfg.dim)
        self.ff = GeGLU(cfg.dim, cfg.ffn_mult, cfg.dropout)
        self.adaln = AdaLayerNorm(cfg.dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_sa(x))
        # FFN with adaptive layer norm on the output (following paper).
        ff_out = self.ff(self.norm_ff(x))
        ff_out = self.adaln(ff_out, cond)
        x = x + ff_out
        return x


# ---------------------------------------------------------------------------
# Multi-condition classifier-free guidance embeddings
# ---------------------------------------------------------------------------

class MultiClassConditioning(nn.Module):
    """Three independent categorical conditions, each with a null slot.

    During training we randomly mask each condition independently with
    probability `cond_drop_prob`, replacing its label with the null
    index. At inference time we can mix guided and unguided predictions
    by setting each condition to either its true value or the null
    index.
    """

    def __init__(self, cfg: DenoiserConfig, dim: int, cond_drop_prob: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.cond_drop_prob = cond_drop_prob

        self.iso_emb = nn.Embedding(cfg.num_isotypes, dim)
        self.vfam_emb = nn.Embedding(cfg.num_v_families, dim)
        self.loc_emb = nn.Embedding(cfg.num_light_loci, dim)

        nn.init.normal_(self.iso_emb.weight, std=0.02)
        nn.init.normal_(self.vfam_emb.weight, std=0.02)
        nn.init.normal_(self.loc_emb.weight, std=0.02)

        # Convenient aliases for null indices (last slot of each table).
        self.null_iso = cfg.num_isotypes - 1
        self.null_vfam = cfg.num_v_families - 1
        self.null_loc = cfg.num_light_loci - 1

    def _maybe_drop(self, labels: torch.Tensor, null_idx: int) -> torch.Tensor:
        if not self.training or self.cond_drop_prob <= 0.0:
            return labels
        mask = torch.rand(labels.shape, device=labels.device) < self.cond_drop_prob
        return torch.where(mask, torch.full_like(labels, null_idx), labels)

    def forward(
        self,
        iso: torch.Tensor,
        vfam: torch.Tensor,
        loc: torch.Tensor,
        drop_all: bool = False,
    ) -> torch.Tensor:
        """Return a (B, dim) conditioning vector summing the three embeddings.

        If `drop_all` is True we force every condition to its null slot
        (used at inference for the unconditional pass in CFG).
        """
        if drop_all:
            iso = torch.full_like(iso, self.null_iso)
            vfam = torch.full_like(vfam, self.null_vfam)
            loc = torch.full_like(loc, self.null_loc)
        else:
            iso = self._maybe_drop(iso, self.null_iso)
            vfam = self._maybe_drop(vfam, self.null_vfam)
            loc = self._maybe_drop(loc, self.null_loc)
        return self.iso_emb(iso) + self.vfam_emb(vfam) + self.loc_emb(loc)


# ---------------------------------------------------------------------------
# The denoiser
# ---------------------------------------------------------------------------

class Denoiser(nn.Module):
    """x_theta(z_t, alpha_t, x_self, classes) -> x̂."""

    def __init__(self, cfg: DenoiserConfig, cond_drop_prob: float = 0.1):
        super().__init__()
        self.cfg = cfg

        # Input projection: maps (latent_dim [+ latent_dim for self-cond]) -> dim.
        in_dim = cfg.latent_dim * (2 if cfg.self_conditioning else 1)
        self.in_proj = nn.Linear(in_dim, cfg.dim)

        # Learned positional embedding across the ℓ latent positions.
        self.pos = LearnedPositionalEmbedding(cfg.latent_len, cfg.dim)

        # Alpha-conditioning path: sin(alpha_t) -> MLP -> time_emb.
        # The embedding dim for sinusoidal features is cfg.dim / 4 per the paper's
        # spirit; we use cfg.dim here for simplicity.
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.dim * 4, cfg.dim),
        )

        # Class conditioning (multi-head: isotype + V-family + light locus).
        self.class_cond = MultiClassConditioning(cfg, cfg.dim, cond_drop_prob=cond_drop_prob)

        # Learned "empty" self-cond embedding (used when x_self isn't provided).
        if cfg.self_conditioning:
            self.empty_self_cond = nn.Parameter(
                torch.zeros(1, 1, cfg.latent_dim)
            )

        # Transformer stack.
        # The conditioning dim for AdaLN is cfg.dim (time_emb + class_emb are summed).
        self.blocks = nn.ModuleList(
            [DenoiserBlock(cfg, cond_dim=cfg.dim) for _ in range(cfg.num_layers)]
        )

        # Dense connections: create a skip projection for each connection so
        # the channel dimension stays at cfg.dim after concatenation.
        self.dense_k = cfg.dense_connections
        if self.dense_k > 0:
            assert cfg.num_layers >= 2 * self.dense_k, (
                "Need num_layers >= 2 * dense_connections"
            )
            # One projection per dense connection: concat(hidden, skip) -> hidden.
            self.dense_proj = nn.ModuleList(
                [nn.Linear(cfg.dim * 2, cfg.dim) for _ in range(self.dense_k)]
            )

        self.final_norm = nn.LayerNorm(cfg.dim)
        self.out_proj = nn.Linear(cfg.dim, cfg.latent_dim)

        init_transformer_weights(self)

    # ---- helpers ----

    def _alpha_embedding(self, alpha_t: torch.Tensor) -> torch.Tensor:
        # alpha_t in [0, 1], shape (B,). We sine-embed at dim cfg.dim then MLP.
        sin = sinusoidal_embedding(alpha_t, self.cfg.dim)
        return self.time_mlp(sin)

    # ---- forward ----

    def forward(
        self,
        z_t: torch.Tensor,            # (B, ℓ, d_ae) noisy latent
        alpha_t: torch.Tensor,        # (B,) in [0, 1]
        iso: torch.Tensor,            # (B,) long
        vfam: torch.Tensor,           # (B,) long
        loc: torch.Tensor,            # (B,) long
        x_self: Optional[torch.Tensor] = None,  # (B, ℓ, d_ae) or None
        force_uncond: bool = False,             # drop all classes to null
    ) -> torch.Tensor:
        B, L, _ = z_t.shape
        cfg = self.cfg

        # 1) Input projection, optionally with self-cond concatenation.
        if cfg.self_conditioning:
            if x_self is None:
                x_self = self.empty_self_cond.expand(B, L, -1)
            inp = torch.cat([z_t, x_self], dim=-1)
        else:
            inp = z_t
        h = self.in_proj(inp)
        h = self.pos(h)

        # 2) Build conditioning vector: time_emb + class_emb.
        time_emb = self._alpha_embedding(alpha_t)                 # (B, dim)
        class_emb = self.class_cond(iso, vfam, loc, drop_all=force_uncond)  # (B, dim)
        cond = time_emb + class_emb

        # 3) Inject time_emb into the sequence (paper: "we add this time
        #    embedding to the input sequence"). We only add time, not class,
        #    to keep the class injection localized to AdaLN on FF output.
        h = h + time_emb.unsqueeze(1)

        # 4) Run the transformer, capturing early-layer outputs for dense skips.
        # We use U-Net-style mirroring (U-ViT, Bao et al. 2023): the LAST
        # late layer consumes the FIRST early layer's output, so the
        # network is symmetric around its midpoint.
        early_outputs: list[torch.Tensor] = []
        n_layers = cfg.num_layers
        k = self.dense_k
        for i, block in enumerate(self.blocks):
            # Dense connection: layer (n_layers - 1 - j) mirrors layer j.
            if k > 0 and i >= n_layers - k:
                j = (n_layers - 1) - i          # 0..k-1, reverse
                skip = early_outputs[j]
                h = self.dense_proj[j](torch.cat([h, skip], dim=-1))
            h = block(h, cond)
            if k > 0 and i < k:
                early_outputs.append(h)

        # 5) Output projection back to d_ae.
        h = self.final_norm(h)
        x_hat = self.out_proj(h)
        return x_hat
