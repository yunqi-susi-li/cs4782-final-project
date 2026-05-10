"""
Shared neural-network building blocks.

Why this file exists: the paper specifies a very particular combination
of tricks (pre-LN + RMSNorm on Q/K, GeGLU FFN, sinusoidal alpha
embedding -> MLP, adaptive layer norm conditioning). Rather than
re-implement these inside autoencoder.py and denoiser.py I factor them
out here so both modules use the *identical* primitives.

Everything is plain PyTorch -- no dependency on einops or external
libraries.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

    Cheaper than LayerNorm (no mean subtraction) and used by the paper
    specifically on queries and keys in attention -- a.k.a. 'QK-Norm' --
    which stabilizes training at larger learning rates.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


# ---------------------------------------------------------------------------
# Gated activations
# ---------------------------------------------------------------------------

class GeGLU(nn.Module):
    """GeGLU FFN (Shazeer, 2020).

    Standard FFN is `Linear -> GELU -> Linear`. GeGLU splits the first
    linear into two halves and gates one with GELU of the other. In the
    paper this is the activation used in the diffusion denoiser.
    Parameter count stays the same if you use 2/3 width compared to a
    standard FFN (here we use the simpler 1x width variant for clarity).
    """

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * dim
        # fc1 outputs 2*hidden so we can split into (x, gate).
        self.fc1 = nn.Linear(dim, 2 * hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(self.dropout(F.gelu(gate) * x))


class GELUFeedForward(nn.Module):
    """Plain GELU feed-forward used by the autoencoder modules."""

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Attention with QK-RMSNorm
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional QK-RMSNorm.

    Supports:
      - self-attention (kv is None)
      - cross-attention (q comes from one stream, kv from another)
      - causal masking (for the AR decoder)
      - arbitrary key padding masks
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
        kv_dim: Optional[int] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        kv_dim = kv_dim if kv_dim is not None else dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(kv_dim, dim)
        self.v_proj = nn.Linear(kv_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = dropout
        self.proj_dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, S_kv) bool, True = pad
        causal: bool = False,
    ) -> torch.Tensor:
        B, S_q, _ = x.shape
        kv_source = x if kv is None else kv
        S_kv = kv_source.shape[1]

        q = self.q_proj(x).view(B, S_q, self.num_heads, self.head_dim)
        k = self.k_proj(kv_source).view(B, S_kv, self.num_heads, self.head_dim)
        v = self.v_proj(kv_source).view(B, S_kv, self.num_heads, self.head_dim)

        # QK-RMSNorm (applied per-head).
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Move head to the front for scaled_dot_product_attention: (B, H, S, D).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # True positions are PAD -> mask them out.
            # Expand to broadcast: (B, 1, 1, S_kv)
            attn_mask = torch.zeros(
                B, 1, 1, S_kv, dtype=q.dtype, device=q.device
            )
            attn_mask = attn_mask.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=causal,
        )
        # Back to (B, S, D).
        out = out.transpose(1, 2).contiguous().view(B, S_q, self.num_heads * self.head_dim)
        return self.proj_dropout(self.out_proj(out))


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    """Absolute learned positional embedding -- used by the paper."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.size(1)
        return x + self.pe[:, :S]


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Standard sinusoidal embedding used to embed alpha_t (or t).

    Args:
        t: (B,) float tensor with values in some finite range (e.g. [0, 1]).
        dim: output dimensionality (must be even).

    Returns:
        (B, dim) tensor.
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Adaptive LayerNorm (AdaLN) used to inject time + class conditioning
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale/shift come from a conditioning vector.

    Following the paper: 'we add this time embedding to the input
    sequence AND apply adaptive layer normalization conditioned on the
    time embedding to the output of every feedforward layer'. We use
    AdaLN only on the FFN outputs (classic DiT-lite setup) to keep
    parameter count sane. The conditioning MLP projects cond -> (scale,
    shift) per feature dim.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        # init scale output to 1, shift to 0: start as identity.
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D), cond: (B, cond_dim)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Weight init helpers
# ---------------------------------------------------------------------------

def init_transformer_weights(module: nn.Module, std: float = 0.02) -> None:
    """Standard small-std init used throughout the codebase."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)
