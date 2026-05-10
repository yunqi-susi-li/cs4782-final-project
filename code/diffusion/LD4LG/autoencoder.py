"""
Language Autoencoder for antibody sequences.

This module implements the "language autoencoder" block from LD4LG paper and
adapted to amino-acid sequences and trained from scratch instead of using BART from paper. 
Four sub-modules were implemented which match with Fig. 1 of the paper:

    w (tokens) ── Encoder E(.) ──► E(w)             (B, L, d_LM)
                                   │
                                   ▼
                       Compression Network f_phi    <-- Perceiver Resampler
                                   │
                                   ▼
                               x ∈ R^(ℓ × d_ae)     (B, 32, 64)       <-- the *latent*
                                   │
                                   ▼
                     Reconstruction Network g_phi   <-- maps back to d_LM
                                   │
                                   ▼
                               g_phi(x)             (B, ℓ, d_LM)
                                   │
                                   ▼
                  Decoder D(.)  (autoregressive, cross-attends to g_phi(x))
                                   │
                                   ▼
                              logits -> w̃ ≈ w

During stage 1, the autoencoder training stage,  we optimize cross-entropy between
the decoder logits and the input tokens. During stage 2 we freeze this entire thing and operate only on x in R^(ℓ × d_ae).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_utils import (
    GELUFeedForward,
    LearnedPositionalEmbedding,
    MultiHeadAttention,
    init_transformer_weights,
)



# Configuration


@dataclass
class AutoencoderConfig:
    # Vocabulary
    vocab_size: int = 24          # 4 specials + 20 amino acids
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # Sequence lengths
    max_source_len: int = 288      # p99 of paired VH+linker+VL is 253
    max_target_len: int = 288

    # Shared transformer dimensionality (= d_LM in the paper)
    dim: int = 768
    num_heads: int = 12
    ffn_mult: int = 4
    dropout: float = 0.1
    qk_norm: bool = True

    # Encoder depth (from scratch; paper uses frozen BART = 6 layers).
    encoder_layers: int = 6

    # Compression / reconstruction: paper uses 3 layers each for BART-base.
    compress_layers: int = 3
    reconstruct_layers: int = 3

    # Latent shape
    latent_len: int = 32           # ℓ in the paper
    latent_dim: int = 64           # d_ae in the paper

    # Decoder depth (from scratch; paper uses frozen BART = 6 layers).
    decoder_layers: int = 6

    # Normalize the latent to unit-variance-ish (|x|^2 = d_ae) like the paper.
    normalize_latent: bool = True


# ---------------------------------------------------------------------------
# Transformer blocks used by the four sub-modules
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Pre-LN transformer encoder block: SA + FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int, dropout: float, qk_norm: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GELUFeedForward(dim, ffn_mult, dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class PerceiverBlock(nn.Module):
    """One block of the Perceiver Resampler.

    The compression network uses *learnable latent queries Z* that
    cross-attend to the concatenation [Z ; E(w)]. This lets the queries
    extract information from the encoder features *and* exchange
    information among themselves (= self-attention on Z) in a single
    pass. Paper equation:

        Z ← Z + MHA(q=Z, kv=[Z; E(w)])

    We implement exactly this and follow with a GELU FFN. Pre-LN.
    """

    def __init__(self, dim: int, num_heads: int, ffn_mult: int, dropout: float, qk_norm: bool):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, qk_norm=qk_norm)
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = GELUFeedForward(dim, ffn_mult, dropout)

    def forward(self, z: torch.Tensor, enc: torch.Tensor, enc_mask: Optional[torch.Tensor]):
        # Concatenate Z with the encoder features -> keys/values.
        kv = torch.cat([z, enc], dim=1)
        kv_norm = self.norm_kv(kv)
        # Mask: Z positions are never padded, encoder positions might be.
        B, L_z, _ = z.shape
        if enc_mask is not None:
            padded = torch.cat(
                [
                    torch.zeros(B, L_z, dtype=enc_mask.dtype, device=enc_mask.device),
                    enc_mask,
                ],
                dim=1,
            )
        else:
            padded = None
        z = z + self.attn(self.norm_q(z), kv=kv_norm, key_padding_mask=padded)
        z = z + self.ffn(self.norm_ff(z))
        return z


class DecoderBlock(nn.Module):
    """Pre-LN causal decoder block: causal SA + cross-attn + FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int, dropout: float, qk_norm: bool):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout, qk_norm=qk_norm)
        self.norm_ca = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout, qk_norm=qk_norm)
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = GELUFeedForward(dim, ffn_mult, dropout)

    def forward(self, x, memory, memory_mask=None):
        x = x + self.self_attn(self.norm_sa(x), causal=True)
        x = x + self.cross_attn(self.norm_ca(x), kv=memory, key_padding_mask=memory_mask)
        x = x + self.ffn(self.norm_ff(x))
        return x


# ---------------------------------------------------------------------------
# The four sub-modules
# ---------------------------------------------------------------------------

class LanguageEncoder(nn.Module):
    """Bidirectional transformer encoder over amino-acid tokens."""

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=cfg.pad_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_source_len, cfg.dim)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(cfg.dim, cfg.num_heads, cfg.ffn_mult, cfg.dropout, cfg.qk_norm)
                for _ in range(cfg.encoder_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.pad_id = cfg.pad_id

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, L). Returns hidden states (B, L, d) and padding mask (B, L).
        pad_mask = tokens.eq(self.pad_id)
        h = self.pos(self.embed(tokens))
        h = self.emb_drop(h)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad_mask)
        return self.final_norm(h), pad_mask


class CompressionNetwork(nn.Module):
    """Perceiver Resampler mapping E(w) ∈ R^(L x d_LM) to x ∈ R^(ℓ x d_ae)."""

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.latent_queries = nn.Parameter(
            torch.randn(1, cfg.latent_len, cfg.dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                PerceiverBlock(cfg.dim, cfg.num_heads, cfg.ffn_mult, cfg.dropout, cfg.qk_norm)
                for _ in range(cfg.compress_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.to_latent = nn.Linear(cfg.dim, cfg.latent_dim)

    def forward(self, enc: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        B = enc.size(0)
        z = self.latent_queries.expand(B, -1, -1).contiguous()
        for block in self.blocks:
            z = block(z, enc, enc_mask)
        z = self.final_norm(z)
        return self.to_latent(z)  # (B, ℓ, d_ae)


class ReconstructionNetwork(nn.Module):
    """Small encoder that maps the latent back to d_LM features.

    Paper: "we project x ∈ R^(ℓ×d_ae) back up to dimension d_LM, add
    learnable absolute position embeddings, and pass it through a
    standard transformer model to obtain features g_phi(x)."
    """

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.up = nn.Linear(cfg.latent_dim, cfg.dim)
        self.pos = LearnedPositionalEmbedding(cfg.latent_len, cfg.dim)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(cfg.dim, cfg.num_heads, cfg.ffn_mult, cfg.dropout, cfg.qk_norm)
                for _ in range(cfg.reconstruct_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.up(x))
        for layer in self.layers:
            h = layer(h)
        return self.final_norm(h)  # (B, ℓ, d_LM)


class LanguageDecoder(nn.Module):
    """Autoregressive transformer decoder.

    Takes the teacher-forced target tokens and cross-attends to the
    reconstruction features g_phi(x). At training time this is the only
    module that ever sees the target tokens; at sampling time we feed
    it the BOS and decode greedily or with beam search.
    """

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=cfg.pad_id)
        self.pos = LearnedPositionalEmbedding(cfg.max_target_len, cfg.dim)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(cfg.dim, cfg.num_heads, cfg.ffn_mult, cfg.dropout, cfg.qk_norm)
                for _ in range(cfg.decoder_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.dim)
        # Output projection to vocab. We deliberately do NOT tie weights
        # with the embedding: the input and output spaces are small (24
        # tokens) so sharing brings little value and complicates init.
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.pad_id = cfg.pad_id

    def forward(self, target_tokens: torch.Tensor, memory: torch.Tensor, memory_mask=None):
        h = self.pos(self.embed(target_tokens))
        h = self.emb_drop(h)
        for layer in self.layers:
            h = layer(h, memory, memory_mask=memory_mask)
        h = self.final_norm(h)
        return self.lm_head(h)

    @torch.no_grad()
    def greedy_generate(
        self,
        memory: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        B = memory.size(0)
        ids = torch.full((B, 1), bos_id, device=memory.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        for _ in range(max_len - 1):
            logits = self.forward(ids, memory)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            # Once a row has emitted EOS, keep padding it.
            nxt = torch.where(done.unsqueeze(-1), torch.zeros_like(nxt), nxt)
            ids = torch.cat([ids, nxt], dim=1)
            done = done | (nxt.squeeze(-1) == eos_id)
            if done.all():
                break
        return ids


# ---------------------------------------------------------------------------
# Full autoencoder wrapper
# ---------------------------------------------------------------------------

class LanguageAutoencoder(nn.Module):
    """Encoder + compression + reconstruction + decoder, all end-to-end trained."""

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = LanguageEncoder(cfg)
        self.compress = CompressionNetwork(cfg)
        self.reconstruct = ReconstructionNetwork(cfg)
        self.decoder = LanguageDecoder(cfg)
        init_transformer_weights(self)

    # -------------- latent utilities --------------

    def _normalize_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Scale so that |x_i|^2 = d_ae on average.

        Paper (sec 3.1): 'we can normalize the latent vectors along the
        feature dimension so that |x_i|^2_2 = d_ae'. We do this as a
        hard projection per-token so the diffusion model always sees
        spheres of fixed radius, which stabilizes training at large
        guidance weights.
        """
        d = x.size(-1)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return x * (d ** 0.5) / norm

    # -------------- high level calls --------------

    def encode(self, source_tokens: torch.Tensor) -> torch.Tensor:
        """Tokens -> latent x (optionally norm-constrained)."""
        enc, pad = self.encoder(source_tokens)
        x = self.compress(enc, pad)
        if self.cfg.normalize_latent:
            x = self._normalize_latent(x)
        return x

    def decode(
        self,
        x: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Latent -> decoder logits (teacher forced)."""
        memory = self.reconstruct(x)
        return self.decoder(target_tokens, memory)

    @torch.no_grad()
    def generate_from_latent(
        self,
        x: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Latent -> generated tokens (greedy)."""
        memory = self.reconstruct(x)
        max_len = max_len or self.cfg.max_target_len
        return self.decoder.greedy_generate(memory, self.cfg.bos_id, self.cfg.eos_id, max_len)

    def forward(
        self,
        source_tokens: torch.Tensor,
        decoder_input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward pass.

        Args:
            source_tokens:       (B, L_src) int64 with <bos>..<eos><pad>
            decoder_input_tokens:(B, L_tgt) int64 = target shifted right (starts with <bos>)
            target_tokens:       (B, L_tgt) int64 = the tokens we want to predict

        Returns:
            loss: scalar, cross-entropy ignoring PAD
            logits: (B, L_tgt, vocab)
        """
        x = self.encode(source_tokens)
        logits = self.decode(x, decoder_input_tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_tokens.view(-1),
            ignore_index=self.cfg.pad_id,
        )
        return loss, logits
