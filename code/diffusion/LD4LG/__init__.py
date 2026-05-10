"""
ab_ld4lg
========
Latent Diffusion for Paired Antibody Sequence Generation.

A from-scratch reproduction of "Latent Diffusion for Language Generation"
(Lovelace et al., NeurIPS 2023) adapted to paired (VH + linker + VL)
antibody sequences with multi-conditional (isotype, V-gene family, light
chain locus) classifier-free guidance.

The architecture follows the paper's core design:
- Language Autoencoder: encoder + Perceiver Resampler compression +
  reconstruction network + AR decoder. Everything trained from scratch on
  amino-acid sequences (alphabet size ~25).
- Denoising Network: 12-layer pre-LN transformer with GeGLU, adaptive
  layer norm conditioned on time + class embeddings, dense connections
  between early/late layers, QK-RMSNorm, and self-conditioning (p=0.5).
- Continuous latent diffusion with cosine noise schedule, v-prediction
  parameterization, and 250-step DDPM sampling.
"""

__version__ = "0.1.0"
