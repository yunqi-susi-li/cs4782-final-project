"""
Continuous latent diffusion in the x_theta parameterisation, trained with
v-prediction loss, following LD4LG sections 2, 3.2 and appendix A.

Given a clean latent x ∈ R^(ℓ x d_ae):

  Forward process at continuous time t ∈ [0, 1]:

      z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps     eps ~ N(0, I)

      with alpha_t = cos(0.5 * pi * t)^2      (cosine schedule)

  Parameterisations (all equivalent mathematically but differ in
  conditioning / loss weighting):

      x_pred:  predict clean x̂ directly
      eps:     predict noise epŝ
      v:       predict v = sqrt(alpha_t) * eps - sqrt(1 - alpha_t) * x

  Given any of these three, the other two can be recovered in closed
  form. We use v-prediction (Salimans & Ho 2022) because the loss is
  well-behaved at both ends of the schedule. At training time we
  convert the output of our x_pred denoiser to v and compare to the
  target v.

  Reverse process (DDPM / ancestral sampler):

      We split [1, 0] into T+1 timesteps 1 = t_1 > t_2 > ... > t_T = 0
      (linearly). At each step we:
          1) predict x̂ from z_t
          2) compute the mean of q(z_s | z_t, x̂) using the analytic
             posterior (appendix A eq for mu_Q)
          3) sample z_s ~ N(mean, sigma^2 I) with sigma^2 = 1 - alpha_{t|s}.

  Classifier-free guidance:

      x̂_t = w * x_cond + (1 - w) * x_uncond

  Applied on the *clean-data estimate* x̂, not on v or eps, so the
  rescaling of x̂ to the latent norm ball (|x|^2 = d_ae) still works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def cosine_alpha(t: torch.Tensor) -> torch.Tensor:
    """alpha_t = cos(pi/2 * t)^2 with t in [0, 1]."""
    return torch.cos(0.5 * math.pi * t).pow(2)


def cosine_schedule_with_shift(t: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """Shifted cosine schedule (Hoogeboom / Chen 2023).

    alpha_t^2 / (1 - alpha_t^2) = SNR = sigmoid(log(snr_cos) + 2 log s)
    s = 1 reduces to the plain cosine schedule. Used for seq2seq in the
    paper (not strictly needed here but kept for completeness).
    """
    alpha_cos = cosine_alpha(t)
    snr = (alpha_cos ** 2) / (1 - alpha_cos ** 2).clamp_min(1e-12)
    log_snr_shift = snr.log() + 2.0 * math.log(s)
    alpha_sq = torch.sigmoid(log_snr_shift)
    return alpha_sq.sqrt()


# ---------------------------------------------------------------------------
# Parameterisation helpers
# ---------------------------------------------------------------------------

def _sqrt_alpha(alpha_t: torch.Tensor) -> torch.Tensor:
    return alpha_t.clamp_min(1e-12).sqrt()


def _sqrt_1m_alpha(alpha_t: torch.Tensor) -> torch.Tensor:
    return (1 - alpha_t).clamp_min(1e-12).sqrt()


def v_from_x_eps(alpha_t: torch.Tensor, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    a = _sqrt_alpha(alpha_t).view(-1, 1, 1)
    b = _sqrt_1m_alpha(alpha_t).view(-1, 1, 1)
    return a * eps - b * x


def x_from_z_v(alpha_t: torch.Tensor, z_t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Given a v-prediction and z_t, recover x̂."""
    a = _sqrt_alpha(alpha_t).view(-1, 1, 1)
    b = _sqrt_1m_alpha(alpha_t).view(-1, 1, 1)
    return a * z_t - b * v


def v_from_x_and_z(alpha_t: torch.Tensor, x_hat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Given x̂ and z_t, recover the implied v. Useful because our denoiser
    returns x̂ internally but the loss is v-based."""
    a = _sqrt_alpha(alpha_t).view(-1, 1, 1)
    b = _sqrt_1m_alpha(alpha_t).view(-1, 1, 1)
    # From z_t = a x + b eps and v = a eps - b x:
    # v = (a z_t - x̂) / b  (kept as a one-liner below)
    return (a * z_t - x_hat) / b.clamp_min(1e-6)


# ---------------------------------------------------------------------------
# The Gaussian diffusion object
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    # Parameterisation
    schedule: str = "cosine"       # "cosine" or "cosine_shifted"
    shift: float = 1.0             # used only if schedule == "cosine_shifted"

    # Loss type applied on v.
    loss_type: str = "l2"          # "l2" or "l1" (paper uses L1 for seq2seq)

    # Sampling
    num_sampling_steps: int = 250

    # Self-conditioning probability (training): with prob `self_cond_prob`
    # we perform a stop-gradient extra forward pass to produce x_self.
    self_cond_prob: float = 0.5

    # Norm-rescaling: if True, rescale predicted x̂ during sampling so that
    # ||x̂_i||^2 = d_ae (matches the autoencoder's normalized latent).
    rescale_predictions: bool = True
    latent_dim: int = 64

    # Classifier-free guidance weight at sampling time (1.0 = no guidance).
    cfg_weight: float = 2.0


class GaussianDiffusion(nn.Module):
    """Wraps a denoiser with the forward / reverse / loss machinery."""

    def __init__(self, denoiser: nn.Module, cfg: DiffusionConfig):
        super().__init__()
        self.denoiser = denoiser
        self.cfg = cfg

    # ---- schedule ----

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.cfg.schedule == "cosine":
            return cosine_alpha(t)
        elif self.cfg.schedule == "cosine_shifted":
            a = cosine_schedule_with_shift(t, s=self.cfg.shift)
            return a.pow(2)  # return alpha_t (squared) to match convention
        raise ValueError(self.cfg.schedule)

    # ---- rescale helpers ----

    def _rescale(self, x_hat: torch.Tensor) -> torch.Tensor:
        if not self.cfg.rescale_predictions:
            return x_hat
        d = self.cfg.latent_dim
        norm = x_hat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return x_hat * (d ** 0.5) / norm

    # ---- training loss ----

    def loss(
        self,
        x0: torch.Tensor,
        iso: torch.Tensor,
        vfam: torch.Tensor,
        loc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the v-prediction loss on a batch of clean latents."""
        B = x0.size(0)
        device = x0.device

        # Sample t ~ Uniform(0, 1) and build z_t.
        t = torch.rand(B, device=device)
        alpha_t = self.alpha(t)
        eps = torch.randn_like(x0)
        a = _sqrt_alpha(alpha_t).view(-1, 1, 1)
        b = _sqrt_1m_alpha(alpha_t).view(-1, 1, 1)
        z_t = a * x0 + b * eps
        v_target = a * eps - b * x0  # = v_from_x_eps(alpha_t, x0, eps)

        # Self-conditioning: with probability p, do an extra no-grad pass to
        # obtain x_self = sg(x_hat(z_t, alpha_t, empty_self_cond)).
        x_self = None
        if self.cfg.self_cond_prob > 0.0:
            do_selfcond = (torch.rand(1, device=device) < self.cfg.self_cond_prob).item()
            if do_selfcond:
                with torch.no_grad():
                    x_self_raw = self.denoiser(
                        z_t=z_t,
                        alpha_t=alpha_t,
                        iso=iso,
                        vfam=vfam,
                        loc=loc,
                        x_self=None,
                    )
                    x_self = x_self_raw.detach()

        # Forward through the denoiser and convert x̂ -> v.
        x_hat = self.denoiser(
            z_t=z_t,
            alpha_t=alpha_t,
            iso=iso,
            vfam=vfam,
            loc=loc,
            x_self=x_self,
        )
        v_pred = v_from_x_and_z(alpha_t, x_hat, z_t)

        if self.cfg.loss_type == "l2":
            return F.mse_loss(v_pred, v_target)
        elif self.cfg.loss_type == "l1":
            return F.l1_loss(v_pred, v_target)
        raise ValueError(self.cfg.loss_type)

    # ---- sampling ----

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        latent_len: int,
        latent_dim: int,
        iso: torch.Tensor,
        vfam: torch.Tensor,
        loc: torch.Tensor,
        device: torch.device,
        num_steps: Optional[int] = None,
        cfg_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """Run DDPM / ancestral sampling and return the final clean latent."""
        T = num_steps or self.cfg.num_sampling_steps
        w = self.cfg.cfg_weight if cfg_weight is None else cfg_weight
        use_cfg = (w != 1.0)

        # Linear grid of timesteps from 1 down to 0.
        ts = torch.linspace(1.0, 0.0, T + 1, device=device)

        # z_{t_1} ~ N(0, I)
        z_t = torch.randn(batch_size, latent_len, latent_dim, device=device)

        # Self-conditioning state.
        x_self = None

        for i in range(T):
            t = ts[i].expand(batch_size)
            s = ts[i + 1].expand(batch_size)

            alpha_t = self.alpha(t)
            alpha_s = self.alpha(s)

            # Predict x̂ (conditional).
            x_cond = self.denoiser(
                z_t=z_t,
                alpha_t=alpha_t,
                iso=iso, vfam=vfam, loc=loc,
                x_self=x_self,
                force_uncond=False,
            )

            if use_cfg:
                x_uncond = self.denoiser(
                    z_t=z_t,
                    alpha_t=alpha_t,
                    iso=iso, vfam=vfam, loc=loc,
                    x_self=x_self,
                    force_uncond=True,
                )
                x_hat = w * x_cond + (1.0 - w) * x_uncond
            else:
                x_hat = x_cond

            # Rescale to the latent norm ball.
            x_hat = self._rescale(x_hat)

            # Stash for next-step self-conditioning.
            x_self = x_hat

            # If this is the final step, just return x̂.
            if i == T - 1:
                return x_hat

            # Compute posterior mean mu_Q(z_t, x̂, s, t) and variance.
            alpha_ts = (alpha_t / alpha_s.clamp_min(1e-12))
            a_s = _sqrt_alpha(alpha_s).view(-1, 1, 1)
            a_ts = _sqrt_alpha(alpha_ts).view(-1, 1, 1)
            one_m_at = (1 - alpha_t).clamp_min(1e-12).view(-1, 1, 1)
            one_m_as = (1 - alpha_s).clamp_min(1e-12).view(-1, 1, 1)
            one_m_ats = (1 - alpha_ts).clamp_min(1e-12).view(-1, 1, 1)

            mu = (a_s * one_m_ats / one_m_at) * x_hat + (a_ts * one_m_as / one_m_at) * z_t
            # sigma^2 = 1 - alpha_{t|s} (paper's choice).
            sigma = one_m_ats.sqrt()

            noise = torch.randn_like(z_t)
            z_t = mu + sigma * noise

        return z_t  # unreachable in practice
