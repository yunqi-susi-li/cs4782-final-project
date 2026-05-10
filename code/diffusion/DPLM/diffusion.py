"""
Forward (corrupting):
    t ~ Uniform[0, 1];  alpha(t) = cos^2(pi t / 2);  gamma(t) = 1 - alpha(t)
    each non-special token is independently replaced with [MASK] w.p. gamma(t).

Reverse (sampling):
    start fully [MASK]; iterate t: 1 -> 0 over T steps. At each step the
    model predicts logits at every position, we rank masked positions by
    confidence, and unmask the top-k (where k matches the schedule's
    target #unmasked at the next time).

Loss:
    cross-entropy on masked positions only, averaged over the batch.
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha(t):
    #alpha(t) = cos^2(pi t / 2). gamma(t) = 1 - alpha(t) is the mask prob
    return torch.cos(0.5 * math.pi * t).pow(2)


def cosine_gamma(t):
    return 1.0 - cosine_alpha(t)


@dataclass
class DPLMDiffusionConfig:
    num_sampling_steps: int = 100
    cfg_weight: float = 2.0          # 1.0 = no guidance
    pad_id: int = 0
    mask_id: int = 24
    bos_id: int = 1
    eos_id: int = 2


class DPLMDiffusion(nn.Module):
    ##Wraps a DPLM model with the absorbing-diffusion training loss 
    # and the iterative-unmask sampler

    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def loss(self, tokens, iso, vfam, loc):
        B, L = tokens.shape
        device = tokens.device
        pad_mask = tokens.eq(self.cfg.pad_id)

        t = torch.rand(B, device=device)
        gamma = cosine_gamma(t)

        # never mask pad / bos / eos -- those aren't part of the AA stream
        keep_visible = pad_mask | tokens.eq(self.cfg.bos_id) | tokens.eq(self.cfg.eos_id)
        rand = torch.rand(B, L, device=device)
        mask = (rand < gamma.unsqueeze(1)) & ~keep_visible

        # rare-edge: if any row ended up with 0 masked positions (small gamma
        # + short seq), force at least one so the loss is well-defined
        no_masked = mask.sum(dim=1) == 0
        if no_masked.any():
            for b in torch.nonzero(no_masked, as_tuple=True)[0]:
                fallback = (~keep_visible[b]).nonzero(as_tuple=True)[0]
                if fallback.numel() > 0:
                    mask[b, fallback[0]] = True

        corrupted = torch.where(mask, torch.full_like(tokens, self.cfg.mask_id), tokens)
        logits = self.model(corrupted, iso, vfam, loc)

        # CE on masked positions only
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tokens.view(-1),
            reduction="none",
        ).view(B, L)
        return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(
        self,
        batch_size, seq_len,
        iso, vfam, loc,
        device,
        num_steps=None, cfg_weight=None,
        temperature=1.0, top_p=1.0,
        sample_mode="stochastic",       # "stochastic" or "greedy"
    ):
        """Iteratively unmask from a fully-masked sequence.
        sample_mode='stochastic' (default): per-position categorical sample with
        temperature + nucleus filtering. Matches the DPLM-1 paper recipe.
        sample_mode='greedy' is for ablation only and pure argmax causes
        mode collapse on antibody data from previous run(same condition -> same output). Y.L.
        """
        T = num_steps or self.cfg.num_sampling_steps
        w = cfg_weight if cfg_weight is not None else self.cfg.cfg_weight
        use_cfg = (w != 1.0)

        # init: bos at 0, eos at last, everything else [MASK]
        tokens = torch.full((batch_size, seq_len), self.cfg.mask_id, device=device, dtype=torch.long)
        tokens[:, 0]  = self.cfg.bos_id
        tokens[:, -1] = self.cfg.eos_id
        active = torch.ones_like(tokens, dtype=torch.bool)
        active[:, 0]  = False
        active[:, -1] = False

        ts = torch.linspace(1.0, 0.0, T + 1, device=device)
        total_active = active.sum(dim=1).float()

        for step in range(T):
            t_next = ts[step + 1]
            gamma_next = cosine_gamma(t_next)

            target_masked_next = torch.round(gamma_next * total_active).long()
            currently_masked = (tokens == self.cfg.mask_id) & active
            n_masked_now = currently_masked.sum(dim=1)
            n_to_unmask = (n_masked_now - target_masked_next).clamp(min=0)

            # forward (with optional CFG)
            logits_cond = self.model(tokens, iso, vfam, loc, force_uncond=False)
            if use_cfg:
                logits_uncond = self.model(tokens, iso, vfam, loc, force_uncond=True)
                logits = logits_uncond + w * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

            # token sampling: stochastic (paper default) vs greedy (ablation)
            if sample_mode == "greedy":
                pred = logits.argmax(dim=-1)
            else:
                scaled = logits / max(temperature, 1e-6)
                if top_p < 1.0:
                    sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
                    sorted_probs = sorted_logits.softmax(dim=-1)
                    keep = sorted_probs.cumsum(dim=-1) <= top_p
                    keep[..., 0] = True   # always keep top-1
                    sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
                    scaled = torch.full_like(scaled, float("-inf"))
                    scaled.scatter_(-1, sorted_idx, sorted_logits)
                probs = scaled.softmax(dim=-1)
                B_, L_, V_ = probs.shape
                pred = torch.multinomial(probs.reshape(-1, V_), 1).reshape(B_, L_)

            # rank by model confidence (clean argmax-prob), but write the sampled token
            conf = logits.softmax(dim=-1).max(dim=-1).values.masked_fill(~currently_masked, -1.0)
            for b in range(batch_size):
                k = int(n_to_unmask[b].item())
                if k <= 0:
                    continue
                topk = torch.topk(conf[b], k=k).indices
                tokens[b, topk] = pred[b, topk]

        # cleanup: anything still [MASK] gets argmax of a final pass
        still_masked = (tokens == self.cfg.mask_id) & active
        if still_masked.any():
            logits_final = self.model(tokens, iso, vfam, loc, force_uncond=False)
            tokens = torch.where(still_masked, logits_final.argmax(dim=-1), tokens)
        return tokens