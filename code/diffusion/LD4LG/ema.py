"""
Exponential moving average of model parameters.

Used in stage 2 (diffusion training) with decay 0.9999 as per the paper.
The EMA weights are the ones we use for sampling; the online model
parameters are the ones we train.
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p_s, p_m in zip(self.shadow.parameters(), model.parameters()):
            p_s.mul_(self.decay).add_(p_m.detach(), alpha=1 - self.decay)
        # Also sync buffers (e.g., running stats) by straight copy.
        for b_s, b_m in zip(self.shadow.buffers(), model.buffers()):
            b_s.copy_(b_m)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)

    @property
    def model(self) -> nn.Module:
        return self.shadow
