from __future__ import annotations

import copy

import torch


class EMATeacher:


    def __init__(self, model: torch.nn.Module, decay: float = 0.99):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: torch.nn.Module) -> None:
        msd = student.state_dict()
        tsd = self.model.state_dict()
        for name, value in tsd.items():
            src = msd[name].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(src.to(value.device), alpha=1.0 - self.decay)
            else:
                value.copy_(src.to(value.device))

    @torch.no_grad()
    def encode(self, *args, **kwargs):
        self.model.eval()
        return self.model(*args, **kwargs)

