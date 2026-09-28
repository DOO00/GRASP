from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class GRASPModel(nn.Module):


    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 10,
        dropout: float = 0.2,
        alpha: float = 1.0,
        beta: float = 1.0,
        backbone: str = "dual_filter",
        fusion: str = "mean",
        pre_norm_fused: bool = True,
    ):
        super().__init__()
        if backbone not in {"dual_filter", "mlp"}:
            raise ValueError("backbone must be 'dual_filter' or 'mlp'")
        if fusion not in {"mean", "gate"}:
            raise ValueError("fusion must be 'mean' or 'gate'")

        self.num_layers = int(num_layers)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.backbone = backbone
        self.fusion = fusion
        self.pre_norm_fused = bool(pre_norm_fused)

        self.text_encoder = MLPEncoder(text_dim, hidden_dim, dropout)
        self.image_encoder = MLPEncoder(image_dim, hidden_dim, dropout)
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.image_linear = nn.Linear(image_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, 2)
        self.text_decoder = nn.Linear(hidden_dim, text_dim)
        self.image_decoder = nn.Linear(hidden_dim, image_dim)

    @staticmethod
    def symmetric_softmax(z: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        scale = 1.0 / torch.sqrt(torch.tensor(float(n), dtype=z.dtype, device=z.device))
        sim = (z.t() @ z) * scale
        exp_sim = torch.exp(sim)
        row = torch.sqrt(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))
        col = torch.sqrt(exp_sim.sum(dim=0, keepdim=True).clamp_min(1e-12))
        return exp_sim / (row @ col).clamp_min(1e-12)

    def _fuse(self, zt: torch.Tensor, zv: torch.Tensor):
        if self.fusion == "gate":
            gate = F.softmax(self.gate(torch.cat([zt, zv], dim=1)), dim=1)
            fused = gate[:, :1] * zt + gate[:, 1:] * zv
        else:
            gate = zt.new_full((zt.size(0), 2), 0.5)
            fused = (zt + zv) / 2.0
        return F.normalize(fused, p=2, dim=1), gate

    def _forward_dual_filter(self, text: torch.Tensor, image: torch.Tensor, norm_adj: torch.Tensor):
        zt = F.normalize(self.text_linear(text), p=2, dim=1)
        zv = F.normalize(self.image_linear(image), p=2, dim=1)

        if self.fusion == "gate":
            gate = F.softmax(self.gate(torch.cat([zt, zv], dim=1)), dim=1)
            fused = gate[:, :1] * zt + gate[:, 1:] * zv
        else:
            gate = zt.new_full((zt.size(0), 2), 0.5)
            fused = (zt + zv) / 2.0
        if self.pre_norm_fused:
            fused = F.normalize(fused, p=2, dim=1)

        with torch.no_grad():
            text_relation = self.symmetric_softmax(zt)
            image_relation = self.symmetric_softmax(zv)
            feature_relation = (text_relation + image_relation) / 2.0

        alpha_term = self.alpha / (self.alpha + 1.0)
        beta_term = self.beta / (self.beta + 1.0)

        graph_filtered = fused
        current = fused
        for _ in range(self.num_layers):
            current = alpha_term * torch.sparse.mm(norm_adj, current)
            graph_filtered = graph_filtered + current

        feature_filter = torch.zeros_like(feature_relation)
        current_power = torch.eye(fused.size(1), dtype=fused.dtype, device=fused.device)
        for _ in range(self.num_layers):
            feature_filter = feature_filter + current_power
            current_power = current_power @ (beta_term * feature_relation)

        h = (graph_filtered @ feature_filter) / ((self.alpha + 1.0) * (self.beta + 1.0))
        h = F.normalize(h, p=2, dim=1)
        rec_t = self.text_decoder(h)
        rec_v = self.image_decoder(h)
        return zt, zv, h, rec_t, rec_v, gate

    def _forward_mlp(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        norm_adj: torch.Tensor,
        graph_scale: float | torch.Tensor = 1.0,
    ):
        zt = self.text_encoder(text)
        zv = self.image_encoder(image)
        fused, gate = self._fuse(zt, zv)
        current = fused
        accum = fused
        scale = float(graph_scale) if not torch.is_tensor(graph_scale) else graph_scale
        for _ in range(self.num_layers):
            current = torch.sparse.mm(norm_adj, current) * scale
            accum = accum + current
        fused = F.normalize(accum / (self.num_layers + 1), p=2, dim=1)
        rec_t = self.text_decoder(fused)
        rec_v = self.image_decoder(fused)
        return zt, zv, fused, rec_t, rec_v, gate

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        norm_adj: torch.Tensor,
        graph_scale: float | torch.Tensor = 1.0,
    ):
        if self.backbone == "mlp":
            return self._forward_mlp(text, image, norm_adj, graph_scale)
        return self._forward_dual_filter(text, image, norm_adj)
