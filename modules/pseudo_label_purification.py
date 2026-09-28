from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableClusterPrototypes(nn.Module):


    def __init__(self, num_clusters: int, hidden_dim: int, temperature: float = 0.2):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(torch.randn(self.num_clusters, self.hidden_dim) * 0.02)

    def initialize(self, centers: torch.Tensor) -> None:
        if centers.shape != self.prototypes.shape:
            raise ValueError(f"expected centers {tuple(self.prototypes.shape)}, got {tuple(centers.shape)}")
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(centers.to(self.prototypes.device), p=2, dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z = F.normalize(features, p=2, dim=1)
        p = F.normalize(self.prototypes, p=2, dim=1)
        logits = torch.matmul(z, p.t()) / self.temperature
        return F.softmax(logits, dim=1)


def top2_confidence_labeling(scores: torch.Tensor, threshold: float = 0.7):
    topv, topi = torch.topk(scores, k=min(2, scores.size(1)), dim=1)
    if topv.size(1) == 1:
        conf = torch.ones(scores.size(0), device=scores.device, dtype=scores.dtype)
    else:
        conf = topv[:, 0] / (topv[:, 0] + topv[:, 1]).clamp_min(1e-12)
    labels = topi[:, 0]
    high = conf >= threshold
    pseudo = torch.where(high, labels, torch.full_like(labels, -1))
    return pseudo, conf, high


def neighbor_consistency_refine(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    confidence: torch.Tensor,
    k: int = 10,
    min_ratio: float = 0.6,
    chunk_size: int = 4096,
):
    n = features.size(0)
    if n <= 1:
        weights = confidence.clone()
        mask = pseudo_labels >= 0
        return pseudo_labels, weights, mask
    kk = min(k, n - 1)
    z = F.normalize(features, p=2, dim=1)
    chunk = int(chunk_size or 0)
    if chunk <= 0:
        chunk = n
    topk_indices = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sim = torch.matmul(z[start:end], z.t())
        diag = torch.arange(end - start, device=features.device)
        sim[diag, start + diag] = -float("inf")
        topk_indices.append(torch.topk(sim, kk, dim=1).indices)
    idx = torch.cat(topk_indices, dim=0)
    neigh_labels = pseudo_labels[idx]
    own = pseudo_labels[:, None]
    valid_neigh = neigh_labels >= 0
    agree = (neigh_labels == own) & valid_neigh & (own >= 0)
    denom = valid_neigh.sum(dim=1).clamp_min(1)
    ratio = agree.sum(dim=1).float() / denom.float()
    high = (pseudo_labels >= 0) & (ratio >= min_ratio)
    refined = torch.where(high, pseudo_labels, torch.full_like(pseudo_labels, -1))
    weights = confidence * ratio.clamp_min(0.0)
    return refined, weights, high


def prototype_supervision_loss(
    proto_scores: torch.Tensor,
    refined_labels: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    mask = refined_labels >= 0
    if not mask.any():
        return proto_scores.new_tensor(0.0)
    logp = torch.log(proto_scores.clamp_min(1e-12))
    losses = F.nll_loss(logp[mask], refined_labels[mask], reduction="none")
    w = sample_weights[mask].to(losses.dtype)
    return (losses * w).sum() / w.sum().clamp_min(1e-12)
