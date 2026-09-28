from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .spectral_topology_alignment import build_knn_graph


@dataclass
class AugConfig:
    feature_mask: float = 0.1
    modality_dropout: float = 0.1
    edge_dropout: float = 0.0
    graph_scale_min: float = 0.9
    graph_scale_max: float = 1.1


def _drop_sparse_edges(adj: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0 or not adj.is_sparse:
        return adj
    adj = adj.coalesce()
    values = adj.values()
    keep = torch.rand(values.numel(), device=values.device) > drop_prob
    if keep.sum() == 0:
        return adj
    return torch.sparse_coo_tensor(
        adj.indices()[:, keep],
        values[keep] / (1.0 - drop_prob),
        adj.shape,
        device=adj.device,
    ).coalesce()


def _mask_features(x: torch.Tensor, mask_prob: float) -> torch.Tensor:
    if mask_prob <= 0:
        return x
    mask = torch.rand_like(x) > mask_prob
    return x * mask.to(x.dtype)


def sample_multiview_embeddings(model, data: dict, num_views: int = 4, aug_cfg=None) -> torch.Tensor:

    cfg = aug_cfg if isinstance(aug_cfg, AugConfig) else AugConfig(**(aug_cfg or {}))
    views = []
    text = data["text"]
    image = data["image"]
    adj = data["norm_adj"]
    for _ in range(num_views):
        t = _mask_features(text, cfg.feature_mask)
        v = _mask_features(image, cfg.feature_mask)
        if cfg.modality_dropout > 0:
            r = torch.rand((), device=text.device)
            if r < cfg.modality_dropout / 2:
                t = torch.zeros_like(t)
            elif r < cfg.modality_dropout:
                v = torch.zeros_like(v)
        a = _drop_sparse_edges(adj, cfg.edge_dropout)
        scale = torch.empty((), device=text.device).uniform_(cfg.graph_scale_min, cfg.graph_scale_max)
        _, _, fused, _, _, _ = model(t, v, a, graph_scale=scale)
        views.append(fused)
    return torch.stack(views, dim=1)


def pairwise_mean_discrepancy(
    view_embeddings: torch.Tensor,
    batch_index: torch.Tensor | None = None,
    chunk_size: int = 512,
) -> torch.Tensor:


    x = view_embeddings if batch_index is None else view_embeddings[batch_index]
    means = x.mean(dim=1)
    n = means.size(0)
    out = torch.empty((n, n), device=x.device, dtype=x.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = torch.cdist(means[start:end], means, p=2)
    return out


def select_local_positive_pairs(
    pseudo_labels: torch.Tensor,
    proto_labels: torch.Tensor,
    distance_matrix: torch.Tensor,
    neighbor_graph: torch.Tensor,
    retain_ratio: float = 0.3,
    max_pairs: int = 20000,
) -> tuple[torch.Tensor, torch.Tensor]:

    n = pseudo_labels.numel()
    if n == 0:
        return pseudo_labels.new_zeros((2, 0)), distance_matrix.new_zeros((0,))
    valid = (pseudo_labels >= 0) & (proto_labels >= 0)
    same_pseudo = pseudo_labels[:, None] == pseudo_labels[None, :]
    same_proto = proto_labels[:, None] == proto_labels[None, :]
    neighbor = neighbor_graph > 0
    not_self = ~torch.eye(n, dtype=torch.bool, device=pseudo_labels.device)
    candidates = valid[:, None] & valid[None, :] & same_pseudo & same_proto & neighbor & not_self
    if not candidates.any():
        return pseudo_labels.new_zeros((2, 0)), distance_matrix.new_zeros((0,))
    distances = distance_matrix[candidates]
    thresh = torch.quantile(distances, float(retain_ratio))
    mask = candidates & (distance_matrix <= thresh)
    pairs = mask.nonzero(as_tuple=False).t()
    if pairs.numel() == 0:
        return pseudo_labels.new_zeros((2, 0)), distance_matrix.new_zeros((0,))
    if pairs.size(1) > max_pairs:
        perm = torch.randperm(pairs.size(1), device=pairs.device)[:max_pairs]
        pairs = pairs[:, perm]
    weights = torch.exp(-distance_matrix[pairs[0], pairs[1]]).detach()
    return pairs, weights


def weighted_infonce_loss(
    z: torch.Tensor,
    pos_pairs: torch.Tensor,
    pair_weights: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:

    if pos_pairs.numel() == 0:
        return z.new_tensor(0.0)
    h = F.normalize(z, p=2, dim=1)
    logits = torch.matmul(h, h.t()) / temperature
    logits = logits.masked_fill(torch.eye(h.size(0), device=h.device, dtype=torch.bool), -1e9)
    log_prob = F.log_softmax(logits, dim=1)
    losses = -log_prob[pos_pairs[0], pos_pairs[1]]
    return (losses * pair_weights.to(losses.dtype)).sum() / pair_weights.sum().clamp_min(1e-12)


def build_neighbor_graph(features: torch.Tensor, k: int = 10, metric: str = "cosine") -> torch.Tensor:
    return build_knn_graph(features, k=k, metric=metric)
