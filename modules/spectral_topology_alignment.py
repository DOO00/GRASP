from __future__ import annotations

import torch
import torch.nn.functional as F


def _maybe_subsample(features: torch.Tensor, max_nodes: int | None):
    if max_nodes is None or max_nodes <= 0 or features.size(0) <= max_nodes:
        return features, None
    idx = torch.randperm(features.size(0), device=features.device)[:max_nodes]
    return features[idx], idx


def build_knn_graph(
    features: torch.Tensor,
    k: int = 5,
    metric: str = "gaussian",
    symmetrize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:


    if features.dim() != 2:
        raise ValueError("features must have shape [N, d]")
    n = features.size(0)
    if n <= 1:
        return torch.zeros((n, n), device=features.device, dtype=features.dtype)

    k = min(max(int(k), 1), n - 1)
    x = F.normalize(features.float(), p=2, dim=1)
    metric = metric.lower()
    if metric == "cosine":
        sim = torch.matmul(x, x.t()).clamp_min(0)
    elif metric == "gaussian":
        dist = torch.cdist(x, x, p=2).pow(2)
        mask = ~torch.eye(n, dtype=torch.bool, device=features.device)
        sigma = torch.median(dist[mask]).detach().clamp_min(eps)
        sim = torch.exp(-dist / (2 * sigma))
    else:
        raise ValueError(f"unsupported metric: {metric}")

    sim = sim.masked_fill(torch.eye(n, dtype=torch.bool, device=features.device), -float("inf"))
    vals, idx = torch.topk(sim, k=k, dim=1)
    vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
    adj = torch.zeros((n, n), device=features.device, dtype=features.dtype)
    adj.scatter_(1, idx, vals.to(features.dtype))
    if symmetrize:
        adj = torch.maximum(adj, adj.t())
    return adj


def laplacian_lowfreq_spectrum(
    adj: torch.Tensor,
    r: int = 20,
    norm: str = "sym",
    eps: float = 1e-12,
) -> torch.Tensor:

    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError("adj must be a square [N, N] matrix")
    n = adj.size(0)
    if n == 0:
        return adj.new_zeros((0,))
    a = adj.float()
    a = torch.maximum(a, a.t())
    deg = a.sum(dim=1).clamp_min(eps)
    eye = torch.eye(n, device=a.device, dtype=a.dtype)
    norm = norm.lower()
    if norm == "sym":
        inv_sqrt = deg.rsqrt()
        lap = eye - inv_sqrt[:, None] * a * inv_sqrt[None, :]
    elif norm == "rw":
        lap = eye - a / deg[:, None]
        lap = 0.5 * (lap + lap.t())
    elif norm == "none":
        lap = torch.diag(deg) - a
    else:
        raise ValueError(f"unsupported norm: {norm}")
    eigvals = torch.linalg.eigvalsh(lap)
    return eigvals[: min(int(r), eigvals.numel())]


def global_topology_loss(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    fused_features: torch.Tensor,
    k: int = 5,
    r: int = 20,
    metric: str = "gaussian",
    max_nodes: int | None = 1024,
) -> torch.Tensor:

    ft, idx = _maybe_subsample(text_features, max_nodes)
    if idx is not None:
        fi = image_features[idx]
        ff = fused_features[idx]
    else:
        fi, ff = image_features, fused_features

    at = build_knn_graph(ft, k=k, metric=metric)
    ai = build_knn_graph(fi, k=k, metric=metric)
    af = build_knn_graph(ff, k=k, metric=metric)
    lt = laplacian_lowfreq_spectrum(at, r=r)
    li = laplacian_lowfreq_spectrum(ai, r=r)
    lf = laplacian_lowfreq_spectrum(af, r=r)
    rr = min(lt.numel(), li.numel(), lf.numel())
    if rr == 0:
        return fused_features.new_tensor(0.0)
    return F.mse_loss(lt[:rr], lf[:rr]) + F.mse_loss(li[:rr], lf[:rr])

