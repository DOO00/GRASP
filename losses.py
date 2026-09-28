import math

import torch
import torch.nn.functional as F

try:
    import torch_cluster

    _random_walk = torch.ops.torch_cluster.random_walk
except Exception:
    torch_cluster = None
    _random_walk = None


def cross_modal_contrastive_loss(z1, z2, zf, margin: float = 0.001):
    def mms(sim):
        sim = sim - margin * torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        target = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, target) + F.cross_entropy(sim.t(), target)

    return mms(z1 @ z2.t()) + mms(z1 @ zf.t()) + mms(zf @ z2.t())


def batched_cross_modal_loss(z1, z2, zf, batch_size: int = -1, margin: float = 0.001):
    n = z1.size(0)
    if batch_size <= 0 or batch_size >= n:
        return cross_modal_contrastive_loss(z1, z2, zf, margin)
    perm = torch.randperm(n, device=z1.device)
    total = z1.new_tensor(0.0)
    count = 0
    for start in range(0, n - batch_size + 1, batch_size):
        idx = perm[start : start + batch_size]
        total = total + cross_modal_contrastive_loss(z1[idx], z2[idx], zf[idx], margin)
        count += 1
    return total / max(count, 1)


def entropy_balance_loss(scores: torch.Tensor):
    entropy = -(scores * scores.clamp_min(1e-12).log()).sum(dim=1).mean()
    mean_prob = scores.mean(dim=0)
    balance = (mean_prob * mean_prob.clamp_min(1e-12).log()).sum()
    return -entropy + balance


def normalized_loss(loss: torch.Tensor, eps: float = 1e-12):
    return loss / loss.detach().abs().clamp_min(eps)


@torch.no_grad()
def determine_similarity_threshold(adj, z1, z2, num_std: float = 1.0):
    adj = adj.coalesce()
    n = adj.size(0)
    m = max(1, adj._nnz())
    rows = torch.randint(0, n, (m,), device=z1.device)
    cols = torch.randint(0, n, (m,), device=z2.device)
    sims = (z1[rows] * z2[cols]).sum(dim=1)
    return (sims.mean() + float(num_std) * sims.std()).item()


@torch.no_grad()
def filter_adjacency_by_similarity(adj, z1, z2, threshold: float):
    adj = adj.coalesce()
    rows, cols = adj.indices()
    rows = rows.to(z1.device, non_blocking=True)
    cols = cols.to(z2.device, non_blocking=True)
    sims = (z1[rows] * z2[cols]).sum(dim=1)
    mask = sims >= threshold
    new_indices = torch.stack([rows[mask], cols[mask]], dim=0)
    new_values = torch.ones(new_indices.size(1), device=adj.device, dtype=adj.dtype)
    return torch.sparse_coo_tensor(new_indices, new_values, adj.shape, device=adj.device).coalesce()


def positive_random_walks(adj, walks_per_node: int, walk_length: int, context_size: int):
    device = adj.device
    nodes = torch.arange(adj.size(0), device=device).repeat(walks_per_node)
    if _random_walk is None:
        return _fallback_positive_contexts(adj, nodes, walk_length, context_size)
    csr = adj.coalesce().to_sparse_csr()
    rw = _random_walk(csr.crow_indices(), csr.col_indices(), nodes, walk_length, 1.0, 1.0)
    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    num_windows = 1 + walk_length + 1 - context_size
    return torch.cat([rw[:, j : j + context_size] for j in range(num_windows)], dim=0)


def negative_random_walks(adj, walks_per_node: int, walk_length: int, context_size: int, num_negative_samples: int = 1):
    device = adj.device
    nodes = torch.arange(adj.size(0), device=device).repeat(walks_per_node * num_negative_samples)
    rw = torch.randint(adj.size(0), (nodes.size(0), walk_length * num_negative_samples), device=device)
    rw = torch.cat([nodes.view(-1, 1), rw], dim=-1)
    num_windows = 1 + walk_length + 1 - context_size
    return torch.cat([rw[:, j : j + context_size] for j in range(num_windows)], dim=0)


def _fallback_positive_contexts(adj, nodes, walk_length: int, context_size: int):
    adj = adj.coalesce()
    rows, cols = adj.indices()
    deg = torch.bincount(rows, minlength=adj.size(0))
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(0)])


    walks = []
    for start in nodes.tolist():
        cur = start
        walk = [cur]
        for _ in range(walk_length):
            lo = int(ptr[cur].item())
            hi = int(ptr[cur + 1].item())
            if hi > lo:
                offset = int(torch.randint(0, hi - lo, (1,), device=cols.device).item())
                cur = int(cols[lo + offset].item())
            walk.append(cur)
        walks.append(walk)
    rw = torch.tensor(walks, device=nodes.device, dtype=torch.long)
    num_windows = 1 + walk_length + 1 - context_size
    return torch.cat([rw[:, j : j + context_size] for j in range(num_windows)], dim=0)


def graph_contrastive_loss(pos_rw, neg_rw, embedding, embedding_dim: int, mapping=None):
    if mapping is None:
        unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))
        mapping = torch.zeros(embedding.size(0), dtype=torch.long, device=embedding.device)
        mapping.scatter_(0, unique, torch.arange(unique.size(0), device=embedding.device))
    pos_rw = F.embedding(pos_rw.reshape(-1), mapping.reshape(-1, 1)).view(pos_rw.size())
    neg_rw = F.embedding(neg_rw.reshape(-1), mapping.reshape(-1, 1)).view(neg_rw.size())

    start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
    h_start = F.embedding(start, embedding).view(pos_rw.size(0), 1, embedding_dim)
    h_rest = F.embedding(rest.reshape(-1), embedding).view(pos_rw.size(0), -1, embedding_dim)
    pos_loss = torch.logsumexp((h_start * h_rest).sum(dim=-1), dim=-1)

    start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
    h_start = F.embedding(start, embedding).view(neg_rw.size(0), 1, embedding_dim)
    h_rest = F.embedding(rest.reshape(-1), embedding).view(neg_rw.size(0), -1, embedding_dim)
    neg_loss = torch.logsumexp((h_start * h_rest).sum(dim=-1), dim=-1)
    neg_loss = torch.logsumexp(torch.cat((neg_loss.view(-1, 1), pos_loss.view(-1, 1)), dim=-1), dim=-1)
    return -torch.mean(pos_loss - neg_loss)


def standardize(tensor, dim=0):
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True).clamp_min(1e-12)
    return (tensor - mean) / std


def cluster_contrastive_loss(h, centroids, labels, positive_threshold_percent: float = 0.2):
    n = h.size(0)
    k = centroids.size(0)
    if n == 0 or k == 0 or positive_threshold_percent <= 0:
        return h.new_tensor(0.0)

    total = h.new_tensor(0.0)
    used = 0
    for cluster_idx in range(k):
        idx = (labels == cluster_idx).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        samples = h[idx]
        own_centroid = centroids[cluster_idx]
        own_sim = samples @ own_centroid
        top_k = min(idx.numel(), max(1, math.ceil(idx.numel() * positive_threshold_percent)))
        pos = torch.exp(torch.topk(own_sim, k=top_k).values).sum()
        denom = torch.exp(samples @ centroids.t()).sum(dim=1).sum().clamp_min(1e-12)
        total = total - torch.log(pos.clamp_min(1e-12) / denom)
        used += 1
    if used == 0:
        return h.new_tensor(0.0)
    return total / n
