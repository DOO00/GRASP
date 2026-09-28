from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer

try:
    import dgl
except ImportError:
    dgl = None


DATASETS = ("Movies", "Toys", "GroceryS", "Grocery", "RedditS", "Reddit", "Photo", "Arts")


def dataset_paths(dataset: str, data_root: str | Path):
    root = Path(data_root)
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset}")
    if dataset == "RedditS":
        text = f"{dataset}_gemma_7b_64_mean.npy"
    elif dataset == "Reddit":
        text = f"{dataset}_gemma_7b_100_mean.npy"
    else:
        text = f"{dataset}_gemma_7b_256_mean.npy"
    base = root / dataset
    return {
        "text": base / "TextFeature" / text,
        "image": base / "ImageFeature" / f"{dataset}_openai_clip-vit-large-patch14.npy",
        "graph": base / f"{dataset}Graph.pt",
        "adj": base / f"{dataset}Adj.npz",
        "csv": base / f"{dataset}.csv",
    }


def load_features(text_file, image_file):
    text = np.load(text_file)
    image = np.load(image_file)
    text = np.where(np.isinf(text), np.nan, text)
    image = np.where(np.isinf(image), np.nan, image)
    imputer = SimpleImputer(strategy="mean")
    text = imputer.fit_transform(text)
    image = imputer.fit_transform(image)
    scaler = Normalizer(norm="l2")
    return scaler.fit_transform(text).astype("float32"), scaler.fit_transform(image).astype("float32")


def load_labels(csv_file):
    return pd.read_csv(csv_file)["label"].to_numpy(dtype=np.int64)


def _load_cached_adj(path: Path):
    data = np.load(path)
    indices = np.vstack((data["rows"], data["cols"]))
    return torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(data["values"], dtype=torch.float32),
        tuple(int(x) for x in data["shape"]),
    ).coalesce()


def _load_dgl_adj(graph_file: Path):
    if dgl is None:
        raise RuntimeError(f"{graph_file} needs DGL or a converted Adj.npz cache")
    graph_data = dgl.load_graphs(str(graph_file))[0]
    adj = graph_data[0].adjacency_matrix(scipy_fmt="csr")
    adj = adj.maximum(adj.transpose().tocsr())
    adj.setdiag(0)
    adj.eliminate_zeros()
    indices = np.vstack(adj.nonzero())
    return torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(adj.data, dtype=torch.float32),
        adj.shape,
    ).coalesce()


def normalize_adj(adj: torch.Tensor, eps: float = 1e-12):
    adj = adj.coalesce()
    deg = torch.sparse.mm(adj, torch.ones(adj.size(0), 1)).view(-1)
    inv = deg.clamp_min(eps).rsqrt()
    row, col = adj.indices()
    values = adj.values() * inv[row] * inv[col]
    return torch.sparse_coo_tensor(adj.indices(), values, adj.shape).coalesce()


def load_dataset(dataset: str, data_root: str | Path, device: str | torch.device = "cpu"):
    paths = dataset_paths(dataset, data_root)
    text, image = load_features(paths["text"], paths["image"])
    labels = load_labels(paths["csv"])
    adj = _load_cached_adj(paths["adj"]) if paths["adj"].exists() else _load_dgl_adj(paths["graph"])
    norm_adj = normalize_adj(adj)
    device = torch.device(device)
    return {
        "dataset": dataset,
        "text": torch.from_numpy(text).to(device),
        "image": torch.from_numpy(image).to(device),
        "adj": adj.to(device),
        "norm_adj": norm_adj.to(device),
        "labels": torch.from_numpy(labels).to(device),
        "labels_np": labels,
        "num_clusters": int(labels.max() + 1),
        "paths": paths,
    }

