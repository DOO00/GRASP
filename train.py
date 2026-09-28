from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from data import DATASETS, load_dataset
from losses import (
    batched_cross_modal_loss,
    cluster_contrastive_loss,
    determine_similarity_threshold,
    entropy_balance_loss,
    filter_adjacency_by_similarity,
    graph_contrastive_loss,
    negative_random_walks,
    normalized_loss,
    positive_random_walks,
    standardize,
)
from metrics import calculate_clustering_metrics
from model import GRASPModel
from modules.ema_teacher import EMATeacher
from modules.spectral_topology_alignment import global_topology_loss
from modules.local_relation_calibration import (
    pairwise_mean_discrepancy,
    build_neighbor_graph,
    sample_multiview_embeddings,
    select_local_positive_pairs,
    weighted_infonce_loss,
)
from modules.pseudo_label_purification import (
    LearnableClusterPrototypes,
    neighbor_consistency_refine,
    prototype_supervision_loss,
    top2_confidence_labeling,
)


METRICS = ["ACC", "NMI", "F1", "ARI", "CS"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_percent(metrics):
    return [float(x) * 100.0 for x in metrics]


def selection_score(metrics_pct, metric_name: str) -> float:
    values = dict(zip(METRICS, [float(x) for x in metrics_pct]))
    name = metric_name.upper()
    if name not in values:
        raise ValueError(f"unknown selection metric {metric_name!r}; choices={METRICS}")
    return values[name]


@torch.no_grad()
def kmeans_metrics(features, labels_np, n_clusters, n_init=10, seed=0):
    x = features.detach().cpu().numpy()
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    pred = km.fit_predict(x)
    metrics = calculate_clustering_metrics(labels_np, pred)
    centers = torch.from_numpy(km.cluster_centers_).float().to(features.device)
    return metrics, pred, centers


def select_eval_features(zt: torch.Tensor, zv: torch.Tensor, h: torch.Tensor, name: str) -> torch.Tensor:
    avg = torch.nn.functional.normalize((zt + zv) / 2.0, p=2, dim=1)
    h_std = torch.nn.functional.normalize(
        (h - h.mean(dim=0, keepdim=True)) / h.std(dim=0, keepdim=True).clamp_min(1e-12),
        p=2,
        dim=1,
    )
    variants = {
        "h": h,
        "h_std": h_std,
        "avg": avg,
        "h_avg_07": torch.nn.functional.normalize(0.7 * h + 0.3 * avg, p=2, dim=1),
        "h_avg_05": torch.nn.functional.normalize(0.5 * h + 0.5 * avg, p=2, dim=1),
        "h_avg_03": torch.nn.functional.normalize(0.3 * h + 0.7 * avg, p=2, dim=1),
        "concat_h_avg": torch.cat([h, avg], dim=1),
        "concat_hstd_avg": torch.cat([h_std, avg], dim=1),
        "concat_zt_zv": torch.cat([zt, zv], dim=1),
        "concat_h_zt_zv": torch.cat([h, zt, zv], dim=1),
    }
    if name not in variants:
        raise ValueError(f"unknown eval feature {name}; choices={sorted(variants)}")
    return variants[name]


class KMeansRunner:
    def __init__(self, n_clusters, n_init=10, mode="sklearn", batch_size=65536, iters=15, sklearn_every=10):
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.mode = mode
        self.batch_size = int(batch_size)
        self.iters = int(iters)
        self.sklearn_every = int(sklearn_every)
        self.centroids = None

    @torch.no_grad()
    def gpu(self, x):
        x = x.detach()
        n, d = x.shape
        k = self.n_clusters
        if self.centroids is None or self.centroids.shape != (k, d):
            self.centroids = x[torch.randperm(n, device=x.device)[:k]].clone()
        centroids = self.centroids
        labels = None
        for _ in range(self.iters):
            chunks = []
            for start in range(0, n, self.batch_size):
                xb = x[start : start + self.batch_size]
                dist = torch.cdist(xb.float(), centroids.float(), p=2)
                chunks.append(torch.argmin(dist, dim=1))
            labels = torch.cat(chunks, dim=0)
            counts = torch.bincount(labels, minlength=k).to(x.dtype).clamp_min(1).unsqueeze(1)
            new_centroids = torch.zeros_like(centroids)
            new_centroids.scatter_add_(0, labels[:, None].expand(-1, d), x)
            new_centroids = new_centroids / counts
            empty = counts.squeeze(1) <= 1
            if empty.any():
                new_centroids[empty] = centroids[empty]
            shift = torch.max(torch.norm(new_centroids - centroids, dim=1))
            centroids = new_centroids
            if shift.item() < 1e-5:
                break
        self.centroids = centroids.detach()
        return labels, centroids, "gpu"

    @torch.no_grad()
    def sklearn(self, x, seed=0):
        km = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=seed)
        labels_np = km.fit_predict(x.detach().cpu().numpy())
        centers = torch.from_numpy(km.cluster_centers_).float().to(x.device)
        self.centroids = centers.detach()
        labels = torch.from_numpy(labels_np).to(x.device, dtype=torch.long)
        return labels, centers, "sklearn"

    def run(self, x, epoch, seed=0, force_sklearn=False):
        if force_sklearn or self.mode == "sklearn":
            return self.sklearn(x, seed=seed)
        if self.mode in ("hybrid", "strict_fast"):
            if self.centroids is None or epoch == 1 or epoch % self.sklearn_every == 0:
                return self.sklearn(x, seed=seed)
            return self.gpu(x)
        return self.gpu(x)


def write_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRASP for multimodal attributed graph clustering")
    parser.add_argument("--dataset", default="Movies", choices=DATASETS)
    parser.add_argument("--data-root", default="./MAG")
    parser.add_argument("--output-root", default="./results")
    parser.add_argument("--method", default="GRASP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--backbone", default="dual_filter", choices=["dual_filter", "mlp"])
    parser.add_argument("--fusion", default="mean", choices=["mean", "gate"])
    parser.add_argument("--pre-norm-fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--selection-metric", default="NMI", choices=METRICS)
    parser.add_argument(
        "--eval-feature",
        default="h",
        choices=[
            "h",
            "h_std",
            "avg",
            "h_avg_07",
            "h_avg_05",
            "h_avg_03",
            "concat_h_avg",
            "concat_hstd_avg",
            "concat_zt_zv",
            "concat_h_zt_zv",
        ],
    )
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--kmeans-mode", default="hybrid", choices=["sklearn", "gpu", "hybrid", "strict_fast"])
    parser.add_argument("--kmeans-iters", type=int, default=15)
    parser.add_argument("--kmeans-batch", type=int, default=65536)
    parser.add_argument("--sklearn-every", type=int, default=10)
    parser.add_argument("--early-stop-min-epoch", type=int, default=0)
    parser.add_argument("--early-stop-min-score", type=float, default=None)
    parser.add_argument("--init-checkpoint", default="")

    parser.add_argument("--spec-start", type=int, default=20)
    parser.add_argument("--local-start", type=int, default=60)
    parser.add_argument("--proto-start", type=int, default=80)
    parser.add_argument("--refresh-interval", type=int, default=10)
    parser.add_argument("--lambda-cross", type=float, default=1.0)
    parser.add_argument("--lambda-graph", type=float, default=1.0)
    parser.add_argument("--lambda-cluster", type=float, default=1.0)
    parser.add_argument("--lambda-spec", type=float, default=0.01)
    parser.add_argument("--lambda-local", type=float, default=0.01)
    parser.add_argument("--lambda-proto", type=float, default=0.05)
    parser.add_argument("--lambda-recon", type=float, default=0.0)
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--normalize-losses", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--disable-spec", action="store_true")
    parser.add_argument("--disable-local", action="store_true")
    parser.add_argument("--disable-proto", action="store_true")
    parser.add_argument("--disable-graph-loss", action="store_true")
    parser.add_argument("--disable-cluster-loss", action="store_true")
    parser.add_argument("--aas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--walks-per-node", type=int, default=10)
    parser.add_argument("--walk-length", type=int, default=5)
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--spec-k", type=int, default=5)
    parser.add_argument("--spec-r", type=int, default=20)
    parser.add_argument("--spec-max-nodes", type=int, default=1024)
    parser.add_argument("--local-max-nodes", type=int, default=2048)
    parser.add_argument("--local-views", type=int, default=3)
    parser.add_argument("--local-k", type=int, default=10)
    parser.add_argument("--local-retain-ratio", type=float, default=0.3)
    parser.add_argument("--top2-threshold", type=float, default=0.7)
    parser.add_argument("--neighbor-ratio", type=float, default=0.6)
    parser.add_argument("--neighbor-chunk-size", type=int, default=4096)
    parser.add_argument("--proto-temperature", type=float, default=0.2)
    parser.add_argument("--teacher-decay", type=float, default=0.99)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = Path(args.output_root) / args.dataset / args.method / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", vars(args))

    data = load_dataset(args.dataset, args.data_root, device=device)
    labels_np = data["labels_np"]
    n_clusters = data["num_clusters"]

    model = GRASPModel(
        text_dim=data["text"].size(1),
        image_dim=data["image"].size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        alpha=args.alpha,
        beta=args.beta,
        backbone=args.backbone,
        fusion=args.fusion,
        pre_norm_fused=args.pre_norm_fused,
    ).to(device)
    prototypes = LearnableClusterPrototypes(n_clusters, args.hidden_dim, args.proto_temperature).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if "prototypes" in ckpt:
            prototypes.load_state_dict(ckpt["prototypes"], strict=False)
        print(f"Loaded init checkpoint: {args.init_checkpoint}", flush=True)
    teacher = EMATeacher(model, decay=args.teacher_decay)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(prototypes.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "method",
                "seed",
                "epoch",
                *METRICS,
                "loss",
                "L_base",
                "L_cross",
                "L_graph",
                "L_cluster",
                "L_spec",
                "L_local",
                "L_proto",
                "L_recon",
                "L_reg",
                "high_conf",
                "pos_pairs",
                "epoch_time",
                "gpu_memory_mb",
            ]
        )

    best = {"NMI": -1.0, "score": -1.0e18, "epoch": 0, "metrics": [0, 0, 0, 0, 0]}
    stop_reason = ""
    proto_initialized = False
    pseudo_labels = torch.full((data["text"].size(0),), -1, dtype=torch.long, device=device)
    proto_labels = torch.full_like(pseudo_labels, -1)
    sample_weights = torch.zeros_like(data["text"][:, 0])
    pos_pairs = torch.zeros((2, 0), dtype=torch.long, device=device)
    pair_weights = torch.zeros((0,), dtype=torch.float32, device=device)
    local_index = None
    mapping = torch.zeros(data["text"].size(0), dtype=torch.int64, device=device)
    kmeans_runner = KMeansRunner(
        n_clusters,
        n_init=args.kmeans_n_init,
        mode=args.kmeans_mode,
        batch_size=args.kmeans_batch,
        iters=args.kmeans_iters,
        sklearn_every=args.sklearn_every,
    )
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        zt, zv, h, rec_t, rec_v, gate = model(data["text"], data["image"], data["norm_adj"])

        l_cross = batched_cross_modal_loss(zt, zv, h, batch_size=args.batch_size)
        l_graph = h.new_tensor(0.0)
        if (not args.disable_graph_loss) and args.lambda_graph > 0:
            adj_for_walk = data["adj"]
            if args.aas:
                threshold = determine_similarity_threshold(data["adj"], zt.detach(), zv.detach(), args.std)
                adj_for_walk = filter_adjacency_by_similarity(data["adj"], zt.detach(), zv.detach(), threshold)
                if adj_for_walk._nnz() == 0:
                    adj_for_walk = data["adj"]
            pos_rw = positive_random_walks(adj_for_walk, args.walks_per_node, args.walk_length, args.context_size)
            neg_rw = negative_random_walks(data["adj"], args.walks_per_node, args.walk_length, args.context_size)
            unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))
            mapping.zero_()
            mapping.scatter_(0, unique, torch.arange(unique.size(0), device=device))
            l_graph = graph_contrastive_loss(pos_rw, neg_rw, h, args.hidden_dim, mapping)

        train_labels = None
        train_centers = None
        if ((not args.disable_cluster_loss) and args.lambda_cluster > 0 and args.theta != -1) or (
            (not args.disable_proto) and epoch >= args.proto_start
        ):
            train_labels, train_centers, _ = kmeans_runner.run(h, epoch, seed=args.seed + epoch)

        l_cluster = h.new_tensor(0.0)
        if (
            train_labels is not None
            and train_centers is not None
            and (not args.disable_cluster_loss)
            and args.lambda_cluster > 0
            and args.theta != -1
        ):
            h_cluster = F.normalize(standardize(h, dim=0), p=2, dim=1)
            centers_cluster = F.normalize(standardize(train_centers.detach(), dim=0), p=2, dim=1)
            l_cluster = cluster_contrastive_loss(h_cluster, centers_cluster, train_labels, args.theta)

        if args.normalize_losses:
            l_base = (
                args.lambda_cross * normalized_loss(l_cross)
                + args.lambda_graph * normalized_loss(l_graph)
                + args.lambda_cluster * normalized_loss(l_cluster)
            )
        else:
            l_base = args.lambda_cross * l_cross + args.lambda_graph * l_graph + args.lambda_cluster * l_cluster
        l_recon = F.mse_loss(rec_t, data["text"]) + F.mse_loss(rec_v, data["image"])
        l_spec = h.new_tensor(0.0)
        l_local = h.new_tensor(0.0)
        l_proto = h.new_tensor(0.0)
        l_reg = h.new_tensor(0.0)
        high_conf_count = 0
        pos_pair_count = int(pos_pairs.size(1))

        if (not args.disable_spec) and epoch >= args.spec_start and epoch % args.refresh_interval == 0:
            l_spec = global_topology_loss(
                zt,
                zv,
                h,
                k=args.spec_k,
                r=args.spec_r,
                max_nodes=args.spec_max_nodes,
            )

        if (not args.disable_proto) and epoch >= args.proto_start:
            if (not proto_initialized) or epoch % args.refresh_interval == 0:
                if train_centers is None:
                    _, centers, _ = kmeans_runner.run(h, epoch, seed=args.seed + epoch)
                else:
                    centers = train_centers
                prototypes.initialize(centers)
                proto_initialized = True

            scores = prototypes(h)
            stu_labels, confidence, stu_high = top2_confidence_labeling(scores, args.top2_threshold)
            with torch.no_grad():
                _, _, teacher_h, _, _, _ = teacher.encode(data["text"], data["image"], data["norm_adj"])
                tea_scores = prototypes(teacher_h)
                tea_labels, _, tea_high = top2_confidence_labeling(tea_scores, args.top2_threshold)
            agree = (stu_labels == tea_labels) & stu_high & tea_high
            proto_labels = torch.where(agree, stu_labels, torch.full_like(stu_labels, -1))
            pseudo_labels, sample_weights, high_mask = neighbor_consistency_refine(
                h.detach(),
                proto_labels,
                confidence.detach(),
                k=args.local_k,
                min_ratio=args.neighbor_ratio,
                chunk_size=args.neighbor_chunk_size,
            )
            high_conf_count = int(high_mask.sum().item())
            l_proto = prototype_supervision_loss(scores, pseudo_labels, sample_weights)
            l_reg = entropy_balance_loss(scores)

        if (
            (not args.disable_local)
            and proto_initialized
            and epoch >= args.local_start
            and epoch % args.refresh_interval == 0
        ):
            with torch.no_grad():
                max_nodes = min(args.local_max_nodes, h.size(0))
                local_index = torch.randperm(h.size(0), device=device)[:max_nodes]
                views = sample_multiview_embeddings(model, data, num_views=args.local_views)
                local_distances = pairwise_mean_discrepancy(views, local_index)
                neighbor_graph = build_neighbor_graph(h.detach()[local_index], k=args.local_k)
                pos_pairs, pair_weights = select_local_positive_pairs(
                    pseudo_labels[local_index],
                    proto_labels[local_index],
                    local_distances,
                    neighbor_graph,
                    retain_ratio=args.local_retain_ratio,
                )
                pos_pair_count = int(pos_pairs.size(1))

        if (not args.disable_local) and local_index is not None and pos_pairs.numel() > 0 and epoch >= args.local_start:
            l_local = weighted_infonce_loss(h[local_index], pos_pairs, pair_weights)

        loss = (
            l_base
            + args.lambda_recon * l_recon
            + args.lambda_spec * l_spec
            + args.lambda_local * l_local
            + args.lambda_proto * l_proto
            + args.lambda_reg * l_reg
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        teacher.update(model)

        epoch_time = time.time() - epoch_start
        should_eval = epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs
        if should_eval:
            model.eval()
            with torch.no_grad():
                zt_eval, zv_eval, h_eval, _, _, _ = model(data["text"], data["image"], data["norm_adj"])
                eval_features = select_eval_features(zt_eval, zv_eval, h_eval, args.eval_feature)
            metrics_raw, pred_np, centers = kmeans_metrics(
                eval_features,
                labels_np,
                n_clusters,
                n_init=args.kmeans_n_init,
                seed=args.seed + epoch,
            )
            metrics_pct = as_percent(metrics_raw)
            score = selection_score(metrics_pct, args.selection_metric)
            if score > best["score"]:
                best = {"NMI": metrics_pct[1], "score": score, "epoch": epoch, "metrics": metrics_pct}
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "prototypes": prototypes.state_dict(),
                        "metrics": metrics_pct,
                        "selection_score": score,
                        "args": vars(args),
                        "eval_feature": args.eval_feature,
                    },
                    run_dir / "best.pt",
                )
                np.save(run_dir / "pred.npy", pred_np.astype(np.int64, copy=False))
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
            row = [
                args.dataset,
                args.method,
                args.seed,
                epoch,
                *[f"{v:.4f}" for v in metrics_pct],
                f"{float(loss.detach()):.6f}",
                f"{float(l_base.detach()):.6f}",
                f"{float(l_cross.detach()):.6f}",
                f"{float(l_graph.detach()):.6f}",
                f"{float(l_cluster.detach()):.6f}",
                f"{float(l_spec.detach()):.6f}",
                f"{float(l_local.detach()):.6f}",
                f"{float(l_proto.detach()):.6f}",
                f"{float(l_recon.detach()):.6f}",
                f"{float(l_reg.detach()):.6f}",
                high_conf_count,
                pos_pair_count,
                f"{epoch_time:.3f}",
                f"{gpu_mem:.1f}",
            ]
            with metrics_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            progress = {
                "dataset": args.dataset,
                "method": args.method,
                "seed": args.seed,
                "epoch": epoch,
                "num_epochs": args.epochs,
                "metrics": dict(zip(METRICS, metrics_pct)),
                "best_epoch": best["epoch"],
                "best_metrics": dict(zip(METRICS, best["metrics"])),
                "selection_score": best["score"],
                "eval_feature": args.eval_feature,
                "loss": float(loss.detach()),
                "high_conf": high_conf_count,
                "pos_pairs": pos_pair_count,
                "elapsed_sec": round(time.time() - start_time, 3),
            }
            write_json(run_dir / "progress.json", progress)
            print(
                f"Progress {args.dataset} epoch {epoch}/{args.epochs} "
                f"{args.selection_metric}={score:.2f} best={best['score']:.2f} "
                f"loss={float(loss.detach()):.4f} pairs={pos_pair_count}",
                flush=True,
            )
            if (
                args.early_stop_min_score is not None
                and epoch >= args.early_stop_min_epoch
                and best["score"] < args.early_stop_min_score
            ):
                stop_reason = (
                    f"early_stop: best_score={best['score']:.4f} < "
                    f"min_score={args.early_stop_min_score:.4f} at epoch={epoch}"
                )
                print(stop_reason, flush=True)
                break

    write_json(
        run_dir / "summary.json",
        {
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "best_epoch": best["epoch"],
            "best_metrics": dict(zip(METRICS, best["metrics"])),
            "selection_metric": args.selection_metric,
            "selection_score": best["score"],
            "eval_feature": args.eval_feature,
            "prediction_file": "pred.npy",
            "stop_reason": stop_reason,
            "total_time_sec": round(time.time() - start_time, 3),
        },
    )
    print(f"Best metrics: {tuple(v / 100.0 for v in best['metrics'])}", flush=True)


if __name__ == "__main__":
    main()
