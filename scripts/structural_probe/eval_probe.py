#!/usr/bin/env python3
"""
Train and evaluate a structural probe for hierarchical tree data (entity-level)
==============================================================================

*Added: JSONL tree‑visualization export*
--------------------------------------
This script now offers **two outputs**:

1. **metrics** — distance & tree metrics (UUAS / NSpearman / Root %) written to
   ``probe_results/metrics_layer<LAYER>_repeat<...>.json``.
2. **visualization JSONL** — one line per tree with predicted vs. gold
   distance matrices and metadata (same schema as provided by the user).  Set
   ``--vis_path`` to generate it, e.g. ``--vis_path probe_results/vis.jsonl``.

Usage example
-------------
::

    python evalprobe.py \
        --embedding_dir embeddings/ \
        --layer 12 \
        --probe_rank 128 \
        --probe_path my_probe.pt \
        --vis_path probe_results/vis_layer12.jsonl

"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm

# ================================================================
# Probe definition
# ================================================================

class TwoWordPSDProbe(nn.Module):
    """Squared L2 distance after low‑rank projection"""

    def __init__(self, model_dim: int, probe_rank: int, device, dtype=torch.float32):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(model_dim, probe_rank, dtype=dtype))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, D)
        x = x.to(self.proj.dtype)
        x = torch.matmul(x, self.proj)                   # (B, N, R)
        x1 = x.unsqueeze(2)                              # (B, N, 1, R)
        dist = (x1 - x1.transpose(1, 2)).pow(2).sum(-1)  # (B, N, N)
        return dist


class L1DistanceLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, gold):
        valid = torch.isfinite(gold)
        if not valid.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        diff = torch.abs(pred[valid] - gold[valid])
        if self.reduction == "sum":
            return diff.sum()
        if self.reduction == "none":
            return diff
        return diff.mean()


# ================================================================
# CLI
# ================================================================

def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--embedding_dir", required=True)
    pa.add_argument("--layer", type=int, required=True)
    pa.add_argument("--probe_rank", type=int, default=128)
    pa.add_argument("--probe_path", required=True, help="pre‑trained probe (.pt)")
    pa.add_argument("--train_ratio", type=float, default=0.8)
    pa.add_argument("--device", default="auto")
    pa.add_argument("--save_dir", default="probe_results")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--icl_repeat_num", type=int, default=None,
                    help="ICL repeat number if embeddings are in repeat_* dirs")
    pa.add_argument("--vis_path", default=None,
                    help="Path to write visualization JSONL (optional)")
    return pa.parse_args()


# ================================================================
# Embedding helpers
# ================================================================

def _collect_embedding_paths(root: str, icl_repeat_num: int = None) -> List[str]:
    if icl_repeat_num is None:
        return [os.path.join(root, f) for f in os.listdir(root) if f.endswith("_embeddings.pt")]
    repeat_tag = f"repeat_{icl_repeat_num}"
    paths = []
    for cur, _, files in os.walk(root):
        if os.path.basename(cur) == repeat_tag:
            paths.extend(os.path.join(cur, f) for f in files if f.endswith("_embeddings.pt"))
    paths.sort()
    return paths


def load_embeddings(emb_dir: str, layer: int, repeat: int = None) -> List[Dict]:
    paths = _collect_embedding_paths(emb_dir, repeat)
    if not paths:
        print(f"ERROR: no *_embeddings.pt found in {emb_dir} (repeat={repeat})", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Loading embeddings from {emb_dir} (repeat={repeat})")
    data = []
    print(f"Found {len(paths)} embedding files")
    for p in paths:
        blob = torch.load(p, map_location="cpu")
        emb = blob["embeddings"]  # (N, L, D)
        if layer >= emb.shape[1]:
            raise ValueError(f"Layer {layer} out of range for {p}")
        md = blob["metadata"]
        data.append({
            "embeddings": emb[:, layer, :],
            "distances": torch.tensor(md["distances"], dtype=torch.float32),
            "nodes": md["nodes"],
            "node_titles": md["node_titles"],
            "tree_id": md["tree_id"],
            "tree_structure": md.get("tree_structure"),
            "hidden_dim": emb.shape[-1],
        })
    return data


# ================================================================
# Split helper
# ================================================================

def create_splits(data, ratio, seed):
    np.random.seed(seed)
    idx = np.random.permutation(len(data))
    n_train = max(1, min(len(data) - 1, int(len(data) * ratio)))
    return [data[i] for i in idx[:n_train]], [data[i] for i in idx[n_train:]]


# ================================================================
# Tree‑metric helpers
# ================================================================

def _mst_edges(dist: np.ndarray) -> List[Tuple[int, int]]:
    n = len(dist)
    selected = [0]
    edges: List[Tuple[int, int]] = []
    while len(selected) < n:
        best = None
        best_val = np.inf
        for i in selected:
            for j in range(n):
                if j in selected:
                    continue
                d = dist[i, j]
                if not np.isfinite(d):
                    continue
                if d < best_val:
                    best_val = d
                    best = (i, j)
        if best is None:  # disconnected (inf)
            remain = [j for j in range(n) if j not in selected]
            best = (selected[0], remain[0])
        edges.append(best)
        selected.append(best[1])
    return edges


def _evaluate_depth(gold: np.ndarray, pred: np.ndarray) -> Tuple[float, bool]:
    finite_mask = np.isfinite(gold)
    gold_root = int(np.argmin(np.where(finite_mask, gold, 0.0).sum(1)))
    pred_root = int(np.argmin(np.where(np.isfinite(pred), pred, 0.0).sum(1)))
    rho, _ = spearmanr(gold[gold_root], pred[pred_root])
    return (rho if np.isfinite(rho) else 0.0), gold_root == pred_root


# ================================================================
# Metrics
# ================================================================

def distance_metrics(probe, dataset, device):
    preds, golds = [], []
    with torch.no_grad():
        for td in dataset:
            d_pred = probe(td["embeddings"].unsqueeze(0).to(device)).squeeze(0)
            gold = td["distances"]
            n = gold.size(0)
            for i in range(n):
                for j in range(i + 1, n):
                    if torch.isfinite(gold[i, j]):
                        preds.append(d_pred[i, j].item())
                        golds.append(gold[i, j].item())
    rho, _ = spearmanr(preds, golds)
    preds, golds = np.array(preds), np.array(golds)
    return {
        "spearman_distance": float(rho),
        "mae": float(np.abs(preds - golds).mean()),
        "rmse": float(np.sqrt(((preds - golds) ** 2).mean())),
    }


def tree_metrics(probe, dataset, device):
    uuas_correct = uuas_total = 0
    depth_rhos = []
    root_hits = 0
    with torch.no_grad():
        for td in dataset:
            pred = probe(td["embeddings"].unsqueeze(0).to(device)).squeeze(0).cpu().float().numpy()
            gold = td["distances"].cpu().float().numpy()
            # UUAS
            pred_edges = {frozenset(e) for e in _mst_edges(0.5 * (pred + pred.T))}
            gold_edges = {frozenset(e) for e in _mst_edges(0.5 * (gold + gold.T))}
            uuas_correct += len(pred_edges & gold_edges)
            uuas_total += len(gold_edges)
            # depth + root
            rho, root_ok = _evaluate_depth(gold, pred)
            depth_rhos.append(rho)
            root_hits += int(root_ok)
    return {
        "uuas": uuas_correct / uuas_total if uuas_total else 0.0,
        "nspearman": float(np.mean(depth_rhos)) if depth_rhos else 0.0,
        "root_acc": root_hits / len(dataset) if dataset else 0.0,
    }


# ================================================================
# Visualization exporter
# ================================================================

def save_tree_visualization_data(probe, tree_data_list, device, save_path):
    """Write 1‑tree‑per‑line JSONL for downstream interactive viz."""
    probe.eval()
    with open(save_path, "w") as fout:
        for td in tqdm(tree_data_list, desc="[viz‑export]"):
            with torch.no_grad():
                pred = probe(td["embeddings"].unsqueeze(0).to(device)).squeeze(0).cpu().float().numpy()
            gold = td["distances"].cpu().float().numpy()
            node_to_idx = {qid: i for i, qid in enumerate(td["nodes"])}
            # edges (supports both schemas described)
            gold_edges = []
            ts = td.get("tree_structure")
            if ts is not None:
                if isinstance(ts, dict) and "edges" in ts:
                    for parent, child in ts["edges"]:
                        if parent in node_to_idx and child in node_to_idx:
                            gold_edges.append({
                                "source": parent,
                                "target": child,
                                "source_idx": node_to_idx[parent],
                                "target_idx": node_to_idx[child],
                            })
                elif isinstance(ts, (list, tuple)):
                    for ninfo in ts:
                        src = ninfo.get("qid")
                        for e in ninfo.get("edges", []):
                            prop = e.get("property")
                            tgt = e.get("target_qid")
                            if prop == "P361":
                                parent, child = tgt, src
                            elif prop == "P527":
                                parent, child = src, tgt
                            else:
                                continue
                            if parent in node_to_idx and child in node_to_idx:
                                gold_edges.append({
                                    "source": parent,
                                    "target": child,
                                    "source_idx": node_to_idx[parent],
                                    "target_idx": node_to_idx[child],
                                })
            finite = np.isfinite(gold)
            root_idx = int(np.argmin(np.where(finite, gold, 0.0).sum(1)))
            depths = {td["nodes"][i]: int(gold[root_idx, i]) for i in range(len(td["nodes"])) if np.isfinite(gold[root_idx, i])}
            record = {
                "tree_id": td["tree_id"],
                "metadata": {
                    "num_nodes": len(td["nodes"]),
                    "root_node": td["nodes"][root_idx],
                    "root_title": td["node_titles"][root_idx],
                    "num_gold_edges": len(gold_edges),
                },
                "nodes": [
                    {
                        "id": td["nodes"][i],
                        "title": td["node_titles"][i],
                        "index": i,
                        "depth_from_root": depths.get(td["nodes"][i]),
                    }
                    for i in range(len(td["nodes"]))
                ],
                "gold_edges": gold_edges,
                "distances": {"predicted": pred.tolist(), "actual": gold.tolist()},
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n✓ Tree‑visualization JSONL saved to {save_path}")


# ================================================================
# Probe loader
# ================================================================

def load_probe(path: str, hidden_dim: int, rank: int, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, TwoWordPSDProbe):
        return obj.to(device)
    if "state_dict" in obj:
        obj = obj["state_dict"]
    probe = TwoWordPSDProbe(hidden_dim, rank, device)
    probe.load_state_dict(obj, strict=False)
    probe.eval()
    return probe


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    data = load_embeddings(args.embedding_dir, args.layer, args.icl_repeat_num)
    hidden_dim = data[0]["hidden_dim"]
    print(f"Layer {args.layer} • hidden_dim {hidden_dim}")

    train_data, test_data = create_splits(data, args.train_ratio, args.seed)

    probe = load_probe(args.probe_path, hidden_dim, args.probe_rank, device)
    print(f"Loaded probe from {args.probe_path}")
    print(f"probe.proj.dtype : {probe.proj.dtype}")

    results: Dict[str, Dict[str, float]] = {}
    for name, split in [("train", train_data), ("test", test_data)]:
        res = distance_metrics(probe, split, device)
        res.update(tree_metrics(probe, split, device))
        results[name] = res
        print(f"{name}: ρ={res['spearman_distance']:.4f}  MAE={res['mae']:.4f}  RMSE={res['rmse']:.4f}  UUAS={res['uuas']:.4f}  NSpr={res['nspearman']:.4f}  Root%={res['root_acc']*100:.2f}")

    if args.icl_repeat_num is not None:
        tag = f"layer{args.layer}_repeat{args.icl_repeat_num if args.icl_repeat_num is not None else 'all'}"
    else:
        tag = f"layer{args.layer}"
    metrics_save_dir = os.path.join(args.save_dir, f'metrics')
    os.makedirs(metrics_save_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_save_dir, f"metrics_{tag}.json")
    with open(metrics_path, "w") as fout:
        json.dump(results, fout, indent=2)
    print(f"\n✓ Metrics JSON saved to {metrics_path}")

    if args.vis_path:
        save_tree_visualization_data(probe, data, device, args.vis_path)


if __name__ == "__main__":
    main()
