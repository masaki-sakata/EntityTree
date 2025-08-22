#!/usr/bin/env python3
"""
Train a structural probe on split-tree embeddings only (train set only, no internal split).

- Expects files like:
  /work03/.../title/tree_1_split_0_embeddings.pt  ... tree_1_split_999_embeddings.pt

- Supports:
  * Distance probe (TwoWordPSDProbe): learns a low-rank projection whose induced L2^2
    distances align with tree distances (metadata['distances']).
  * Depth probe (OneWordPSDProbe): learns a projection whose L2^2 norms align with
    node depths. If depths are not in metadata, they are computed from tree_structure.

Notes:
- Uses all files under --embedding_dir matching --glob (default "*_split_*_embeddings.pt").
- No train/test split is performed inside; everything is treated as training data.
- Periodic evaluation is done on the training set for monitoring (can disable with --no_eval).

Usage example:
  python train_probe.py \
    --embedding_dir /work03/masaki/model/hierarchical-repr/gpt2/train250/title \
    --layer 6 \
    --probe_type distance \
    --probe_rank 128 \
    --epochs 300 \
    --lr 1e-3 \
    --device auto \
    --save_dir probe_results_gpt2_train250_title
"""

import os, sys, glob, json, argparse
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scipy.stats import spearmanr

# =================== Probes ===================

class TwoWordPSDProbe(nn.Module):
    """Squared L2 distance after low-rank projection"""
    def __init__(self, model_dim: int, probe_rank: int, device, dtype=torch.bfloat16):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(model_dim, probe_rank, dtype=dtype))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        x = x.to(self.proj.dtype)
        x = torch.matmul(x, self.proj)          # (B, N, R)
        x1 = x.unsqueeze(2)                     # (B, N, 1, R)
        dist = (x1 - x1.transpose(1, 2)).pow(2).sum(-1)  # (B, N, N)
        return dist

class OneWordPSDProbe(nn.Module):
    """Squared L2 norm after projection -> depth"""
    def __init__(self, model_dim: int, probe_rank: int, device, dtype=torch.bfloat16):
        super().__init__()
        self.proj = nn.Parameter(torch.zeros(model_dim, probe_rank, dtype=dtype))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, N, D)
        batch = batch.to(self.proj.dtype)
        transformed = torch.matmul(batch, self.proj)  # (B, N, R)
        B, N, R = transformed.shape
        # Squared norms
        norms = torch.bmm(
            transformed.view(B*N, 1, R),
            transformed.view(B*N, R, 1)
        ).view(B, N)
        return norms  # (B, N)

# =================== Losses ===================

class L1DistanceLoss(nn.Module):
    """L1 on finite entries of the gold distance matrix."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical safety regardless of model dtype.
        pred = pred.float()
        gold = gold.float()
        valid = torch.isfinite(gold)
        if not valid.any():
            return torch.tensor(0., device=pred.device)
        diff = torch.abs(pred[valid] - gold[valid])
        if self.reduction == "sum":
            return diff.sum()
        if self.reduction == "none":
            return diff
        return diff.mean()

class L1DepthLoss(nn.Module):
    """L1 loss for depth vectors; label -1 is treated as invalid (masked out)."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, gold: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = pred.float()
        gold = gold.float()
        valid_mask = (gold != -1.0)
        if not valid_mask.any():
            return torch.tensor(0., device=pred.device)
        if lengths is not None:
            # Normalize per sentence length (only counting valid positions)
            B = pred.size(0)
            loss = 0.0
            denom = 0.0
            for b in range(B):
                vb = valid_mask[b]
                if vb.any():
                    per_sent = torch.abs(pred[b][vb] - gold[b][vb]).mean()
                    loss += per_sent
                    denom += 1.0
            return loss / max(denom, 1.0)
        else:
            return torch.abs(pred[valid_mask] - gold[valid_mask]).mean()

# =================== Utilities ===================

def pick_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)

def probe_dtype_for(device: torch.device) -> torch.dtype:
    # Keep bfloat16 on CUDA; prefer float32 on CPU for safety.
    if device.type == 'cuda':
        return torch.bfloat16
    return torch.float32

def collect_split_paths(root: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(root, pattern)))
    if not paths:
        print(f"[ERROR] No files matched: {os.path.join(root, pattern)}")
        sys.exit(1)
    # Be explicit: only use split embeddings
    paths = [p for p in paths if "_split_" in os.path.basename(p)]
    if not paths:
        print(f"[ERROR] Pattern matched files, but none contained '_split_' in filename.")
        sys.exit(1)
    return paths

def compute_depths_from_tree(tree_structure: Dict, node_order: List[str]) -> List[int]:
    """
    Compute node depths from directed edges (parent -> child).
    Multiple roots are allowed; each root has depth 0.
    Unreachable nodes get depth -1.
    """
    nodes = set(tree_structure['nodes'].keys())
    edges = tree_structure['edges']  # list of (parent, child)
    parents = {n: set() for n in nodes}
    children = {n: set() for n in nodes}
    for p, c in edges:
        if p in nodes and c in nodes:
            parents[c].add(p)
            children[p].add(c)
    roots = [n for n in nodes if len(parents[n]) == 0]
    from collections import deque
    depth = {n: -1 for n in nodes}
    dq = deque()
    for r in roots:
        depth[r] = 0
        dq.append(r)
    while dq:
        u = dq.popleft()
        for v in children[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                dq.append(v)
    # Return in the order used by the embedding tensor
    return [depth.get(qid, -1) for qid in node_order]

def load_train_embeddings(embedding_dir: str, layer: int, probe_type: str, pattern: str) -> List[Dict]:
    """
    Load ONLY split-tree embeddings for training.
    Returns a list of dicts with:
      - embeddings: (N, D_layer)
      - nodes: ordered node qids
      - node_titles
      - tree_id
      - hidden_dim
      - distances (for distance probe)
      - depths (for depth probe)
    """
    paths = collect_split_paths(embedding_dir, pattern)
    print(f"Found {len(paths)} split embedding files")

    data_list = []
    for p in paths:
        try:
            try:
                blob = torch.load(p, map_location='cpu', weights_only=False)
            except TypeError:
                # For older PyTorch that doesn't have weights_only kwarg
                blob = torch.load(p, map_location='cpu')



            emb = blob['embeddings']  # (N, L, D)
            if layer >= emb.shape[1]:
                raise ValueError(f"Requested layer {layer} but embeddings have {emb.shape[1]} layers")
            layer_emb = emb[:, layer, :]  # keep original dtype

            md = blob['metadata']
            entry = {
                'embeddings': layer_emb,                 # (N, D)
                'nodes': md['nodes'],                    # ordered qids
                'node_titles': md['node_titles'],
                'tree_id': md['tree_id'],
                'hidden_dim': layer_emb.shape[1],
            }

            if probe_type == 'distance':
                if 'distances' not in md:
                    raise ValueError("Metadata missing 'distances' for distance probe.")
                # Ensure numeric tensor
                d = torch.tensor(md['distances'], dtype=torch.float32)  # (N, N)
                entry['distances'] = d
            else:  # depth
                if 'depths' in md:
                    entry['depths'] = torch.tensor(md['depths'], dtype=torch.float32)
                else:
                    ts = md.get('tree_structure', None)
                    if ts is None:
                        raise ValueError("Metadata missing 'depths' and 'tree_structure'; cannot compute depths.")
                    depths = compute_depths_from_tree(ts, md['nodes'])
                    entry['depths'] = torch.tensor(depths, dtype=torch.float32)

            data_list.append(entry)
            print(f"  loaded {os.path.basename(p)} "
                  f"(tree {entry['tree_id']}, nodes={len(entry['nodes'])}, dtype={layer_emb.dtype})")
        except Exception as e:
            print(f"[FATAL] Failed to load {p}: {e}")
            sys.exit(1)

    return data_list

# =================== Evaluation (on train set for monitoring) ===================

def eval_distance_probe(probe, data, device):
    probe.eval()
    preds, golds = [], []
    with torch.no_grad():
        for td in data:
            x = td['embeddings'].unsqueeze(0).to(device)   # (1, N, D)
            d_pred = probe(x).squeeze(0).float().cpu()     # (N, N)
            d_gold = td['distances'].float()               # (N, N)
            N = d_gold.size(0)
            for i in range(N):
                for j in range(i+1, N):
                    if torch.isfinite(d_gold[i, j]):
                        preds.append(d_pred[i, j].item())
                        golds.append(d_gold[i, j].item())
    if not preds:
        return dict(spearman=0.0, mae=float('inf'), rmse=float('inf'), num_pairs=0)
    preds, golds = np.array(preds), np.array(golds)
    s, _ = spearmanr(preds, golds)
    return dict(
        spearman=float(s),
        mae=float(np.abs(preds-golds).mean()),
        rmse=float(np.sqrt(((preds-golds)**2).mean())),
        num_pairs=int(len(preds))
    )

def eval_depth_probe(probe, data, device):
    probe.eval()
    preds, golds = [], []
    with torch.no_grad():
        for td in data:
            x = td['embeddings'].unsqueeze(0).to(device)  # (1, N, D)
            y = td['depths'].float()                      # (N,)
            p = probe(x).squeeze(0).float().cpu()         # (N,)
            valid = (y != -1)
            if valid.any():
                preds.extend(p[valid].numpy().tolist())
                golds.extend(y[valid].numpy().tolist())
    if not preds:
        return dict(spearman=0.0, mae=float('inf'), rmse=float('inf'), num_points=0)
    preds, golds = np.array(preds), np.array(golds)
    s, _ = spearmanr(preds, golds)
    return dict(
        spearman=float(s),
        mae=float(np.abs(preds-golds).mean()),
        rmse=float(np.sqrt(((preds-golds)**2).mean())),
        num_points=int(len(preds))
    )

# =================== Training ===================

def train_distance(probe, train_data, args, device):
    opt = optim.Adam(probe.parameters(), lr=args.lr)
    loss_fn = L1DistanceLoss()
    pbar = tqdm(range(args.epochs))
    for ep in pbar:
        probe.train()
        tot = 0.0
        count = 0
        for td in train_data:
            x = td['embeddings'].unsqueeze(0).to(device)     # (1, N, D)
            y = td['distances'].unsqueeze(0).to(device)      # (1, N, N)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            count += 1
        if (not args.no_eval) and (ep % args.eval_every == 0):
            m = eval_distance_probe(probe, train_data, device)
            pbar.set_description(f"ep{ep} loss {tot/max(count,1):.4f} | train ρ {m['spearman']:.4f}")
    return probe

def train_depth(probe, train_data, args, device):
    opt = optim.Adam(probe.parameters(), lr=args.lr)
    loss_fn = L1DepthLoss()
    pbar = tqdm(range(args.epochs))
    for ep in pbar:
        probe.train()
        tot = 0.0
        count = 0
        for td in train_data:
            x = td['embeddings'].unsqueeze(0).to(device)     # (1, N, D)
            y = td['depths'].unsqueeze(0).to(device)         # (1, N)
            lengths = torch.tensor([len(td['nodes'])], device=device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(x), y, lengths)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            count += 1
        if (not args.no_eval) and (ep % args.eval_every == 0):
            m = eval_depth_probe(probe, train_data, device)
            pbar.set_description(f"ep{ep} loss {tot/max(count,1):.4f} | train ρ {m['spearman']:.4f}")
    return probe

# =================== CLI ===================

def parse_args():
    pa = argparse.ArgumentParser(description="Train structural probe on split-tree embeddings (train only).")
    pa.add_argument('--embedding_dir', required=True, help='Directory containing *_split_*_embeddings.pt files')
    pa.add_argument('--layer', type=int, required=True, help='Layer index to use from saved embeddings')
    pa.add_argument('--probe_type', choices=['distance', 'depth'], default='distance')
    pa.add_argument('--probe_rank', type=int, default=128)
    pa.add_argument('--epochs', type=int, default=1000)
    pa.add_argument('--lr', type=float, default=1e-3)
    pa.add_argument('--batch_size', type=int, default=1)  # kept for interface parity; training iterates per-tree
    pa.add_argument('--device', default='auto')
    pa.add_argument('--save_dir', default='probe_results')
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--glob', default='*_split_*_embeddings.pt',
                    help='Glob under embedding_dir to pick training files (default: *_split_*_embeddings.pt)')
    pa.add_argument('--eval_every', type=int, default=100, help='Evaluate on train set every K epochs')
    pa.add_argument('--no_eval', action='store_true', help='Disable periodic evaluation')
    return pa.parse_args()

def main():
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Probe type: {args.probe_type}")
    print(f"Reading training splits from: {args.embedding_dir} (glob={args.glob})")

    # ---- Load training data (split trees only) ----
    train_data = load_train_embeddings(args.embedding_dir, args.layer, args.probe_type, args.glob)
    hidden_dim = train_data[0]['hidden_dim']
    print(f"\nLayer {args.layer}, hidden dim {hidden_dim}, train trees={len(train_data)}")

    # ---- Build probe ----
    dtype = probe_dtype_for(device)
    if args.probe_type == 'distance':
        probe = TwoWordPSDProbe(hidden_dim, args.probe_rank, device, dtype=dtype)
        print(f"TwoWordPSDProbe rank {args.probe_rank} (dtype={dtype})")
        probe = train_distance(probe, train_data, args, device)
        if not args.no_eval:
            m = eval_distance_probe(probe, train_data, device)
            print(f"\n[Final train] ρ={m['spearman']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  pairs={m['num_pairs']}")
        save_name = f'probe_distance_layer{args.layer}_rank{args.probe_rank}.pt'
    else:
        probe = OneWordPSDProbe(hidden_dim, args.probe_rank, device, dtype=dtype)
        print(f"OneWordPSDProbe rank {args.probe_rank} (dtype={dtype})")
        probe = train_depth(probe, train_data, args, device)
        if not args.no_eval:
            m = eval_depth_probe(probe, train_data, device)
            print(f"\n[Final train] ρ={m['spearman']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  points={m['num_points']}")
        save_name = f'probe_depth_layer{args.layer}_rank{args.probe_rank}.pt'

    # ---- Save ----
    out_path = os.path.join(args.save_dir, save_name)
    torch.save(probe.state_dict(), out_path)
    print(f"✓ Saved probe to {out_path}")

    # ---- Save simple training manifest for reproducibility ----
    manifest = {
        "embedding_dir": args.embedding_dir,
        "glob": args.glob,
        "layer": args.layer,
        "probe_type": args.probe_type,
        "probe_rank": args.probe_rank,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": str(device),
        "seed": args.seed,
        "num_train_files": len(train_data),
        "save_path": out_path,
    }
    with open(os.path.join(args.save_dir, save_name + ".json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("✓ Done")

if __name__ == "__main__":
    main()
