#!/usr/bin/env python3
"""
Train and test structural probe for hierarchical tree data

This script trains a structural probe to learn whether language model representations
encode hierarchical tree structure between named entities.

Usage:
    python train_tree_probe.py --embedding_dir embeddings/ --layer 12 --probe_rank 128 --epochs 1000

The script outputs a JSONL file with tree structure data that can be loaded as:
    import json
    trees = []
    with open('tree_structure_layer12.jsonl', 'r') as f:
        for line in f:
            trees.append(json.loads(line))
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from tqdm import tqdm
import json
from IPython import embed

# Import probe and loss classes
class TwoWordPSDProbe(nn.Module):
    """Computes squared L2 distance after projection by a matrix."""
    def __init__(self, model_dim, probe_rank, device, dtype=torch.float32):
        super(TwoWordPSDProbe, self).__init__()
        self.model_dim = model_dim
        self.probe_rank = probe_rank
        self.dtype = dtype
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank, dtype=dtype))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """
        Args:
            batch: word representations of shape (batch_size, seq_len, representation_dim)
        Returns:
            A tensor of distances of shape (batch_size, seq_len, seq_len)
        """
        # Ensure batch is in the same dtype as the probe
        batch = batch.to(self.dtype)
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances

class L1DistanceLoss(nn.Module):
    """L1 loss that safely ignores entries whose gold距離が Inf の部分."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self,
                predictions: torch.Tensor,   # (B, N, N)
                gold:         torch.Tensor   # (B, N, N)
               ) -> torch.Tensor:
        """
        0. まず有効な要素 (finite) を布で抽出
        1. その要素だけを集めて L1 差を計算
        2. reduction に従い平均 or 総和
        """
        # True になっている所だけ学習に使う
        valid = torch.isfinite(gold)

        if not valid.any():
            # バッチ内に学習対象が無い場合
            return torch.tensor(0.0,
                                device=predictions.device,
                                dtype=predictions.dtype)

        diff = torch.abs(predictions[valid] - gold[valid])

        if self.reduction == "sum":
            return diff.sum()
        elif self.reduction == "none":
            return diff
        # default: mean
        return diff.mean()


def parse_args():
    parser = argparse.ArgumentParser(description='Train structural probe for hierarchical data')
    parser.add_argument('--embedding_dir', type=str, required=True,
                       help='Directory containing the embeddings')
    parser.add_argument('--layer', type=int, required=True,
                       help='Which layer to use (0-indexed)')
    parser.add_argument('--probe_rank', type=int, default=128,
                       help='Rank of the probe projection matrix')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (number of trees per batch)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save_dir', type=str, default='probe_results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--icl_repeat_num', type=int, default=None,
                       help='')
    
    return parser.parse_args()

def load_embeddings(embedding_dir, layer):
    """Load embeddings from saved files"""
    embeddings_data = []
    
    # Find all embedding files
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('_embeddings.pt')]
    
    if not embedding_files:
        print(f"Error: No embedding files found in {embedding_dir}")
        sys.exit(1)
    
    print(f"Found {len(embedding_files)} embedding files")
    
    for filename in sorted(embedding_files):
        filepath = os.path.join(embedding_dir, filename)
        try:
            # Load the torch saved file
            data = torch.load(filepath, map_location='cpu')
            
            # Extract embeddings at specified layer
            embeddings = data['embeddings']  # Shape: [num_nodes, num_layers, hidden_dim]
            
            if layer >= embeddings.shape[1]:
                print(f"Error: Layer {layer} not available. Model has {embeddings.shape[1]} layers.")
                sys.exit(1)
            
            layer_embeddings = embeddings[:, layer, :]  # Keep original dtype (float32)
            
            # Get metadata
            metadata = data['metadata']
            
            embeddings_data.append({
                'embeddings': layer_embeddings,
                'distances': torch.tensor(metadata['distances'], dtype=torch.float32),
                'nodes': metadata['nodes'],
                'node_titles': metadata['node_titles'],
                'tree_id': metadata['tree_id'],
                'tree_structure': metadata['tree_structure'],  # Add this to preserve edge information
                'hidden_dim': layer_embeddings.shape[1]
            })
            
            print(f"  Loaded tree {metadata['tree_id']}: {len(metadata['nodes'])} nodes, "
                  f"hidden_dim={layer_embeddings.shape[1]}, dtype={layer_embeddings.dtype}")
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            sys.exit(1)
    
    return embeddings_data

def create_data_splits(embeddings_data, train_ratio, seed):
    """Split data into train and test sets by entire trees"""
    np.random.seed(seed)
    
    # Split by entire trees, not nodes
    num_trees = len(embeddings_data)
    indices = np.arange(num_trees)
    np.random.shuffle(indices)
    
    train_size = int(num_trees * train_ratio)
    if train_size == 0:
        train_size = 1  # At least one tree for training
    if train_size == num_trees:
        train_size = num_trees - 1  # At least one tree for testing
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = [embeddings_data[i] for i in train_indices]
    test_data = [embeddings_data[i] for i in test_indices]
    
    # Print detailed split information
    print(f"\nData splits created (by entire trees):")
    print(f"  Train: {len(train_data)} trees")
    for tree in train_data:
        print(f"    - Tree {tree['tree_id']}: {len(tree['nodes'])} nodes")
    print(f"  Test: {len(test_data)} trees (unseen during training)")
    for tree in test_data:
        print(f"    - Tree {tree['tree_id']}: {len(tree['nodes'])} nodes")
    
    return train_data, test_data

def evaluate_probe(probe, data, device):
    """Evaluate probe on data"""
    probe.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for tree_data in data:
            # Prepare batch
            embeddings = tree_data['embeddings'].unsqueeze(0).to(device)  # [1, num_nodes, hidden_dim]
            distances = tree_data['distances'].to(device)  # [num_nodes, num_nodes]
            
            # Get predictions
            predicted_distances = probe(embeddings).squeeze(0)  # [num_nodes, num_nodes]
            
            # Flatten and collect valid pairs
            n = predicted_distances.shape[0]
            for i in range(n):
                for j in range(i+1, n):  # Only upper triangle
                    if not torch.isinf(distances[i, j]):
                        # Convert to float32 for numerical stability in metrics
                        all_predictions.append(predicted_distances[i, j].float().item())
                        all_labels.append(distances[i, j].float().item())
    
    # Calculate metrics
    if len(all_predictions) > 0:
        # Spearman correlation
        spearman_corr, _ = spearmanr(all_predictions, all_labels)
        
        # Mean absolute error
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_labels)))
        
        # Root mean squared error
        rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_labels))**2))
        
        return {
            'spearman': spearman_corr,
            'mae': mae,
            'rmse': rmse,
            'num_pairs': len(all_predictions)
        }
    else:
        return {
            'spearman': 0.0,
            'mae': float('inf'),
            'rmse': float('inf'),
            'num_pairs': 0
        }

def train_probe(probe, train_data, test_data, args, device):
    """Train the probe"""
    optimizer = optim.Adam(probe.parameters(), lr=args.lr)
    loss_fn = L1DistanceLoss()
    
    train_losses = []
    test_metrics = []
    
    print("\nTraining probe...")
    pbar = tqdm(range(args.epochs))
    
    for epoch in pbar:
        probe.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training loop
        for tree_data in train_data:
            # Prepare batch
            embeddings = tree_data['embeddings'].unsqueeze(0).to(device)  # [1, num_nodes, hidden_dim]
            distances = tree_data['distances'].unsqueeze(0).to(device)  # [1, num_nodes, num_nodes]
            
            # Forward pass
            optimizer.zero_grad()
            predicted_distances = probe(embeddings)
            
            # Compute loss
            loss = loss_fn(predicted_distances, distances)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_loss)
        
        # Evaluate on test set every 10 epochs
        if epoch % 10 == 0:
            test_metrics_epoch = evaluate_probe(probe, test_data, device)
            test_metrics.append(test_metrics_epoch)
            
            pbar.set_description(
                f"Loss: {avg_loss:.4f}, Test Spearman: {test_metrics_epoch['spearman']:.4f}, "
                f"Test MAE: {test_metrics_epoch['mae']:.4f}"
            )
    
    return train_losses, test_metrics

def save_tree_visualization_data(probe, tree_data_list, device, save_path):
    """
    Save data for tree structure visualization in JSONL format.
    Handles two possible formats of `tree_structure`:

      a) {'nodes': {...}, 'edges': [(parent, child), ...]}
      b) [ {'qid': qid, 'edges': [...]}, ... ]   ← 古い想定
    """
    probe.eval()

    with open(save_path, "w") as f:
        for tree_data in tree_data_list:
            # ---------- 推定距離行列 ----------
            with torch.no_grad():
                preds = probe(tree_data["embeddings"]
                               .unsqueeze(0).to(device)).squeeze(0).float().cpu().numpy()
            gold = tree_data["distances"].float().numpy()

            # ---------- Gold edge 取得 ----------
            gold_edges = []
            node_to_idx = {qid: idx for idx, qid in enumerate(tree_data["nodes"])}

            ts = tree_data.get("tree_structure", None)
            if ts is not None:
                # pattern-1  : dict で 'edges' がある
                if isinstance(ts, dict) and "edges" in ts:
                    for parent_qid, child_qid in ts["edges"]:
                        if parent_qid in node_to_idx and child_qid in node_to_idx:
                            gold_edges.append({
                                "source": parent_qid,
                                "target": child_qid,
                                "source_idx": node_to_idx[parent_qid],
                                "target_idx": node_to_idx[child_qid],
                                "property": "unknown",
                                "direction": "parent_to_child"
                            })
                # pattern-2 : 旧式 list 形式
                elif isinstance(ts, (list, tuple)):
                    for node_info in ts:
                        src_qid = node_info.get("qid")
                        for edge in node_info.get("edges", []):
                            prop = edge.get("property")
                            tgt_qid = edge.get("target_qid")

                            if prop == "P361":  # (src が子)
                                parent_qid, child_qid = tgt_qid, src_qid
                            elif prop == "P527":  # (src が親)
                                parent_qid, child_qid = src_qid, tgt_qid
                            else:
                                continue

                            if parent_qid in node_to_idx and child_qid in node_to_idx:
                                gold_edges.append({
                                    "source": parent_qid,
                                    "target": child_qid,
                                    "source_idx": node_to_idx[parent_qid],
                                    "target_idx": node_to_idx[child_qid],
                                    "property": prop,
                                    "direction": "parent_to_child"
                                })

            # ---------- root 推定 & depth ----------
            # 無限大を除いた合計で root を決める
            finite_mask = np.isfinite(gold)
            row_sums = np.where(finite_mask, gold, 0).sum(axis=1)
            root_idx = int(np.argmin(row_sums))   

            depths = {tree_data["nodes"][i]: int(gold[root_idx, i])
                      for i in range(len(tree_data["nodes"])) if np.isfinite(gold[root_idx, i])}

            # ---------- 可視化用 dict ----------
            vis = {
                "tree_id": tree_data["tree_id"],
                "metadata": {
                    "num_nodes": len(tree_data["nodes"]),
                    "root_node": tree_data["nodes"][root_idx],
                    "root_node_title": tree_data["node_titles"][root_idx],
                    "num_gold_edges": len(gold_edges)
                },
                "nodes": [
                    {
                        "id": tree_data["nodes"][i],
                        "title": tree_data["node_titles"][i],
                        "index": i,
                        "depth_from_root": depths.get(tree_data["nodes"][i], None)
                    } for i in range(len(tree_data["nodes"]))
                ],
                "gold_edges": gold_edges,
                "distances": {
                    "predicted": preds.tolist(),
                    "actual": gold.tolist()
                }
            }

            f.write(json.dumps(vis, ensure_ascii=False) + "\n")

            print(f"Tree {tree_data['tree_id']:>3}: "
                  f"{len(tree_data['nodes'])} nodes, "
                  f"{len(gold_edges)} gold edges, "
                  f"root = {tree_data['node_titles'][root_idx]}")
    print(f"\n✓ Tree-visualization data written to {save_path}")

def main():
    args = parse_args()
    
    print("Starting training script with arguments:")
    print(f"lr: {args.lr}, epochs: {args.epochs}, batch_size: {args.batch_size}, "
          f"probe_rank: {args.probe_rank}, layer: {args.layer}, "
          f"embedding_dir: {args.embedding_dir}, save_dir: {args.save_dir}, "
          f"train_ratio: {args.train_ratio}, seed: {args.seed}, device: {args.device}")

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load embeddings
    print(f"\nLoading embeddings from {args.embedding_dir}")
    embeddings_data = load_embeddings(args.embedding_dir, args.layer)
    
    if not embeddings_data:
        print("Error: No embeddings loaded")
        sys.exit(1)
    
    # Get hidden dimension
    hidden_dim = embeddings_data[0]['hidden_dim']
    print(f"\nUsing layer {args.layer} with hidden dimension {hidden_dim}")
    
    # Create data splits
    train_data, test_data = create_data_splits(embeddings_data, args.train_ratio, args.seed)
    
    # Initialize probe
    probe = TwoWordPSDProbe(hidden_dim, args.probe_rank, device, dtype=torch.float32)
    print(f"\nInitialized probe with rank {args.probe_rank} using float32")
    
    # Train probe
    train_losses, test_metrics = train_probe(probe, train_data, test_data, args, device)
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation (Tree-level generalization):")
    
    # Evaluate on train set
    train_metrics = evaluate_probe(probe, train_data, device)
    print(f"\nTrain set (seen trees):")
    print(f"  Trees: {[tree['tree_id'] for tree in train_data]}")
    print(f"  Spearman correlation: {train_metrics['spearman']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  Num pairs: {train_metrics['num_pairs']}")
    
    # Evaluate on test set
    test_metrics_final = evaluate_probe(probe, test_data, device)
    print(f"\nTest set (unseen trees):")
    print(f"  Trees: {[tree['tree_id'] for tree in test_data]}")
    print(f"  Spearman correlation: {test_metrics_final['spearman']:.4f}")
    print(f"  MAE: {test_metrics_final['mae']:.4f}")
    print(f"  RMSE: {test_metrics_final['rmse']:.4f}")
    print(f"  Num pairs: {test_metrics_final['num_pairs']}")
    
    # Save results
    results = {
        'args': vars(args),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics_final,
        'train_losses': train_losses,
        'test_metrics_history': test_metrics
    }
    
    results_path = os.path.join(args.save_dir, f'results_layer{args.layer}_rank{args.probe_rank}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Save probe
    probe_save_dir = os.path.join(args.save_dir, f'probe')
    os.makedirs(probe_save_dir, exist_ok=True)

    probe_path = os.path.join(probe_save_dir, f'probe_layer{args.layer}_rank{args.probe_rank}.pt')
    torch.save(probe.state_dict(), probe_path)
    print(f"Probe saved to {probe_path}")
    
    # Save tree structure visualization data
    if test_data:
        vis_path = os.path.join(args.save_dir, f'tree_structure_layer{args.layer}_rank{args.probe_rank}.jsonl')
        num_examples = min(len(test_data), 5)
        print(f"\nSaving tree structure visualization data for {num_examples} test examples...")
        save_tree_visualization_data(probe, test_data[:num_examples], device, vis_path)
        print(f"Tree structure data saved to {vis_path}")
    
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()