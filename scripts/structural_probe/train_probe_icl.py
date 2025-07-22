#!/usr/bin/env python3
"""
Train / test structural probe for hierarchical tree data
"""

import argparse, os, sys, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from scipy.stats import spearmanr
from tqdm import tqdm
from typing import List, Dict

# ========== Probe と損失 ==========
class TwoWordPSDProbe(nn.Module):
    """Squared L2 distance after low-rank projection"""
    def __init__(self, model_dim: int, probe_rank: int, device, dtype=torch.bfloat16):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(model_dim, probe_rank, dtype=dtype))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, N, D)
        x = x.to(self.proj.dtype)
        x = torch.matmul(x, self.proj)                           # (B, N, R)
        x1 = x.unsqueeze(2)                                      # (B, N, 1, R)
        dist = (x1 - x1.transpose(1, 2)).pow(2).sum(-1)          # (B, N, N)
        return dist

class L1DistanceLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred, gold):
        valid = torch.isfinite(gold)
        if not valid.any():
            return torch.tensor(0., device=pred.device, dtype=pred.dtype)
        diff = torch.abs(pred[valid] - gold[valid])
        if self.reduction == "sum":
            return diff.sum()
        if self.reduction == "none":
            return diff
        return diff.mean()

# ========== 引数 ==========
def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('--embedding_dir', required=True)
    pa.add_argument('--layer', type=int, required=True)
    pa.add_argument('--probe_rank', type=int, default=128)
    pa.add_argument('--epochs', type=int, default=1000)
    pa.add_argument('--lr', type=float, default=1e-3)
    pa.add_argument('--train_ratio', type=float, default=.8)
    pa.add_argument('--batch_size', type=int, default=1)
    pa.add_argument('--device', default='auto')
    pa.add_argument('--save_dir', default='probe_results')
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--icl_repeat_num', type=int, default=None,
                    help='例: 1 を指定すると repeat_1 ディレクトリのみを読み込む')
    return pa.parse_args()

# ========== Embedding 読み込み ==========
def _collect_embedding_paths(root: str, icl_repeat_num: int = None) -> List[str]:
    """指定 repeat のみの .pt を再帰的に収集する"""
    paths = []
    if icl_repeat_num is None:
        # 従来通り: 直下の *_embeddings.pt
        paths = [os.path.join(root, f) for f in os.listdir(root)
                 if f.endswith('_embeddings.pt')]
    else:
        repeat_tag = f"repeat_{icl_repeat_num}"
        for cur, dirs, files in os.walk(root):
            if os.path.basename(cur) == repeat_tag:
                for f in files:
                    if f.endswith('_embeddings.pt'):
                        paths.append(os.path.join(cur, f))
    paths.sort()
    return paths

def load_embeddings(embedding_dir: str, layer: int,
                    icl_repeat_num: int = None) -> List[Dict]:
    paths = _collect_embedding_paths(embedding_dir, icl_repeat_num)
    if not paths:
        print(f"ERROR: No *_embeddings.pt found (repeat={icl_repeat_num}) in {embedding_dir}")
        sys.exit(1)

    data_list = []
    print(f"Found {len(paths)} embedding files")
    for p in paths:
        try:
            blob = torch.load(p, map_location='cpu')
            emb = blob['embeddings']                      # (N, L, D)
            if layer >= emb.shape[1]:
                raise ValueError(f"layer {layer} >= {emb.shape[1]}")
            layer_emb = emb[:, layer, :]                  # keep dtype (bfloat16)

            md = blob['metadata']
            data_list.append({
                'embeddings': layer_emb,
                'distances': torch.tensor(md['distances'], dtype=torch.bfloat16),
                'nodes': md['nodes'],
                'node_titles': md['node_titles'],
                'tree_id': md['tree_id'],
                'tree_structure': md.get('tree_structure'),
                'hidden_dim': layer_emb.shape[1]
            })
            print(f"  loaded {os.path.basename(p)}  "
                  f"(tree {md['tree_id']}, {len(md['nodes'])} nodes, dtype={layer_emb.dtype})")
        except Exception as e:
            print(f"FAILED to load {p} : {e}")
            sys.exit(1)
    return data_list

# ========== split ==========
def create_data_splits(data, ratio, seed):
    np.random.seed(seed)
    idx = np.random.permutation(len(data))
    n_train = max(1, min(len(data)-1, int(len(data)*ratio)))
    train = [data[i] for i in idx[:n_train]]
    test  = [data[i] for i in idx[n_train:]]
    print(f"\nData split: train {len(train)} trees / test {len(test)} trees")
    return train, test

# ========== 評価 ==========
def evaluate_probe(probe, data, device):
    probe.eval()
    preds, golds = [], []
    with torch.no_grad():
        for td in data:
            d_pred = probe(td['embeddings'].unsqueeze(0).to(device)).squeeze(0)
            d_gold = td['distances']
            n = d_gold.size(0)
            for i in range(n):
                for j in range(i+1, n):
                    if torch.isfinite(d_gold[i, j]):
                        preds.append(d_pred[i, j].float().item())
                        golds.append(d_gold[i, j].float().item())
    if not preds:
        return dict(spearman=0., mae=float('inf'), rmse=float('inf'), num_pairs=0)
    s, _ = spearmanr(preds, golds)
    preds, golds = np.array(preds), np.array(golds)
    return dict(spearman=s, mae=np.abs(preds-golds).mean(),
                rmse=np.sqrt(((preds-golds)**2).mean()), num_pairs=len(preds))

# ========== 学習 ==========
def train_probe(probe, train_data, test_data, args, device):
    opt = optim.Adam(probe.parameters(), lr=args.lr)
    loss_fn = L1DistanceLoss()
    pbar = tqdm(range(args.epochs))
    for ep in pbar:
        probe.train(); tot=0; c=0
        for td in train_data:
            x = td['embeddings'].unsqueeze(0).to(device)
            y = td['distances'].unsqueeze(0).to(device)
            opt.zero_grad()
            loss = loss_fn(probe(x), y)
            loss.backward(); opt.step()
            tot += loss.item(); c+=1
        if ep % 10 == 0:
            metrics = evaluate_probe(probe, test_data, device)
            pbar.set_description(f"ep{ep} loss {tot/c:.4f}  testρ {metrics['spearman']:.4f}")
    return probe

# ========== 可視化データ保存（省略可：元スクリプトと同じロジックを使用したい場合は貼り替え） ==========
def save_tree_visualization_data(*_): pass   # ここでは省略（必要なら元関数をそのまま貼り付けて下さい）

# ========== main ==========
def main():
    args = parse_args()
    device = torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available())
                          else (args.device if args.device!='auto' else 'cpu'))
    print(f"Device: {device}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Load ----
    data = load_embeddings(args.embedding_dir, args.layer, args.icl_repeat_num)
    hidden_dim = data[0]['hidden_dim']
    print(f"\nLayer {args.layer}, hidden dim {hidden_dim}")

    # ---- Split ----
    train_data, test_data = create_data_splits(data, args.train_ratio, args.seed)

    # ---- Probe ----
    probe = TwoWordPSDProbe(hidden_dim, args.probe_rank, device)
    print(f"Probe rank {args.probe_rank} (dtype=bfloat16)")

    probe = train_probe(probe, train_data, test_data, args, device)

    # ---- Final evaluation ----
    print("\n=== Final evaluation ===")
    for name, d in [('train', train_data), ('test', test_data)]:
        m = evaluate_probe(probe, d, device)
        print(f"{name}: ρ={m['spearman']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f} "
              f"pairs={m['num_pairs']}")

    # ---- Save ----
    torch.save(probe.state_dict(),
               os.path.join(args.save_dir,
                            f'probe_layer{args.layer}_rank{args.probe_rank}.pt'))
    print("✓ done")

if __name__ == '__main__':
    main()