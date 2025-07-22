#!/usr/bin/env python3
"""
Generate random embeddings for hierarchical data
------------------------------------------------
各ツリーごとに異なる乱数シードを使い，かつ --random_seed により
再現性が確保できるように改良した版。

Usage:
    python generate_random_embeddings.py \
        --data_path data.jsonl \
        --output_dir embeddings/ \
        --embedding_dim 768 \
        --random_seed 42
"""

import json
import argparse
import os
import sys
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#                              Argument parsing                               #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate random embeddings for hierarchical data')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the JSONL data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the embeddings')

    # --- Random embedding settings -----------------------------------------
    parser.add_argument('--embedding_dim', type=int, default=768,
                        help='Dimension of random embeddings')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers to simulate '
                             '(for compatibility with original format)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Base random seed for reproducibility')

    # --- Text mode ---------------------------------------------------------
    parser.add_argument('--text_mode', type=str, default='title',
                        choices=['title', 'title_desc_title'],
                        help='Text construction mode')

    # --- Other settings ----------------------------------------------------
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu, cuda)')
    parser.add_argument('--is_verbose', action='store_true',
                        help='Enable verbose output for debugging')
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#                              Data utilities                                 #
# --------------------------------------------------------------------------- #
def load_data(data_path):
    """Load JSONL data and organize by tree_id"""
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist!")
        sys.exit(1)

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]

    data = []
    for i, line in enumerate(lines):
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error parsing line {i+1}: {e}")
            sys.exit(1)

    if not data:
        print("Error: No valid data found!")
        sys.exit(1)

    tree_groups = defaultdict(list)
    for item in data:
        tree_groups[item['tree_id']].append(item)

    print(f"Loaded {len(data)} nodes across {len(tree_groups)} trees")
    return tree_groups


def build_tree_structure(tree_data):
    """Build node/edge dicts from a list of node dicts belonging to one tree"""
    nodes = {}
    edges = []

    for item in tree_data:
        nodes[item['qid']] = {
            'qid': item['qid'],
            'title': item['wiki_title'],
            'description': item['description']
        }

        for edge in item['edges']:
            if edge['property'] == 'P527':          # has part
                edges.append((item['qid'], edge['target_qid']))
            elif edge['property'] == 'P361':        # part of
                edges.append((edge['target_qid'], item['qid']))

    return {'nodes': nodes, 'edges': edges}


def build_input_text(node, text_mode):
    """Return text string according to text_mode"""
    title = node['title'].strip()
    desc = (node.get('description') or '').strip()

    if text_mode == 'title':
        return title
    elif text_mode == 'title_desc_title':
        return f"{title}. {desc}. {title}" if desc else title
    else:
        raise ValueError(f"Unknown text_mode: {text_mode}")


def calculate_tree_distances(tree_structure):
    """Compute all-pairs shortest path distances (Floyd–Warshall, undirected)"""
    nodes = list(tree_structure['nodes'].keys())
    n = len(nodes)
    dist = np.full((n, n), np.inf)
    for i in range(n):
        dist[i, i] = 0
    idx = {q: i for i, q in enumerate(nodes)}

    for u, v in tree_structure['edges']:
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            dist[i, j] = dist[j, i] = 1  # undirected

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist, nodes


# --------------------------------------------------------------------------- #
#                       Random embedding generation                           #
# --------------------------------------------------------------------------- #
def generate_random_embeddings(texts,
                               embedding_dim,
                               num_layers,
                               dtype,
                               device,
                               debug=False):
    """
    Generate random embeddings. 乱数シードは呼び出し側で設定すること！
    Returns: tensor [N, num_layers, embedding_dim]
    """
    num_texts = len(texts)
    # 直接 device 上に生成した方が速い
    embeddings = torch.randn(num_texts,
                             num_layers,
                             embedding_dim,
                             dtype=dtype,
                             device=device)

    if debug:
        print(f"  Generated embeddings shape: {tuple(embeddings.shape)}")
        print(f"  Mean {embeddings.mean().item():.4f}, "
              f"Std {embeddings.std().item():.4f}")
    return embeddings


def save_embeddings(embeddings_tensor, metadata, output_path):
    """Save embeddings + metadata via torch.save"""
    torch.save({
        'embeddings': embeddings_tensor,
        'metadata': metadata,
        'embedding_shape': tuple(embeddings_tensor.shape),
        'torch_version': torch.__version__
    }, output_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✓ Saved to {output_path}  ({size_mb:.2f} MB)")


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device           : {device}")
    print(f"Base random seed : {args.random_seed}")
    print(f"Embedding dim    : {args.embedding_dim}")
    print(f"Num layers       : {args.num_layers}")

    # 出力ディレクトリ
    output_root = os.path.join(args.output_dir, args.text_mode)
    os.makedirs(output_root, exist_ok=True)

    # データ読み込み
    tree_groups = load_data(args.data_path)

    # dtype  (bfloat16 は CPU 非対応環境ではエラーになるので注意)
    dtype = torch.bfloat16 if args.device.startswith('cuda') else torch.float32

    # ---- 各ツリーを処理 ----------------------------------------------------
    for tree_idx, (tree_id, tree_data) in enumerate(tree_groups.items()):
        print(f"\n{'='*20} Tree {tree_id} ({tree_idx}) {'='*20}")

        # ツリー固有シード = base_seed + tree_idx
        tree_seed = (args.random_seed + tree_idx) % 2**32
        torch.manual_seed(tree_seed)
        np.random.seed(tree_seed)
        if args.is_verbose:
            print(f"  Using tree_seed = {tree_seed}")

        # 構造・入力文
        tree_structure = build_tree_structure(tree_data)
        distances, nodes = calculate_tree_distances(tree_structure)
        node_texts = [
            build_input_text(tree_structure['nodes'][n], args.text_mode)
            for n in nodes
        ]

        # 埋め込み生成
        embeddings = generate_random_embeddings(
            texts=node_texts,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            dtype=dtype,
            device=device,
            debug=args.is_verbose)

        # メタデータ
        metadata = dict(
            tree_id=tree_id,
            nodes=nodes,
            node_titles=[tree_structure['nodes'][n]['title'] for n in nodes],
            input_texts=node_texts,
            text_mode=args.text_mode,
            tree_structure=tree_structure,
            distances=distances,
            model_type='random',
            model_name=f'random_embeddings_dim{args.embedding_dim}_seed{args.random_seed}',
            num_layers=args.num_layers,
            embedding_dim=args.embedding_dim,
            random_seed=tree_seed
        )

        # 保存
        out_path = os.path.join(output_root, f'tree_{tree_id}_embeddings.pt')
        save_embeddings(embeddings, metadata, out_path)

    print(f"\n✓ All random embeddings saved under: {output_root}")


if __name__ == "__main__":
    main()