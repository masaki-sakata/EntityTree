#!/usr/bin/env python3
"""
Extract hidden states from language models for hierarchical data

This script extracts hidden states from all layers of a language model
for the last token of each wiki_title in the hierarchical data.

Usage:
    python extract_embeddings.py --data_path paste.txt --output_dir embeddings/ --model_name meta-llama/Meta-Llama-3-8B
"""

import json
import argparse
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Extract hidden states / fastText vectors')
    parser.add_argument('--data_path',  type=str, required=True,
                        help='Path to the JSONL data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the embeddings')

    # --- モデル関連 ------------------------------------------------------
    parser.add_argument('--model_type', choices=['hf', 'fasttext'], default='hf',
                        help='hf  : HuggingFace Transformer\n'
                             'fasttext : fastText vectors')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B',
                        help='HuggingFace model name（--model_type hf のときに使用）')
    parser.add_argument('--vector_path', type=str, default=None,
                        help='fastText の .bin へのパス（--model_type fasttext のとき必須）')

    # --- その他の設定 ----------------------------------------------------
    parser.add_argument('--max_length', type=int, default=500,
                        help='Maximum sequence length (hf 用)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (現状未使用)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--text_mode', type=str, default='title',
                        choices=['title', 'title_desc_title'],
                        help='テキスト構築モード（fastText では title のみ可）')
    parser.add_argument('--is_verbose', action='store_true',
                        help='Enable verbose output for debugging')

    return parser.parse_args()

def load_data(data_path):
    """Load JSONL data and organize by tree_id"""
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist!")
        sys.exit(1)
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {data_path}: {e}")
        sys.exit(1)
    
    # Parse JSONL
    lines = content.strip().split('\n')
    data = []
    
    for i, line in enumerate(lines):
        if line.strip():
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i+1}: {e}")
                sys.exit(1)
    
    if not data:
        print("Error: No valid data found!")
        sys.exit(1)
    
    # Group by tree_id
    tree_groups = defaultdict(list)
    for item in data:
        tree_groups[item['tree_id']].append(item)
    
    print(f"Loaded {len(data)} nodes across {len(tree_groups)} trees")
    for tree_id, items in tree_groups.items():
        print(f"  Tree {tree_id}: {len(items)} nodes")
    
    return tree_groups

def build_tree_structure(tree_data, text_mode):
    """Build tree structure from data"""
    nodes = {}
    edges = []
    
    # Collect node information
    for item in tree_data:
        if text_mode == 'title':
            nodes[item['qid']] = {
                'qid': item['qid'],
                'title': item['wiki_title'],
            }
        else:
            nodes[item['qid']] = {
                'qid': item['qid'],
                'title': item['wiki_title'],
                'description': item['description']
            }
        
        # Collect edge information
        for edge in item['edges']:
            if edge['property'] == 'P527':  # has part
                edges.append((item['qid'], edge['target_qid']))
            elif edge['property'] == 'P361':  # part of
                edges.append((edge['target_qid'], item['qid']))
    
    return {'nodes': nodes, 'edges': edges}


def build_input_text(node, text_mode):
    """
    node  … tree_structure['nodes'][qid] で得られる dict
    text_mode … 'title' あるいは 'title_desc_title'
    """
    title = node['title'].strip()
    desc  = (node.get('description') or '').strip()

    if title == '':
        raise ValueError(f"Title is empty for node {node}")

    if text_mode == 'title':
        return title

    elif text_mode == 'title_desc_title':
        # description が空の場合にドットが重ならないよう注意
        if desc:
            return f"{title}. {desc}. {title}"
        else:
            raise ValueError("Description is empty, cannot use 'title_desc_title' mode without a description.")

    else:
        raise ValueError(f"Unknown text_mode: {text_mode}")


def calculate_tree_distances(tree_structure):
    """Calculate tree distance matrix using Floyd-Warshall"""
    nodes = list(tree_structure['nodes'].keys())
    n = len(nodes)
    
    # Initialize with infinity
    distances = np.full((n, n), np.inf)
    
    # Node to index mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Diagonal elements are 0
    for i in range(n):
        distances[i, i] = 0
    
    # Direct edges have distance 1
    for parent, child in tree_structure['edges']:
        if parent in node_to_idx and child in node_to_idx:
            i, j = node_to_idx[parent], node_to_idx[child]
            distances[i, j] = 1
            distances[j, i] = 1  # Treat as undirected
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distances[i, k] + distances[k, j] < distances[i, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
    
    return distances, nodes

def load_model_and_tokenizer(args, device):
    """
    HuggingFace / fastText を自動で切り替えて読み込む
    戻り値
      tokenizer … hf の場合は AutoTokenizer, fastText の場合 None
      model     … hf の場合は AutoModel,        fastText の場合 fasttext.FastText
    """
    if args.model_type == 'hf':
        try:
            print(f"Loading HF model: {args.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoModel.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                output_hidden_states=True
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model.to(device).eval()
            print(f"✓ HF model loaded on {device}  (layers={model.config.num_hidden_layers})")
            return tokenizer, model

        except Exception as e:
            print(f"[ERROR] HuggingFace model load failed: {e}")
            sys.exit(1)

    # ---------- fastText ------------------------------------------------
    elif args.model_type == 'fasttext':
        if args.vector_path is None:
            print("[ERROR] --model_type fasttext では --vector_path が必須です")
            sys.exit(1)

        try:
            import fasttext
            print(f"Loading fastText vectors from {args.vector_path}")
            ft_model = fasttext.load_model(args.vector_path)
            print(f"✓ fastText model loaded  (dim={ft_model.get_dimension()})")
            return None, ft_model     # tokenizer は使わない
        except Exception as e:
            print(f"[ERROR] fastText model load failed: {e}")
            sys.exit(1)

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")


def _extract_hidden_states_hf(
    texts,
    tokenizer,
    model,
    device,
    max_length: int = 500,
    debug: bool = False,          # ← True にすると詳細表示
    debug_max_print: int = 5      # ← 何件分表示するか
):
    """
    Return
    ------
    torch.Tensor  [num_texts, num_layers, hidden_dim]
        各テキストの「最後の非 PAD トークン」上の隠れ状態
    """

    all_hidden_states = []
    printed = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting embeddings"):

            # --- Tokenize -------------------------------------------------
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # --- Forward --------------------------------------------------
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states   # tuple(len=L+1)

            # --- 最後の非 PAD トークン index ------------------------------
            attn = inputs['attention_mask'][0]          # [seq_len]
            last_idx = attn.nonzero()[-1].item()        # int

            # --- DEBUG PRINT ---------------------------------------------
            if debug and printed < debug_max_print:
                ids      = inputs['input_ids'][0]               # [seq_len]
                id_list  = ids.tolist()
                tokens   = tokenizer.convert_ids_to_tokens(id_list)
                last_id  = id_list[last_idx]
                last_tok = tokens[last_idx]
                last_str = tokenizer.decode([last_id]).strip()

                print("\n" + "=" * 72)
                print(f"TEXT             : {text}")
                print(f"SEQ LEN          : {len(tokens)} (max={max_length})")
                print(f"LAST IDX         : {last_idx}")
                print(f"  └ token id     : {last_id}")
                print(f"  └ token (subwd): {last_tok}")
                print(f"  └ decoded str  : \"{last_str}\"")
                print("ATTN MASK tail   :", attn.tolist()[-10:])  # 末尾10個だけ
                print("=" * 72)
                printed += 1

            # --- collect per-layer vector ---------------------------------
            vecs = []
            for layer_h in hidden_states:          # embedding + each layer
                vecs.append(layer_h[0, last_idx, :].cpu())

            all_hidden_states.append(torch.stack(vecs))  # [L, D]

    return torch.stack(all_hidden_states)                # [N, L, D]



def _extract_fasttext_embeddings(
        texts,
        ft_model,
        debug: bool = False,
        debug_max_print: int = 5):
    """
    fastText 版 : 各 title の token 平均ベクトル
      戻り値  [N, 1, D]  （D = fastText の次元）
    """
    dim = ft_model.get_dimension()
    vec_list = []
    printed = 0

    for text in tqdm(texts, desc="[fastText] Averaging vectors"):
        tokens = text.strip().split()

        if len(tokens) == 0:
            vec = np.zeros(dim, dtype=np.float32)
        else:
            vecs = [ft_model.get_word_vector(tok) for tok in tokens]
            vec  = np.mean(vecs, axis=0)                       # [D]

        # ------------ DEBUG ---------------------------------
        if debug and printed < debug_max_print:
            print("\n" + "-" * 70)
            print(f"[fastText-DEBUG] TEXT        : {text}")
            print(f"[fastText-DEBUG] TOKENS ({len(tokens)}) : {tokens}")
            print(f"[fastText-DEBUG] VEC  L2-norm : {np.linalg.norm(vec):.4f}")
            print("-" * 70)
            printed += 1

        vec_list.append(torch.tensor(vec, dtype=torch.float32))

    return torch.stack(vec_list).unsqueeze(1)                  # [N, 1, D]



def extract_hidden_states(
        texts,
        args,
        tokenizer,
        model,
        device):
    """
    モデル種別を見て内部関数を呼び分けるラッパ
    """
    if args.model_type == 'hf':
        return _extract_hidden_states_hf(
            texts,
            tokenizer,
            model,
            device,
            max_length=args.max_length,
            debug=args.is_verbose
        )

    elif args.model_type == 'fasttext':
        return _extract_fasttext_embeddings(
            texts,
            model,
            debug=args.is_verbose
        )

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")


def save_embeddings(embeddings_tensor, metadata, output_path):
    """Save embeddings and metadata using torch.save"""
    data = {
        'embeddings': embeddings_tensor,
        'metadata': metadata,
        'embedding_shape': tuple(embeddings_tensor.shape),
        'torch_version': torch.__version__
    }
    
    # Use torch.save for efficient tensor storage
    torch.save(data, output_path)
    
    print(f"✓ Embeddings saved to {output_path}")
    print(f"  Tensor shape: {embeddings_tensor.shape}")
    print(f"  Tensor dtype: {embeddings_tensor.dtype}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")



def main():
    args = parse_args()

    # --- デバイス設定（fastText では未使用だが併せて出力） ---------------
    if args.device == 'auto':
        device = torch.device('cuda' if (torch.cuda.is_available() and args.model_type == 'hf') else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}  (model_type={args.model_type})")

    # --- fastText 用のエラーチェック -------------------------------------
    if args.model_type == 'fasttext':
        if args.text_mode != 'title':
            print("[ERROR] fastText では text_mode は 'title' のみ使用可能です")
            sys.exit(1)

    # --- 出力ディレクトリ -------------------------------------------------
    output_root = os.path.join(args.output_dir, args.text_mode)
    os.makedirs(output_root, exist_ok=True)
    print(f"Embeddings will be saved under: {output_root}")

    # --- データ読み込み ---------------------------------------------------
    tree_groups = load_data(args.data_path)

    # --- モデル読み込み ---------------------------------------------------
    tokenizer, model = load_model_and_tokenizer(args, device)

    # --- 各ツリーごとに処理 ----------------------------------------------
    for tree_id, tree_data in tree_groups.items():
        print(f"\n{'='*20} Processing Tree {tree_id} {'='*20}")

        tree_structure = build_tree_structure(tree_data, args.text_mode)
        distances, nodes = calculate_tree_distances(tree_structure)

        node_texts = [build_input_text(tree_structure['nodes'][n], args.text_mode) for n in nodes]

        embeddings = extract_hidden_states(
            node_texts,
            args=args,
            tokenizer=tokenizer,
            model=model,
            device=device
        )

        metadata = {
            'tree_id': tree_id,
            'nodes': nodes,
            'node_titles': [tree_structure['nodes'][n]['title'] for n in nodes],
            'input_texts': node_texts,
            'text_mode': args.text_mode,
            'tree_structure': tree_structure,
            'distances': distances,
            'model_type': args.model_type,
            'model_name': args.model_name if args.model_type == 'hf' else args.vector_path,
            'num_layers': embeddings.shape[1]
        }

        out_path = os.path.join(output_root, f'tree_{tree_id}_embeddings.pt')
        save_embeddings(embeddings, metadata, out_path)

    print(f"\n✓ All embeddings extracted and saved into {output_root}")

if __name__ == "__main__":
    main()





