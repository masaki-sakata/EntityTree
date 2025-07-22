#!/usr/bin/env python3
"""
Extract last-token hidden states of every node-title that appears in a
natural-language “in-context learning (ICL)” tree description.

2025-06-16  NL-prompt 版（重複除去対応, metadata 拡充版）
  • root 文   : «X is a broad category.»
  • child 文  : «Y is a kind of X.»
  • センテンス内の重複（同じ parent-child 組やノードの再訪）を排除
  • --is_verbose でタイトル＋token を表示
  • 位置が合わなければ RuntimeError
  • 出力 .pt に extract_embeddings.py と互換の metadata を格納
"""

import argparse, json, os, sys
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from IPython import embed


# --------------------------------------------------------------------------- #
# 引数処理
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',  required=True)
    p.add_argument('--output_dir', required=True)

    p.add_argument('--model_name', default='meta-llama/Meta-Llama-3-8B')
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--device', default='auto')

    p.add_argument('--text_mode', default='title')
    p.add_argument('--num_repeats', type=int, default=10,
                   help='How many times the same tree is repeated (k)')
    p.add_argument('--is_verbose', action='store_true')
    return p.parse_args()


# --------------------------------------------------------------------------- #
# データ読み込み
# --------------------------------------------------------------------------- #
def load_jsonl(path: str):
    data = []
    with open(path, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            if not (line := line.strip()):
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f'[ERROR] line {ln}: {e}')
                sys.exit(1)
    groups = defaultdict(list)
    for d in data:
        groups[d['tree_id']].append(d)
    print(f'Loaded {len(data)} nodes in {len(groups)} trees')
    return groups


# --------------------------------------------------------------------------- #
# 木構造構築
# --------------------------------------------------------------------------- #
def build_tree_structure(tree_data):
    """
    Return
    -------
    nodes      : {qid: {"title": str}}
    child_map  : {parent_qid: [child_qid, ...]}    (重複なし, ソート済み)
    root_qid   : single root id
    """
    nodes: Dict[str, dict] = {}
    child_map: Dict[str, set] = defaultdict(set)   # set で重複禁止
    all_children = set()

    for item in tree_data:
        qid = item['qid']
        nodes[qid] = {'title': item['wiki_title']}

        for e in item['edges']:
            prop = e['property']
            tgt  = e['target_qid']

            if prop == 'P527':              # parent (=item) → child (=tgt)
                child_map[qid].add(tgt)
                all_children.add(tgt)

            elif prop == 'P361':            # child (=item) → parent (=tgt)
                child_map[tgt].add(qid)
                all_children.add(qid)

    roots = [q for q in nodes if q not in all_children]
    if len(roots) != 1:
        raise ValueError(f'Cannot determine unique root: {roots}')

    # set → ソート済み list へ
    child_map_sorted = {k: sorted(v) for k, v in child_map.items()}
    return nodes, child_map_sorted, roots[0]


# --------------------------------------------------------------------------- #
# 正規化器
# --------------------------------------------------------------------------- #
def get_normalizer(tokenizer) -> Callable[[str], str]:
    """
    Return a function norm(text) -> str
    • fast SentencePiece tokenizer : use internal normalizer
    • slow tokenizer               : identity function
    """
    btok = getattr(tokenizer, "backend_tokenizer", None)
    if btok is not None and getattr(btok, "normalizer", None) is not None:
        return btok.normalizer.normalize_str
    else:
        print("  [INFO] slow tokenizer detected → identity normalizer")
        return lambda x: x


# --------------------------------------------------------------------------- #
# NL テキスト生成（重複除去）
# --------------------------------------------------------------------------- #
def make_nl_text_and_char_end(nodes, child_map, root_qid,
                              normalizer: Callable[[str], str]
                              ) -> Tuple[str, Dict[str, int], str]:
    """
    Build natural-language description for one tree, without internal duplicates.

    Returns
    -------
    nl_text_norm : full text (normalized)
    char_end_norm: {qid: char_index_after_last_char_of_title (normalized)}
    nl_text_raw  : non-normalized raw text (for metadata / prompt_raw)
    """
    norm = normalizer
    parts: List[str] = []
    char_end_norm: Dict[str, int] = {}

    nlen = lambda s: len(norm(s))

    # ---------- root ----------
    root_title = nodes[root_qid]['title']
    root_sent  = f'{root_title} is a broad category.\n'
    parts.append(root_sent)
    char_end_norm[root_qid] = nlen(root_title)

    # ---------- children ----------
    stack = [root_qid]          # DFS
    visited_nodes = {root_qid}  # ノード見たかどうか
    visited_edges = set()       # (parent, child) 重複防止

    while stack:
        parent = stack.pop()
        for child in child_map.get(parent, []):
            if (parent, child) in visited_edges:
                continue
            visited_edges.add((parent, child))

            child_title  = nodes[child]['title']
            parent_title = nodes[parent]['title']
            sent = f'{child_title} is a kind of {parent_title}.\n'
            parts.append(sent)

            # char_end = これまでの prefix の長さ + child_title の長さ
            prefix_raw = ''.join(parts[:-1])
            char_end_norm[child] = len(norm(prefix_raw)) + nlen(child_title)

            if child not in visited_nodes:
                visited_nodes.add(child)
                stack.append(child)

    nl_raw  = ''.join(parts)
    nl_norm = norm(nl_raw)
    return nl_norm, char_end_norm, nl_raw


# --------------------------------------------------------------------------- #
# プロンプト組み立て
# --------------------------------------------------------------------------- #
def prepare_prompt_and_lookup(text_norm: str,
                              text_raw : str,
                              char_end_norm: Dict[str, int],
                              repeats: int,
                              normalizer: Callable[[str], str]):
    sep_norm = normalizer('\n')
    unit_norm = text_norm + sep_norm
    unit_len_norm = len(unit_norm)

    unit_raw     = text_raw + '\n'         # 非正規化
    lookup = {}
    for r in range(1, repeats + 1):
        offset = (r - 1) * unit_len_norm
        for qid, end in char_end_norm.items():
            lookup[(r, qid)] = offset + end

    prompt_norm = unit_norm * repeats
    prompt_raw  = unit_raw  * repeats
    return prompt_norm, prompt_raw, lookup


# --------------------------------------------------------------------------- #
# モデル読み込み
# --------------------------------------------------------------------------- #
def load_model(model_name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(model_name,
                                    torch_dtype=torch.bfloat16,
                                    output_hidden_states=True)
    mdl.to(device).eval()
    return tok, mdl


# --------------------------------------------------------------------------- #
# 埋め込み抽出
# --------------------------------------------------------------------------- #
def extract_embeddings(prompt: str,
                       lookup: dict,
                       tokenizer, model,
                       device, max_length,
                       qid2title,
                       verbose=False):
    enc = tokenizer(prompt,
                    return_tensors='pt',
                    return_offsets_mapping=True,
                    padding=False,
                    truncation=False)
    if enc['input_ids'].shape[1] > max_length:
        print(f"{prompt=}")
        raise ValueError(f'Prompt too long: {enc["input_ids"].shape[1]} tokens')

    offsets = enc.pop('offset_mapping')[0].tolist()      # [(char_s, char_e), ...]
    char2tok = {e: i for i, (s, e) in enumerate(offsets) if e != 0}

    input_ids = enc['input_ids'][0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
    hidden = out.hidden_states
    num_layers = len(hidden)
    
    rep_embeddings = defaultdict(dict)
    missing = []
    buf = []

    for (rep, qid), char_end in lookup.items():
        tok_idx = char2tok.get(char_end)

        # fallback: look back a few characters
        if tok_idx is None:
            for off in range(1, 9):
                tok_idx = char2tok.get(char_end - off)
                if tok_idx is not None:
                    break

        if tok_idx is None:
            missing.append((rep, qid))
            continue

        vec = torch.stack([h[0, tok_idx].cpu() for h in hidden])   # [L,D]
        rep_embeddings[rep][qid] = vec

        if verbose and len(buf) < 30:
            tok_str = tokenizer.convert_ids_to_tokens(input_ids[tok_idx],
                                                      skip_special_tokens=False)
            buf.append(f'rep={rep:2d}  "{qid2title[qid]}"  '
                       f'tok={tok_idx:4d}  {tok_str}')

    if verbose and buf:
        print('  [VERBOSE] sampled tokens:')
        for l in buf:
            print('   ', l)

    if missing:
        sample = ', '.join(qid2title[q] for _, q in missing[:5])
        raise RuntimeError(f'{len(missing)} nodes could not be aligned '
                           f'(e.g., {sample})')

    return rep_embeddings, num_layers


# --------------------------------------------------------------------------- #
# 距離行列（Floyd-Warshall）
# --------------------------------------------------------------------------- #
def calc_tree_distances(nodes_order: List[str],
                        child_map: Dict[str, List[str]]):
    """
    Treat the (directed) tree as an undirected graph and compute
    all-pairs shortest paths with Floyd-Warshall.

    Returns
    -------
    np.ndarray shape=(N,N)  : distances (np.inf で到達不能)
    nodes_order            : echo back
    """
    n = len(nodes_order)
    idx = {q: i for i, q in enumerate(nodes_order)}

    dist = np.full((n, n), np.inf, dtype=np.float32)
    for i in range(n):
        dist[i, i] = 0.0

    for p, childs in child_map.items():
        for c in childs:
            if p in idx and c in idx:
                i, j = idx[p], idx[c]
                dist[i, j] = dist[j, i] = 1.0

    # Floyd-Warshall
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])

    return dist, nodes_order


# --------------------------------------------------------------------------- #
# 保存
# --------------------------------------------------------------------------- #
def save_embeddings(rep_embeddings, num_layers,
                    tree_id, nodes_order, node_titles,
                    prompt_raw, distances,
                    output_root, model_name,
                    text_mode, repeat_idx):
    outdir = os.path.join(output_root,
                          text_mode,
                          f'tree_{tree_id}',
                          f'repeat_{repeat_idx}')
    os.makedirs(outdir, exist_ok=True)

    embeds = [rep_embeddings[qid] for qid in nodes_order]
    tensor = torch.stack(embeds)                           # [N,L,D]

    meta = {
        # --- 共通情報 --------------------------------------------------
        'tree_id'     : tree_id,
        'repeat_idx'  : repeat_idx,
        'text_mode'   : text_mode,
        'model_type'  : 'hf',
        'model_name'  : model_name,

        # --- ノード ---------------------------------------------------
        'nodes'       : nodes_order,          # qid
        'node_titles' : node_titles,          # human readable

        # --- プロンプト -----------------------------------------------
        'input_prompt': prompt_raw,
        'num_repeats' : repeat_idx,

        # --- モデル ---------------------------------------------------
        'num_layers'  : num_layers,

        # --- 木構造 ---------------------------------------------------
        'distances'   : distances,            # numpy.ndarray

        # --- 追加メタ --------------------------------------------------
        'embedding_shape': tuple(tensor.shape),
        'torch_version'  : torch.__version__
    }
    path = os.path.join(outdir, f'tree_{tree_id}_embeddings.pt')
    torch.save({'embeddings': tensor, 'metadata': meta}, path)
    print(f'  saved → {path}  shape={tuple(tensor.shape)}')


# --------------------------------------------------------------------------- #
# メイン
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    device = (torch.device('cuda') if (args.device == 'auto' and
                                       torch.cuda.is_available())
              else torch.device(args.device if args.device != 'auto'
                                else 'cpu'))
    print(f'Using device : {device}')

    trees = load_jsonl(args.data_path)

    tokenizer, model = load_model(args.model_name, device)
    normalizer = get_normalizer(tokenizer)
    output_root = args.output_dir

    for tree_id, tree_data in trees.items():
        print(f'\n======= Tree {tree_id} =======')
        nodes, child_map, root_qid = build_tree_structure(tree_data)

        # 1. 自然言語テキスト（重複なし）
        text_norm, char_end_norm, text_raw = make_nl_text_and_char_end(
            nodes, child_map, root_qid, normalizer
        )

        # 2. プロンプト全体と lookup
        prompt_norm, prompt_raw, lookup = prepare_prompt_and_lookup(
            text_norm, text_raw, char_end_norm, args.num_repeats, normalizer
        )

        # embed()

        qid2title = {qid: info['title'] for qid, info in nodes.items()}

        # 3. 埋め込み抽出
        rep_embeds, num_layers = extract_embeddings(
            prompt_norm, lookup, tokenizer, model,
            device, args.max_length,
            qid2title=qid2title,
            verbose=args.is_verbose
        )

        nodes_order = list(nodes.keys())
        node_titles = [nodes[q]['title'] for q in nodes_order]

        # 4. 距離行列
        distances, _ = calc_tree_distances(nodes_order, child_map)

        # 5. 保存（リピートごとにフォルダを分ける）
        for r in range(1, args.num_repeats + 1):
            save_embeddings(rep_embeds[r], num_layers,
                            tree_id,
                            nodes_order,
                            node_titles,
                            prompt_raw,          # full raw prompt
                            distances,
                            output_root, args.model_name,
                            args.text_mode, r)

    print('\n✓ finished')


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    main()