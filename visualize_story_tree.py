#!/usr/bin/env python
# visualize_story_tree.py
# -----------------------------------------------
import argparse
from pathlib import Path
import numpy as np
import json
from IPython import embed

import util
from hierarchy_node import HierarchyNode
import summarization
from html_tree_encoding import HTMLTreeEncoding as TreeEncoding

# ★ NEW: 自作 Embedding ラッパ
from embeddings import EmbeddingConfig, EmbeddingModel


# ------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build StoryTree and export HTML")
    p.add_argument("--input", required=True, help="Input text file")
    p.add_argument("--output", default="tree.html", help="Output HTML file")
    p.add_argument("--scale", type=float, default=1.0)

    # ★ NEW: Embedding params ---------------------
    p.add_argument("--model", choices=["gpt2", "meta-llama/Meta-Llama-3-8B", "fasttext"],
                   default="gpt2")
    p.add_argument("--method", choices=["average", "last_token"],
                   default="last_token")
    p.add_argument("--layer", type=int, default=-1,
                   help="Transformer hidden layer index (0-based, -1 for last)")
    p.add_argument("--device", default="cuda")
    # --------------------------------------------
    return p


# ------------------------------------------------
def run(args):
    # 1) テキスト読み込み
    text = Path(args.input).read_text(encoding="utf8")

    # 2) 文・前処理
    par_sent_dict, sents, prep_sents = util.get_text_data(text, module="nltk")
    original_sents = sents

    # 3) EmbeddingModel 準備（1インスタンスを再利用）
    cfg = EmbeddingConfig(model_type=args.model,
                          method=args.method,
                          layer=args.layer,
                          device=args.device)
    embedder = EmbeddingModel(cfg)

    embs = embedder.encode([" ".join(toks) for toks in prep_sents])
    labels = {idx: sent for idx, sent in enumerate(sents)}

    # 7) StoryTree 構築
    hierarchy = HierarchyNode(embs)
    hierarchy.calculate_persistence()
    adjacency = hierarchy.h_nodes_adj
    n_leaves = np.min(list(adjacency.keys()))
    n_nodes = np.max(list(adjacency.keys())) + 1

    # 8) サマリ抽出
    trimming_summary, trimmed, important = summarization.get_hierarchy_summary_ids(embs)
    kcenter_summary = summarization.get_k_center_summary_ids(
        summary_length=len(trimming_summary),
        embs=embs,
    )

    # 9) 可視化
    # highlights = set(trimming_summary) | set(kcenter_summary) | set(important)
    highlights = None

    TE = TreeEncoding(                  
        adjacency=adjacency,
        births=hierarchy.birth_time,
        n_leaves=n_leaves,
        n_nodes=n_nodes,
        highlights=highlights,
        labels=labels            
    )
    TE.draw(args.output)    
    print(f"[DONE] wrote {args.output}")


# ------------------------------------------------
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    # 辞書として整形表示
    print("Parsed arguments:")
    print(json.dumps(vars(args), indent=2))

    run(args)                          
