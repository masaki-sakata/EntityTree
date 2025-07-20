#!/usr/bin/env python
# =============================================================================
# visualize_story_tree.py  
# =============================================================================
"""Build StoryTree HTML visualizations.

出力パスの規則:
    * --output に “ディレクトリ” を渡した場合
        <out_dir>/layer{N}/tree.html
    * --output に “.html” ファイルを渡した場合（従来互換）
        そのファイルひとつだけ生成
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import util
from embeddings import EmbeddingConfig, EmbeddingModel
from hierarchy_node import HierarchyNode
from html_tree_encoding import HTMLTreeEncoding as TreeEncoding
import summarization


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build StoryTree and export HTML")
    p.add_argument("--input", required=True, help="Input text file")
    p.add_argument(
        "--output",
        default="tree.html",
        help=(
            "出力ディレクトリまたはファイル名。"
            "ディレクトリを渡すと <out>/layerX/tree.html 形式で保存する"
        ),
    )
    # Embedding params --------------------------------------------------------
    p.add_argument(
        "--model",
        choices=["gpt2", "meta-llama/Meta-Llama-3-8B", "fasttext"],
        default="gpt2",
    )
    p.add_argument("--method", choices=["average", "last_token"], default="last_token")
    p.add_argument(
        "--layer",
        default="all",
        help='Transformer hidden layer index (0-based). Use "all" for every layer.',
    )
    p.add_argument("--device", default="cuda")
    return p


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def build_output_path(
    base: Path,
    layer_idx: int | None,
) -> Path:
    """出力先ファイルパスを生成するユーティリティ."""
    if base.suffix.lower() == ".html":
        # ユーザがファイル名を直接渡した場合
        return base

    # ディレクトリが渡された場合
    layer_dir = base / f"layer{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    fname = f"tree.html"
    return layer_dir / fname


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run(args):
    # 1) Load and preprocess text -------------------------------------------
    text = Path(args.input).read_text(encoding="utf8")
    _, sents, prep_sents = util.get_text_data(text, module="nltk")

    # 2) Decide target layers ----------------------------------------------
    if str(args.layer).lower() == "all":
        inspect_cfg = EmbeddingConfig(
            model_type=args.model,
            method=args.method,
            layer="all",
            device=args.device,
        )
        inspect_model = EmbeddingModel(inspect_cfg)
        target_layers = range(inspect_model.num_layers)
    else:
        target_layers = [int(args.layer)]

    base_out = Path(args.output)

    # 3) Loop over layers ---------------------------------------------------
    for l_idx in target_layers:
        cfg = EmbeddingConfig(
            model_type=args.model,
            method=args.method,
            layer=l_idx,
            device=args.device,
        )
        embedder = EmbeddingModel(cfg)
        embs = embedder.encode([" ".join(toks) for toks in prep_sents])

        labels = {idx: sent for idx, sent in enumerate(sents)}

        # Build StoryTree hierarchy
        hierarchy = HierarchyNode(embs)
        hierarchy.calculate_persistence()
        adjacency = hierarchy.h_nodes_adj
        n_leaves = np.min(list(adjacency.keys()))
        n_nodes = np.max(list(adjacency.keys())) + 1

        # Summary extraction (currently unused for highlights)
        trimming_summary, _, important = summarization.get_hierarchy_summary_ids(embs)
        kcenter_summary = summarization.get_k_center_summary_ids(
            summary_length=len(trimming_summary),
            embs=embs,
        )
        highlights = None  # set(trimming_summary) | set(kcenter_summary) | set(important)

        TE = TreeEncoding(
            adjacency=adjacency,
            births=hierarchy.birth_time,
            n_leaves=n_leaves,
            n_nodes=n_nodes,
            highlights=highlights,
            labels=labels,
        )

        # 4) Output ---------------------------------------------------------
        out_path = build_output_path(base_out, l_idx)
        TE.draw(str(out_path))
        print(f"[DONE] layer {l_idx}: wrote {out_path}")


# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    print("Arguments:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    run(args)
