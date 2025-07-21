#!/usr/bin/env python
# =============================================================================
# visualize_tree.py
# =============================================================================
"""
Build Tree HTML visualizations.

Changes in this version
-----------------------
* When ``--layer all`` (default) is specified, sentence‑level embeddings for
  **every** Transformer layer are obtained in a **single** forward pass via
  ``EmbeddingModel(layer="all")``.  The resulting tensor of shape
  ``(L, N, D)`` is cached and re‑used for each layer so that the expensive
  Transformer forward computation runs **exactly once**.

  This removes redundant work that previously executed the model once **per
  layer**, leading to a ~``#layers`` speed‑up and identical results.

Output path rules
~~~~~~~~~~~~~~~~~
* ``--output`` **directory** →  ``{output_dir}/layer{N}/tree.html`` per layer
* ``--output`` **.html file** → generate that single file (legacy behaviour)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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
    p = argparse.ArgumentParser(description="Build Tree and export HTML")
    p.add_argument("--input", required=True, help="Input text file")
    p.add_argument(
        "--output",
        default="tree.html",
        help=(
            "出力ディレクトリまたはファイル名。"
            "ディレクトリを渡すと /layerX/tree.html 形式で保存する"
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

def build_output_path(base: Path, layer_idx: int | None) -> Path:
    """Generate an output ``.html`` file path for the given layer index."""
    if base.suffix.lower() == ".html":
        # User passed a file path directly → reuse as‑is
        return base

    # Directory was supplied → create per‑layer sub‑directory
    layer_dir = base / f"layer{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    return layer_dir / "tree.html"


# --------------------------------------------------------------------------- #
# Embedding utility                                                           #
# --------------------------------------------------------------------------- #

def _encode_sentences(
    sentences: Sequence[str],
    model_type: str,
    method: str,
    layer: str | int,
    device: str,
) -> tuple[np.ndarray, Sequence[int]]:
    """Embed *sentences* and return ``(embeddings, target_layers)``.

    * If *layer* == ``"all"`` **and** the backend provides per‑layer hidden
      states, the returned tensor is ``(L, N, D)`` with ``target_layers =
      range(L)``.
    * If the backend ignores *layer* (e.g. FastText) and yields ``(N, D)``, we
      wrap it to ``(1, N, D)`` so that downstream code still works and treat it
      as a single layer ``[0]``.
    * When *layer* is an int, the tensor is forced to shape ``(1, N, D)`` with
      ``target_layers = [layer]``.
    """
    cfg = EmbeddingConfig(
        model_type=model_type,
        method=method,
        layer=layer,
        device=device,
    )
    embedder = EmbeddingModel(cfg)
    embs = embedder.encode(list(sentences))  # shapes: (L,N,D) | (N,D)

    want_all = isinstance(layer, str) and layer.lower() == "all"

    if want_all:
        if embs.ndim == 3:  # (L,N,D) — true multi‑layer model
            target_layers: Sequence[int] = range(embs.shape[0])
        elif embs.ndim == 2:  # (N,D) — fallback: single layer only
            embs = embs[None]  # → (1,N,D)
            target_layers = [0]
        else:
            raise ValueError(
                f"Unexpected embedding shape {embs.shape}; expected (L,N,D) or (N,D)"
            )
    else:
        # Specific layer requested → always treat as single layer
        if embs.ndim == 2:
            embs = embs[None]
        target_layers = [int(layer)]

    return embs, target_layers


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def run(args) -> None:
    # 1) Load & preprocess ----------------------------------------------------
    text = Path(args.input).read_text(encoding="utf8")
    _, sents, prep_sents = util.get_text_data(text, module="nltk")
    sentences = [" ".join(toks) for toks in prep_sents]

    # 2) Embed sentences (single forward pass when possible) -----------------
    all_embs, target_layers = _encode_sentences(
        sentences=sentences,
        model_type=args.model,
        method=args.method,
        layer=args.layer,
        device=args.device,
    )

    base_out = Path(args.output)

    # 3) Build & export Tree per layer -----------------------------------
    for out_idx, l_idx in enumerate(target_layers):
        embs = all_embs[out_idx]  # (N, D) for the current layer

        # Build Tree hierarchy
        hierarchy = HierarchyNode(embs)
        hierarchy.calculate_persistence()
        adjacency = hierarchy.h_nodes_adj
        n_leaves = int(np.min(list(adjacency.keys())))
        n_nodes = int(np.max(list(adjacency.keys())) + 1)

        # Summary extraction (currently unused for highlights)
        trimming_summary, _, important = summarization.get_hierarchy_summary_ids(embs)
        kcenter_summary = summarization.get_k_center_summary_ids(
            summary_length=len(trimming_summary),
            embs=embs,
        )
        highlights = None  # set(trimming_summary) | set(kcenter_summary) | set(important)

        # Encode tree to HTML
        tree_encoder = TreeEncoding(
            adjacency=adjacency,
            births=hierarchy.birth_time,
            n_leaves=n_leaves,
            n_nodes=n_nodes,
            highlights=highlights,
            labels={idx: sent for idx, sent in enumerate(sents)},
        )

        # 4) Output ---------------------------------------------------------
        out_path = build_output_path(base_out, l_idx)
        tree_encoder.draw(str(out_path))
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
