#!/usr/bin/env python
# =============================================================================
# eval_tree.py - Tree Distance Evaluation (JRF-compliant)
# =============================================================================
"""
Evaluate predicted trees against gold trees using *true* Jaccard-Robinson-Foulds
distance (Robinson & Foulds 1981, Pompei et al. 2012).

変更点
-------
1. JRF 距離を split matching + Hungarian 法で正確に実装
2. k=1,2 の両方をサポート
3. サードパーティ依存: scipy>=1.9 が必要
   $ pip install scipy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
import pandas as pd
import numpy as np
from itertools import combinations

# --- New dependency ---------------------------------------------------------
from scipy.optimize import linear_sum_assignment  # <-- add

# --- Your project-specific imports (変更なし) -------------------------------
import util
from embeddings import EmbeddingConfig, EmbeddingModel
from hierarchy_node import HierarchyNode
from html_tree_encoding import HTMLTreeEncoding as TreeEncoding
from multibranch_tree_encoding import MultiBranchTreeEncoding
import template

from IPython import embed  # for debugging

# Optional dependencies for PNG export
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import time
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False


# Profession color mapping (same as visualize_tree.py)
PROFESSION_COLORS = {
    "Politician": "#322FEA",
    "Actor": "#FF9500",
    "Athlete": "#E93434",
    "Musician": "#32E352",
    "Scientist": "#00FBFF",
    "Business Person": "#E221E2",
    "Person": "#A0A0A0"
}

# --------------------------------------------------------------------------- #
# 1. Gold tree utilities（元コードから変更なし）
# --------------------------------------------------------------------------- #
class TreeNode:
    """Simple tree node for gold tree reconstruction."""
    def __init__(self, name: str, is_entity: bool = False):
        self.name = name
        self.is_entity = is_entity
        self.children: List[TreeNode] = []
        self.parent: Optional[TreeNode] = None

    def add_child(self, child: 'TreeNode'):
        child.parent = self
        self.children.append(child)

    def get_entity_leaves(self) -> Set[str]:
        if self.is_entity:
            return {self.name}
        entities = set()
        for child in self.children:
            entities.update(child.get_entity_leaves())
        return entities


def build_gold_tree(df: pd.DataFrame
                    ) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str]]:
    """Build gold tree structure from JSONL data."""
    entity_names: List[str] = []
    profession_map: Dict[str, str] = {}
    category_to_entities: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        name = row['wiki_title']
        is_entity = row['is_entity']

        if is_entity:
            entity_names.append(name)
            if 'edges' in row and row['edges']:
                profession = row['edges'][0]['target_label']
                profession_map[name] = profession
                category_to_entities.setdefault(profession, []).append(name)
        else:
            profession_map[name] = name

    entity_names.sort()
    return category_to_entities, entity_names, profession_map


def reconstruct_entity_tree_adjacency(
        df: pd.DataFrame, entity_names: List[str]
) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """Reconstruct gold tree adjacency preserving original hierarchy."""
    entity_to_idx = {n: i for i, n in enumerate(entity_names)}
    n_entities = len(entity_names)
    node_labels: Dict[int, str] = {}
    node_name_to_id: Dict[str, int] = {}
    next_internal_id = n_entities

    for idx, name in enumerate(entity_names):
        node_labels[idx] = name
        node_name_to_id[name] = idx

    category_nodes = df[df['is_entity'] == False]
    name_to_children: Dict[str, List[str]] = {}

    for _, row in category_nodes.iterrows():
        cat = row['wiki_title']
        edges = row.get('edges', [])
        if cat not in node_name_to_id:
            node_name_to_id[cat] = next_internal_id
            node_labels[next_internal_id] = cat
            next_internal_id += 1
        name_to_children[cat] = [e['target_label'] for e in edges]

    adjacency: Dict[int, List[int]] = {}
    def would_create_cycle(parent: int, child: int) -> bool:
        stack = [child]
        while stack:
            cur = stack.pop()
            if cur == parent:
                return True
            stack.extend(adjacency.get(cur, []))
        return False

    for cat, children in name_to_children.items():
        pid = node_name_to_id[cat]
        adjacency[pid] = []
        for child in children:
            if child in node_name_to_id:
                cid = node_name_to_id[child]
                if not would_create_cycle(pid, cid):
                    adjacency[pid].append(cid)
        if not adjacency[pid]:
            del adjacency[pid]  # remove empty internal node

    return adjacency, node_labels

# --------------------------------------------------------------------------- #
# 2. **True JRF distance implementation**
# --------------------------------------------------------------------------- #
def _split_jaccard(a: frozenset, b: frozenset, n_leaves: int) -> float:
    # 片側しか入っていないので補集合も試す
    comp_a = frozenset(range(n_leaves)) - a
    comp_b = frozenset(range(n_leaves)) - b
    j1 = len(a & b) / len(a | b)
    j2 = len(a & comp_b) / len(a | comp_b)
    # もう片方（comp_a と …）は左右入れ替えなので同じ値
    return max(j1, j2)

def _collect_splits(adj, n_leaves):
    splits = set()
    def leaves_under(node, visited):
        if node in visited: return set()
        visited.add(node)
        if node < n_leaves: return {node}
        leaves = set()
        for ch in adj.get(node, []):
            leaves |= leaves_under(ch, visited.copy())
        return leaves

    for parent, children in adj.items():
        for ch in children:                       # ← 子ごとに edge-split
            subset = leaves_under(ch, set())
            if 0 < len(subset) < n_leaves:
                # 小さい側を代表にして正規化
                rep = frozenset(subset if len(subset) <= n_leaves/2
                                else set(range(n_leaves)) - subset)
                splits.add(rep)
    return list(splits)



def jaccard_robinson_foulds_distance(
        tree1_adj: Dict[int, List[int]],
        tree2_adj: Dict[int, List[int]],
        n_leaves: int,
        k: int = 1
) -> float:
    """
    Compute JRF distance (Pompei et al. 2012) between two trees.

    Parameters
    ----------
    tree1_adj, tree2_adj : adjacency dicts
    n_leaves             : number of leaf nodes
    k                    : JRF parameter (typically 1 or 2)

    Returns
    -------
    distance : float   (0 = identical, 2·m = maximally different,
                        m = max(#splits1, #splits2))
    """
    splits1 = _collect_splits(tree1_adj, n_leaves)
    splits2 = _collect_splits(tree2_adj, n_leaves)

    m, n = len(splits1), len(splits2)
    size = max(m, n)                       # square matrix for Hungarian
    cost = np.ones((size, size))           # default cost 1 (no match)

    for i, s1 in enumerate(splits1):
        for j, s2 in enumerate(splits2):
            sim = _split_jaccard(s1, s2, n_leaves) ** k
            cost[i, j] = 1.0 - sim         # turn similarity into cost

    row_ind, col_ind = linear_sum_assignment(cost)
    total_sim = (1.0 - cost[row_ind, col_ind]).sum()

    max_pairs = size                       # = max(m, n)
    # JRF distance = 2 * (max_pairs − Σ similarity)
    return 2.0 * (max_pairs - total_sim)

# --------------------------------------------------------------------------- #
# 3. Predicted tree builder（変更なし）
# --------------------------------------------------------------------------- #
def build_predicted_tree(
        entity_names: List[str],
        model_type: str = "gpt2",
        method: str = "last_token",
        layer: int = 0,
        device: str = "cuda",
        template_name: str = "entity_only"
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """Build predicted tree using hierarchical clustering persistence."""
    template_str = template.get_template(template_name)
    texts = [template.apply_template(template_str, n) for n in entity_names]

    cfg = EmbeddingConfig(model_type=model_type,
                          method=method,
                          layer=layer,
                          device=device,
                          verbose=False)
    embs = EmbeddingModel(cfg).encode(texts, entity_names)
    if embs.ndim == 3:      # (L, N, D)
        embs = embs[0]

    hierarchy = HierarchyNode(embs)
    hierarchy.calculate_persistence()
    return hierarchy.h_nodes_adj, hierarchy.birth_time

# --------------------------------------------------------------------------- #
# 4. Visualization helpers（元コードから変更なし）
# --------------------------------------------------------------------------- #
def export_tree_visualization(adjacency: Dict[int, List[int]],
                              birth_time: Dict[int, float],
                              entity_names: List[str],
                              profession_map: Dict[str, str],
                              output_path: Path,
                              title: str,
                              is_gold_tree: bool = False,
                              gold_node_labels: Dict[int, str] = None,
                              group_spacing_multiplier: float = 10.0,
                              sibling_spacing_multiplier: float = 0.8):
    """Export tree visualization to HTML."""
    n_leaves = len(entity_names)
    node_colors = {
        i: PROFESSION_COLORS.get(profession_map.get(n, "Person"), "#CCCCCC")
        for i, n in enumerate(entity_names)
    }
    labels = (gold_node_labels if is_gold_tree and gold_node_labels
              else {i: n for i, n in enumerate(entity_names)})

    if is_gold_tree:
        enc = MultiBranchTreeEncoding(
            adjacency, birth_time, n_leaves,
            n_nodes=max(adjacency.keys()) + 1 if adjacency else n_leaves,
            highlights=None, labels=labels, node_colors=node_colors,
            title=title, height_px=1000, width_pct=100, font_size=16,
            group_spacing_multiplier=group_spacing_multiplier,
            sibling_spacing_multiplier=sibling_spacing_multiplier)
    else:
        bin_adj = {p: tuple(c) for p, c in adjacency.items() if len(c) == 2}
        enc = TreeEncoding(
            bin_adj, birth_time, n_leaves,
            n_nodes=max(adjacency.keys()) + 1 if adjacency else n_leaves,
            highlights=None, labels=labels, node_colors=node_colors,
            title=title, height_px=1000, width_pct=100, font_size=16)

    enc.draw(str(output_path))

# --------------------------------------------------------------------------- #
# 5. Main entry
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate tree distance between gold and predicted trees")
    ap.add_argument("--input", required=True, help="Input JSONL (gold tree)")
    ap.add_argument("--output_dir", required=True, help="Output dir")
    ap.add_argument("--model", default="gpt2",
                    choices=["gpt2", "meta-llama/Meta-Llama-3-8B",
                             "fasttext", "random_emb"])
    ap.add_argument("--method", default="last_token",
                    choices=["average", "last_token"])
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--template", default="entity_only")
    ap.add_argument("--export_visualizations", action="store_true")
    ap.add_argument("--group_spacing_multiplier", type=float, default=10.0)
    ap.add_argument("--sibling_spacing_multiplier", type=float, default=0.8)
    args = ap.parse_args()

    df = pd.read_json(args.input, lines=True)
    print(f"Loaded {len(df)} records")

    print("Building gold tree …")
    _, entity_names, prof_map = build_gold_tree(df)
    gold_adj, gold_labels = reconstruct_entity_tree_adjacency(df, entity_names)

    print("Building predicted tree …")
    pred_adj, pred_birth = build_predicted_tree(
        entity_names, args.model, args.method,
        args.layer, args.device, args.template)

    n_leaves = len(entity_names)
    jrf1 = jaccard_robinson_foulds_distance(gold_adj, pred_adj, n_leaves, k=1)
    jrf2 = jaccard_robinson_foulds_distance(gold_adj, pred_adj, n_leaves, k=2)
    # embed()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset                : {Path(args.input).name}")
    print(f"Model / Layer          : {args.model} / {args.layer}")
    print(f"Template               : {args.template}")
    print(f"Entities               : {n_leaves}")
    print(f"JRF Distance (k=1)     : {jrf1:.4f}")
    print(f"JRF Distance (k=2)     : {jrf2:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = dict(dataset=str(args.input), model=args.model, layer=args.layer,
                   template=args.template, n_entities=n_leaves,
                   jrf_k1=jrf1, jrf_k2=jrf2,
                   gold_internal_nodes=len(gold_adj),
                   pred_internal_nodes=len(pred_adj))
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir/'evaluation_results.json'}")

    if args.export_visualizations:
        print("\nExporting visualizations …")
        # Fake birth times for gold tree (depth-based)
        gold_birth: Dict[int, float] = {}
        def set_birth(n: int, d: float = 0.0):
            gold_birth[n] = d
            for ch in gold_adj.get(n, []):
                set_birth(ch, d + 0.1)
        roots = set(gold_adj) - {c for cs in gold_adj.values() for c in cs}
        for r in roots:
            set_birth(r)

        export_tree_visualization(
            gold_adj, gold_birth, entity_names, prof_map,
            out_dir / "gold_tree.html",
            f"Gold Tree – {Path(args.input).name}",
            is_gold_tree=True, gold_node_labels=gold_labels,
            group_spacing_multiplier=args.group_spacing_multiplier,
            sibling_spacing_multiplier=args.sibling_spacing_multiplier)

        export_tree_visualization(
            pred_adj, pred_birth, entity_names, prof_map,
            out_dir / "predicted_tree.html",
            f"Predicted Tree – {args.model} L{args.layer}")

        print("Visualizations exported.")

if __name__ == "__main__":
    main()
