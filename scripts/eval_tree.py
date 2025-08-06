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
4. Gold Treeの2分岐木変換モデルを2種類追加:
   - gold_binary_left: 左寄せアルゴリズム
   - gold_binary_balanced: バランス型アルゴリズム
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
# 3. Gold tree binary conversion (two methods)
# --------------------------------------------------------------------------- #
def convert_multibranch_to_binary_left(
        adjacency: Dict[int, List[int]], 
        n_leaves: int
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Convert a multibranch tree to a left-leaning binary tree.
    
    For nodes with >2 children, create intermediate nodes to binarize
    using left-leaning approach (sequential pairing).
    Example: parent -> [A, B, C, D] becomes:
             parent -> [internal1, D]
             internal1 -> [internal2, C]
             internal2 -> [A, B]
    
    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Original multibranch adjacency
    n_leaves : int
        Number of leaf nodes
    
    Returns
    -------
    binary_adj : Dict[int, List[int]]
        Binary tree adjacency
    birth_time : Dict[int, float]
        Birth times for nodes (synthetic)
    """
    binary_adj = {}
    next_internal_id = max(max(adjacency.keys(), default=n_leaves-1), 
                          max((c for cs in adjacency.values() for c in cs), default=n_leaves-1)) + 1
    
    # Process each parent node
    for parent, children in adjacency.items():
        if len(children) <= 2:
            # Already binary or unary
            binary_adj[parent] = children
        else:
            # Need to binarize: use left-leaning approach
            remaining = list(children)
            current_parent = parent
            
            while len(remaining) > 2:
                # Take first two children
                left = remaining.pop(0)
                right = remaining.pop(0)
                
                # Create new internal node
                new_internal = next_internal_id
                next_internal_id += 1
                
                # Connect current parent to new internal and remaining
                if len(remaining) == 1:
                    binary_adj[current_parent] = [new_internal, remaining[0]]
                    binary_adj[new_internal] = [left, right]
                    remaining = []
                else:
                    # More children to process
                    next_internal = next_internal_id
                    next_internal_id += 1
                    binary_adj[current_parent] = [new_internal - 1, next_internal]
                    binary_adj[new_internal - 1] = [left, right]
                    current_parent = next_internal
                    remaining.insert(0, remaining.pop())  # Move last to front for next iteration
            
            if len(remaining) == 2:
                binary_adj[current_parent] = remaining
    
    # Create synthetic birth times based on tree depth
    birth_time = {}
    
    def assign_birth_times(node: int, depth: float = 0.0, visited: Set[int] = None):
        if visited is None:
            visited = set()
        if node in visited:
            return
        visited.add(node)
        birth_time[node] = depth
        for child in binary_adj.get(node, []):
            assign_birth_times(child, depth + 0.1, visited)
    
    # Find roots
    all_children = {c for cs in binary_adj.values() for c in cs}
    roots = set(binary_adj.keys()) - all_children
    
    # Also check for leaf nodes that might be roots
    for i in range(n_leaves):
        if i not in all_children and i not in binary_adj:
            roots.add(i)
    
    for root in roots:
        assign_birth_times(root)
    
    # Ensure all leaf nodes have birth times
    for i in range(n_leaves):
        if i not in birth_time:
            birth_time[i] = 1.0  # Default depth for unconnected leaves
    
    return binary_adj, birth_time


def convert_multibranch_to_binary_balanced(
        adjacency: Dict[int, List[int]], 
        n_leaves: int
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Convert a multibranch tree to a balanced binary tree.
    
    For nodes with >2 children, create intermediate nodes to binarize
    using balanced splitting (divide children list in half recursively).
    Example: parent -> [A, B, C, D] becomes:
             parent -> [internal1, internal2]
             internal1 -> [A, B]
             internal2 -> [C, D]
    
    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Original multibranch adjacency
    n_leaves : int
        Number of leaf nodes
    
    Returns
    -------
    binary_adj : Dict[int, List[int]]
        Binary tree adjacency
    birth_time : Dict[int, float]
        Birth times for nodes (synthetic)
    """
    binary_adj = {}
    next_internal_id = max(max(adjacency.keys(), default=n_leaves-1), 
                          max((c for cs in adjacency.values() for c in cs), default=n_leaves-1)) + 1
    
    def binarize_children(children: List[int]) -> Tuple[int, Dict[int, List[int]]]:
        """
        Recursively binarize a list of children nodes.
        Returns the root of the binary subtree and updates to adjacency.
        """
        nonlocal next_internal_id
        local_adj = {}
        
        if len(children) == 0:
            return None, {}
        elif len(children) == 1:
            return children[0], {}
        elif len(children) == 2:
            # Create internal node for two children
            new_node = next_internal_id
            next_internal_id += 1
            local_adj[new_node] = children
            return new_node, local_adj
        else:
            # Split children list in half for balanced tree
            mid = len(children) // 2
            left_children = children[:mid]
            right_children = children[mid:]
            
            # Recursively binarize each half
            left_root, left_adj = binarize_children(left_children)
            right_root, right_adj = binarize_children(right_children)
            
            # Create parent node for the two subtrees
            new_node = next_internal_id
            next_internal_id += 1
            local_adj[new_node] = [left_root, right_root]
            
            # Merge adjacencies
            local_adj.update(left_adj)
            local_adj.update(right_adj)
            
            return new_node, local_adj
    
    # Process each parent node
    for parent, children in adjacency.items():
        if len(children) == 0:
            continue
        elif len(children) == 1:
            binary_adj[parent] = children
        elif len(children) == 2:
            binary_adj[parent] = children
        else:
            # Need to binarize: use balanced approach
            root, sub_adj = binarize_children(children)
            if root is not None:
                # Connect original parent to the root of binarized subtree
                if root in sub_adj:
                    # root is an internal node, connect parent to its children
                    binary_adj[parent] = sub_adj[root]
                    del sub_adj[root]
                else:
                    # root is a single child
                    binary_adj[parent] = [root]
                binary_adj.update(sub_adj)
    
    # Create synthetic birth times based on tree depth
    birth_time = {}
    
    def assign_birth_times(node: int, depth: float = 0.0, visited: Set[int] = None):
        if visited is None:
            visited = set()
        if node in visited:
            return
        visited.add(node)
        birth_time[node] = depth
        for child in binary_adj.get(node, []):
            assign_birth_times(child, depth + 0.1, visited)
    
    # Find roots
    all_children = {c for cs in binary_adj.values() for c in cs}
    roots = set(binary_adj.keys()) - all_children
    
    # Also check for leaf nodes that might be roots
    for i in range(n_leaves):
        if i not in all_children and i not in binary_adj:
            roots.add(i)
    
    for root in roots:
        assign_birth_times(root)
    
    # Ensure all leaf nodes have birth times
    for i in range(n_leaves):
        if i not in birth_time:
            birth_time[i] = 1.0  # Default depth for unconnected leaves
    
    return binary_adj, birth_time

# --------------------------------------------------------------------------- #
# 4. Predicted tree builder (modified to support gold_binary variants)
# --------------------------------------------------------------------------- #
def build_predicted_tree(
        entity_names: List[str],
        model_type: str = "gpt2",
        method: str = "last_token",
        layer: int = 0,
        device: str = "cuda",
        template_name: str = "entity_only",
        gold_adj: Dict[int, List[int]] = None,
        n_leaves: int = None
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Build predicted tree using hierarchical clustering persistence 
    or gold tree binary conversion.
    
    Parameters
    ----------
    entity_names : List[str]
        List of entity names
    model_type : str
        Model type (gpt2, llama, fasttext, random_emb, gold_binary_left, gold_binary_balanced)
    method : str
        Embedding method
    layer : int
        Layer number for transformer models
    device : str
        Device for computation
    template_name : str
        Template name for text generation
    gold_adj : Dict[int, List[int]]
        Gold tree adjacency (required for gold_binary variants)
    n_leaves : int
        Number of leaf nodes (required for gold_binary variants)
    
    Returns
    -------
    adjacency : Dict[int, List[int]]
        Predicted tree adjacency
    birth_time : Dict[int, float]
        Node birth times
    """
    if model_type == "gold_binary_left":
        if gold_adj is None or n_leaves is None:
            raise ValueError("gold_adj and n_leaves required for gold_binary_left model")
        return convert_multibranch_to_binary_left(gold_adj, n_leaves)
    elif model_type == "gold_binary_balanced":
        if gold_adj is None or n_leaves is None:
            raise ValueError("gold_adj and n_leaves required for gold_binary_balanced model")
        return convert_multibranch_to_binary_balanced(gold_adj, n_leaves)
    
    # Original implementation for other models
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
# 5. Visualization helpers（元コードから変更なし）
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
# 6. Main entry
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate tree distance between gold and predicted trees")
    ap.add_argument("--input", required=True, help="Input JSONL (gold tree)")
    ap.add_argument("--output_dir", required=True, help="Output dir")
    ap.add_argument("--model", default="gpt2",
                    choices=["gpt2", "meta-llama/Meta-Llama-3-8B",
                             "fasttext", "random_emb", 
                             "gold_binary_left", "gold_binary_balanced"])
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
    n_leaves = len(entity_names)

    print("Building predicted tree …")
    if args.model in ["gold_binary_left", "gold_binary_balanced"]:
        binary_method = "left-leaning" if args.model == "gold_binary_left" else "balanced"
        print(f"  Using gold tree binary conversion ({binary_method}) as predicted tree")
        pred_adj, pred_birth = build_predicted_tree(
            entity_names, args.model, args.method,
            args.layer, args.device, args.template,
            gold_adj=gold_adj, n_leaves=n_leaves)
    else:
        pred_adj, pred_birth = build_predicted_tree(
            entity_names, args.model, args.method,
            args.layer, args.device, args.template)

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
    
    if args.model in ["gold_binary_left", "gold_binary_balanced"]:
        print(f"\nGold tree structure:")
        print(f"  Original internal nodes : {len(gold_adj)}")
        print(f"  Binary internal nodes   : {len(pred_adj)}")
        binary_method = "left-leaning" if args.model == "gold_binary_left" else "balanced"
        print(f"  Binary method           : {binary_method}")

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

        if args.model in ["gold_binary_left", "gold_binary_balanced"]:
            # For gold_binary variants, visualize as binary tree
            binary_method = "Left-leaning" if args.model == "gold_binary_left" else "Balanced"
            export_tree_visualization(
                pred_adj, pred_birth, entity_names, prof_map,
                out_dir / "predicted_tree.html",
                f"Gold Tree ({binary_method} Binary Conversion)",
                is_gold_tree=False)  # Use binary tree visualization
        else:
            export_tree_visualization(
                pred_adj, pred_birth, entity_names, prof_map,
                out_dir / "predicted_tree.html",
                f"Predicted Tree – {args.model} L{args.layer}")

        print("Visualizations exported.")

if __name__ == "__main__":
    main()