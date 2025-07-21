#!/usr/bin/env python
# =============================================================================
# eval_tree.py
# =============================================================================
"""
Evaluate tree similarities using R TreeDist package.

Computes 6 tree distance metrics between gold tree (from JSONL) and 
predicted trees (from hierarchy algorithm):
1. Jaccard-Robinson-Foulds distance (k=1)
2. Jaccard-Robinson-Foulds distance (k=2)  
3. Matching split distance
4. Phylogenetic information distance
5. Clustering information distance
6. Path distance (Frobenius distance of shortest path matrices)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Dict, List, Tuple, Set
import pandas as pd
import numpy as np

import util
from embeddings import EmbeddingConfig, EmbeddingModel
from hierarchy_node import HierarchyNode
import summarization

# R interface
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    numpy2ri.activate()
    
    # Import R packages
    base = importr('base')
    utils = importr('utils')
    ape = importr('ape')
    treedist = importr('TreeDist')
    
    HAS_R = True
except ImportError:
    print("Warning: rpy2 not available. Install with: pip install rpy2")
    print("Also install R packages: install.packages(c('ape', 'TreeDist'))")
    HAS_R = False

# --------------------------------------------------------------------------- #
# Tree Structure Classes                                                      #
# --------------------------------------------------------------------------- #

class TreeNode:
    def __init__(self, name: str, is_leaf: bool = False):
        self.name = name
        self.is_leaf = is_leaf
        self.children: List[TreeNode] = []
        self.parent: TreeNode | None = None
    
    def add_child(self, child: 'TreeNode'):
        child.parent = self
        self.children.append(child)
    
    def get_leaves(self) -> List[str]:
        """Get all leaf names in this subtree"""
        if self.is_leaf:
            return [self.name]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate tree distances using TreeDist")
    p.add_argument("--input", required=True, help="Input JSONL file with gold tree")
    p.add_argument("--output", required=True, help="Output CSV file for results")

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
# Gold Tree Parsing                                                          #
# --------------------------------------------------------------------------- #

def parse_gold_tree(jsonl_path: str) -> TreeNode:
    """Parse the gold tree from JSONL format"""
    df = pd.read_json(jsonl_path, lines=True)
    
    # Build node lookup
    nodes = {}
    for _, row in df.iterrows():
        qid = row['qid']
        name = row['wiki_title']
        is_leaf = row['is_entity']
        nodes[qid] = TreeNode(name, is_leaf)
    
    # Build parent-child relationships
    root = None
    for _, row in df.iterrows():
        qid = row['qid']
        node = nodes[qid]
        
        if row['edges']:
            for edge in row['edges']:
                target_qid = edge['target_qid']
                if target_qid in nodes:
                    if edge['target_label'] == 'Person':
                        # This node is a child of Person (root)
                        if target_qid not in [nodes[qid].name for nodes in nodes.values()]:
                            # Person is the root
                            root = nodes[target_qid]
                        root.add_child(node)
                    else:
                        # Regular parent-child relationship
                        parent = nodes[target_qid]
                        parent.add_child(node)
        else:
            # Node with no edges might be root
            if root is None:
                root = node
    
    # Find actual root (node with no parent)
    if root is None:
        for node in nodes.values():
            if node.parent is None:
                root = node
                break
    
    return root

def build_gold_tree_from_jsonl(jsonl_path: str) -> TreeNode:
    """Build gold tree structure from JSONL file"""
    df = pd.read_json(jsonl_path, lines=True)
    
    # Create nodes
    nodes = {}
    for _, row in df.iterrows():
        qid = row['qid']
        name = row['wiki_title']
        is_leaf = row['is_entity']
        nodes[qid] = TreeNode(name, is_leaf)
    
    # Build hierarchy based on edges
    root_candidates = set(nodes.keys())
    
    for _, row in df.iterrows():
        qid = row['qid']
        node = nodes[qid]
        
        if 'edges' in row and row['edges']:
            for edge in row['edges']:
                target_qid = edge['target_qid']
                if target_qid in nodes:
                    parent = nodes[target_qid]
                    parent.add_child(node)
                    # Remove from root candidates
                    if qid in root_candidates:
                        root_candidates.remove(qid)
    
    # Root should be the remaining candidate
    if len(root_candidates) == 1:
        root_qid = list(root_candidates)[0]
        return nodes[root_qid]
    else:
        # Fallback: find Person node
        for qid, node in nodes.items():
            if node.name == "Person":
                return node
        # If still not found, return first node
        return list(nodes.values())[0]

# --------------------------------------------------------------------------- #
# Tree Format Conversion                                                     #
# --------------------------------------------------------------------------- #

def tree_to_newick(node: TreeNode, leaf_names: Set[str] = None) -> str:
    """Convert tree to Newick format for R"""
    if leaf_names is None:
        # First pass: collect all leaf names to ensure consistency
        leaf_names = set()
        def collect_leaves(n):
            if n.is_leaf:
                leaf_names.add(n.name)
            for child in n.children:
                collect_leaves(child)
        collect_leaves(node)
    
    def to_newick_recursive(n: TreeNode) -> str:
        if n.is_leaf:
            # Clean name for Newick format
            clean_name = n.name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(';', '').replace(':', '')
            return clean_name
        
        if not n.children:
            return n.name.replace(' ', '_')
        
        child_strings = [to_newick_recursive(child) for child in n.children]
        return f"({','.join(child_strings)})"
    
    newick = to_newick_recursive(node)
    if not newick.endswith(';'):
        newick += ';'
    return newick

def hierarchy_to_tree(adjacency: Dict[int, Tuple[int, int]], 
                     sentences: List[str], 
                     n_leaves: int) -> TreeNode:
    """Convert hierarchy adjacency to TreeNode structure"""
    
    # Create leaf nodes
    nodes = {}
    for i in range(n_leaves):
        clean_name = sentences[i].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(';', '').replace(':', '')
        nodes[i] = TreeNode(clean_name, is_leaf=True)
    
    # Create internal nodes and build tree bottom-up
    max_node = max(adjacency.keys()) if adjacency else n_leaves - 1
    
    for node_id in range(n_leaves, max_node + 1):
        nodes[node_id] = TreeNode(f"node_{node_id}", is_leaf=False)
    
    # Build parent-child relationships
    for parent_id, (right_id, left_id) in adjacency.items():
        parent = nodes[parent_id]
        if right_id in nodes:
            parent.add_child(nodes[right_id])
        if left_id in nodes:
            parent.add_child(nodes[left_id])
    
    # Find root (node with no parent)
    root = None
    for node in nodes.values():
        if node.parent is None and not node.is_leaf:
            root = node
            break
    
    return root

# --------------------------------------------------------------------------- #
# Distance Computation                                                       #
# --------------------------------------------------------------------------- #

def compute_tree_distances(gold_newick: str, pred_newick: str) -> Dict[str, float]:
    """Compute tree distances using R TreeDist package"""
    if not HAS_R:
        raise RuntimeError("rpy2 not available. Cannot compute tree distances.")
    
    results = {}
    
    try:
        # Read trees in R
        robjects.r(f'''
        gold_tree <- ape::read.tree(text="{gold_newick}")
        pred_tree <- ape::read.tree(text="{pred_newick}")
        ''')
        
        # Compute distances
        
        # 1. Jaccard-Robinson-Foulds (k=1)
        jrf_k1 = robjects.r('TreeDist::JaccardRobinsonFoulds(gold_tree, pred_tree, k=1)')[0]
        results['JRF_k1'] = float(jrf_k1)
        
        # 2. Jaccard-Robinson-Foulds (k=2)  
        jrf_k2 = robjects.r('TreeDist::JaccardRobinsonFoulds(gold_tree, pred_tree, k=2)')[0]
        results['JRF_k2'] = float(jrf_k2)
        
        # 3. Matching Split Distance
        msd = robjects.r('TreeDist::MatchingSplitDistance(gold_tree, pred_tree)')[0]
        results['MatchingSplit'] = float(msd)
        
        # 4. Phylogenetic Information Distance
        pid = robjects.r('TreeDist::PhylogeneticInfoDistance(gold_tree, pred_tree)')[0]
        results['PhylogeneticInfo'] = float(pid)
        
        # 5. Clustering Information Distance
        cid = robjects.r('TreeDist::ClusteringInfoDistance(gold_tree, pred_tree)')[0]
        results['ClusteringInfo'] = float(cid)
        
        # 6. Path Distance
        path_dist = robjects.r('TreeDist::PathDist(gold_tree, pred_tree)')[0]
        results['PathDistance'] = float(path_dist)
        
    except Exception as e:
        print(f"Error computing distances: {e}")
        # Return NaN for failed computations
        results = {
            'JRF_k1': np.nan,
            'JRF_k2': np.nan, 
            'MatchingSplit': np.nan,
            'PhylogeneticInfo': np.nan,
            'ClusteringInfo': np.nan,
            'PathDistance': np.nan
        }
    
    return results

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
    """Embed sentences and return (embeddings, target_layers)"""
    cfg = EmbeddingConfig(
        model_type=model_type,
        method=method,
        layer=layer,
        device=device,
    )
    embedder = EmbeddingModel(cfg)
    embs = embedder.encode(list(sentences))

    want_all = isinstance(layer, str) and layer.lower() == "all"

    if want_all:
        if embs.ndim == 3:  # (L,N,D)
            target_layers: Sequence[int] = range(embs.shape[0])
        elif embs.ndim == 2:  # (N,D)
            embs = embs[None]  # → (1,N,D)
            target_layers = [0]
        else:
            raise ValueError(f"Unexpected embedding shape {embs.shape}")
    else:
        if embs.ndim == 2:
            embs = embs[None]
        target_layers = [int(layer)]

    return embs, target_layers

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def run(args) -> None:
    if not HAS_R:
        print("Error: rpy2 not available. Please install rpy2 and R packages:")
        print("pip install rpy2")
        print("R: install.packages(c('ape', 'TreeDist'))")
        return

    # 1) Parse gold tree
    print("Parsing gold tree...")
    gold_tree = build_gold_tree_from_jsonl(args.input)
    gold_newick = tree_to_newick(gold_tree)
    print(f"Gold tree Newick: {gold_newick}")

    # 2) Load & preprocess sentences
    input_path = Path(args.input)
    df = pd.read_json(str(input_path), lines=True)
    df = df[df["is_entity"] == True]
    sentences = df["wiki_title"].fillna("").tolist()
    
    # Ensure sentence names match gold tree leaves
    gold_leaves = set(gold_tree.get_leaves())
    sentence_names = [s.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(';', '').replace(':', '') for s in sentences]
    
    print(f"Found {len(sentences)} entity sentences")
    print(f"Gold tree has {len(gold_leaves)} leaves")

    # 3) Embed sentences
    print("Computing embeddings...")
    all_embs, target_layers = _encode_sentences(
        sentences=sentences,
        model_type=args.model,
        method=args.method,
        layer=args.layer,
        device=args.device,
    )

    # 4) Evaluate each layer
    results = []
    
    for out_idx, l_idx in enumerate(target_layers):
        print(f"Evaluating layer {l_idx}...")
        
        embs = all_embs[out_idx]  # (N, D)

        # Build predicted hierarchy
        hierarchy = HierarchyNode(embs)
        hierarchy.calculate_persistence()
        adjacency = hierarchy.h_nodes_adj
        n_leaves = int(np.min(list(adjacency.keys())))
        n_nodes = int(np.max(list(adjacency.keys())) + 1)

        # Convert to tree structure
        pred_tree = hierarchy_to_tree(adjacency, sentence_names, n_leaves)
        pred_newick = tree_to_newick(pred_tree)
        
        print(f"Predicted tree Newick: {pred_newick}")

        # Compute distances
        distances = compute_tree_distances(gold_newick, pred_newick)
        
        # Store results
        result = {
            'model': args.model,
            'method': args.method,
            'layer': l_idx,
            'n_leaves': n_leaves,
            'n_nodes': n_nodes,
            **distances
        }
        results.append(result)
        
        print(f"Layer {l_idx} distances: {distances}")

    # 5) Save results
    results_df = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(results_df)

# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    print("Arguments:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    run(args)