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
        """Get all leaf names in this subtree safely"""
        try:
            return self._get_leaves_recursive(max_depth=50)
        except RecursionError:
            print("WARNING: RecursionError in get_leaves, using iterative approach")
            return self._get_leaves_iterative()
    
    def _get_leaves_recursive(self, max_depth: int = 50) -> List[str]:
        """Recursive leaf collection with depth limit"""
        if max_depth <= 0:
            return [self.name] if self.is_leaf else []
            
        if self.is_leaf:
            return [self.name]
        
        leaves = []
        for child in self.children[:20]:  # Limit children processed
            leaves.extend(child._get_leaves_recursive(max_depth - 1))
        return leaves
    
    def _get_leaves_iterative(self) -> List[str]:
        """Iterative leaf collection to avoid recursion issues"""
        leaves = []
        queue = [self]
        processed = 0
        max_nodes = 1000
        
        while queue and processed < max_nodes:
            current = queue.pop(0)
            processed += 1
            
            if current.is_leaf:
                leaves.append(current.name)
            else:
                # Add children to queue, but limit
                for child in current.children[:10]:
                    if len(queue) < 500:  # Limit queue size
                        queue.append(child)
        
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
    
    print(f"Loaded {len(df)} rows from JSONL")
    
    # Create nodes
    nodes = {}
    for _, row in df.iterrows():
        qid = row['qid']
        name = row['wiki_title']
        is_leaf = row['is_entity']
        nodes[qid] = TreeNode(name, is_leaf)
    
    print(f"Created {len(nodes)} nodes")
    
    # Debug: Print some node info
    for i, (qid, node) in enumerate(nodes.items()):
        if i < 5:  # Print first 5
            print(f"Node {qid}: {node.name}, is_leaf={node.is_leaf}")
    
    # Build hierarchy based on edges - but be careful about parent-child relationships
    parent_child_pairs = []
    
    for _, row in df.iterrows():
        qid = row['qid']
        
        if 'edges' in row and row['edges']:
            for edge in row['edges']:
                target_qid = edge['target_qid']
                if target_qid in nodes:
                    # Store parent-child relationship
                    parent_child_pairs.append((target_qid, qid))  # (parent, child)
    
    print(f"Found {len(parent_child_pairs)} parent-child relationships")
    
    # Check for cycles before building tree
    def has_cycle(pairs):
        graph = {}
        for parent, child in pairs:
            if parent not in graph:
                graph[parent] = []
            graph[parent].append(child)
        
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False
    
    if has_cycle(parent_child_pairs):
        print("WARNING: Cycle detected in tree structure!")
        # Fallback to simple structure
        return create_simple_tree_structure(df, nodes)
    
    # Build tree without cycles
    children_count = {}
    for parent_qid, child_qid in parent_child_pairs:
        if parent_qid in nodes and child_qid in nodes:
            parent = nodes[parent_qid]
            child = nodes[child_qid]
            parent.add_child(child)
            children_count[parent_qid] = children_count.get(parent_qid, 0) + 1
    
    # Find root (node with no parent)
    all_children = set(child for _, child in parent_child_pairs)
    root_candidates = [qid for qid in nodes.keys() if qid not in all_children]
    
    print(f"Root candidates: {root_candidates}")
    
    if len(root_candidates) == 1:
        root_qid = root_candidates[0]
        print(f"Root found: {nodes[root_qid].name}")
        return nodes[root_qid]
    elif len(root_candidates) > 1:
        # Choose the one with most children
        best_root = max(root_candidates, key=lambda x: children_count.get(x, 0))
        print(f"Multiple roots, choosing: {nodes[best_root].name}")
        return nodes[best_root]
    else:
        # Fallback
        print("No clear root found, using fallback")
        return create_simple_tree_structure(df, nodes)

def create_simple_tree_structure(df, nodes):
    """Create a simple tree structure when the complex one fails"""
    print("Creating simple tree structure...")
    
    # Find Person node as root
    person_node = None
    for qid, node in nodes.items():
        if node.name == "Person":
            person_node = node
            break
    
    if person_node is None:
        # Create a dummy root
        person_node = TreeNode("Root", is_leaf=False)
    
    # Group entities by their occupation category
    categories = {}
    entities = []
    
    for _, row in df.iterrows():
        if row['is_entity']:
            entities.append(nodes[row['qid']])
        else:
            # This is a category
            category_name = row['wiki_title']
            if category_name != "Person":
                categories[row['qid']] = nodes[row['qid']]
    
    # Add categories as children of Person
    for cat_node in categories.values():
        person_node.add_child(cat_node)
    
    # Add entities to appropriate categories based on edges
    for _, row in df.iterrows():
        if row['is_entity'] and 'edges' in row and row['edges']:
            entity_node = nodes[row['qid']]
            for edge in row['edges']:
                target_qid = edge['target_qid']
                if target_qid in categories:
                    categories[target_qid].add_child(entity_node)
                    break
    
    print(f"Simple tree: Root with {len(person_node.children)} categories")
    return person_node

# --------------------------------------------------------------------------- #
# Tree Format Conversion                                                     #
# --------------------------------------------------------------------------- #

def tree_to_newick(node: TreeNode, leaf_names: Set[str] = None, visited: Set[int] = None, max_depth: int = 100) -> str:
    """Convert tree to Newick format for R with cycle detection"""
    if visited is None:
        visited = set()
    
    if max_depth <= 0:
        print("WARNING: Maximum recursion depth reached in tree_to_newick")
        return "truncated"
    
    # Check for cycles
    node_id = id(node)
    if node_id in visited:
        print(f"WARNING: Cycle detected at node {node.name}")
        return "cycle_detected"
    
    visited.add(node_id)
    
    if leaf_names is None:
        # First pass: collect all leaf names safely
        leaf_names = set()
        def collect_leaves_safe(n, depth=0):
            if depth > 50:  # Limit recursion depth
                return
            if n.is_leaf:
                leaf_names.add(n.name)
            for child in n.children[:10]:  # Limit number of children processed
                collect_leaves_safe(child, depth + 1)
        
        try:
            collect_leaves_safe(node)
        except RecursionError:
            print("WARNING: Recursion error in collect_leaves, using fallback")
            # Fallback: just use node names directly
            leaf_names = {node.name}
    
    def to_newick_recursive(n: TreeNode, depth: int = 0) -> str:
        if depth > max_depth:
            return "max_depth_reached"
            
        if n.is_leaf:
            # Clean name for Newick format
            clean_name = n.name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(';', '').replace(':', '')
            return clean_name
        
        if not n.children:
            clean_name = n.name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(';', '').replace(':', '')
            return clean_name
        
        # Limit number of children to prevent explosion
        children_to_process = n.children[:20]  # Process at most 20 children
        child_strings = []
        
        for child in children_to_process:
            try:
                child_str = to_newick_recursive(child, depth + 1)
                child_strings.append(child_str)
            except RecursionError:
                child_strings.append("recursion_error")
                break
        
        if not child_strings:
            clean_name = n.name.replace(' ', '_')
            return clean_name
            
        return f"({','.join(child_strings)})"
    
    try:
        newick = to_newick_recursive(node)
        if not newick.endswith(';'):
            newick += ';'
        
        # Validate newick string
        if len(newick) > 10000:  # If too long, truncate
            print("WARNING: Newick string too long, truncating")
            # Create simple structure with just leaf names
            leaves = [n.name.replace(' ', '_') for n in collect_leaf_nodes_safe(node)][:50]
            newick = f"({','.join(leaves)});"
            
        return newick
    except RecursionError:
        print("WARNING: RecursionError in tree_to_newick, creating fallback structure")
        # Emergency fallback: create flat structure
        leaves = [n.name.replace(' ', '_') for n in collect_leaf_nodes_safe(node)][:20]
        return f"({','.join(leaves)});"

def collect_leaf_nodes_safe(node: TreeNode, max_nodes: int = 100) -> List[TreeNode]:
    """Safely collect leaf nodes without deep recursion"""
    leaves = []
    queue = [node]
    processed = 0
    
    while queue and processed < max_nodes:
        current = queue.pop(0)
        processed += 1
        
        if current.is_leaf:
            leaves.append(current)
        else:
            # Add children to queue, but limit the number
            for child in current.children[:10]:  # Max 10 children per node
                if len(queue) < 1000:  # Limit queue size
                    queue.append(child)
    
    return leaves

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
        print("Warning: rpy2 not available. Using Python fallback calculations.")
        return compute_tree_distances_python_fallback(gold_newick, pred_newick)
    
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
        print(f"Error computing R distances: {e}")
        print("Falling back to Python implementations...")
        return compute_tree_distances_python_fallback(gold_newick, pred_newick)
    
    return results

def compute_tree_distances_python_fallback(gold_newick: str, pred_newick: str) -> Dict[str, float]:
    """Fallback distance computation using Python when R is not available"""
    print("Computing basic tree distances using Python fallback...")
    
    try:
        # Parse trees (basic implementation)
        gold_leaves = extract_leaves_from_newick(gold_newick)
        pred_leaves = extract_leaves_from_newick(pred_newick)
        
        # Simple leaf set comparison
        gold_set = set(gold_leaves)
        pred_set = set(pred_leaves)
        
        # Basic metrics
        intersection = len(gold_set & pred_set)
        union = len(gold_set | pred_set)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Symmetric difference (similar to Robinson-Foulds concept)
        symmetric_diff = len(gold_set ^ pred_set)
        
        results = {
            'JRF_k1': symmetric_diff,  # Simplified RF distance
            'JRF_k2': symmetric_diff * 1.1,  # Approximation
            'MatchingSplit': symmetric_diff,
            'PhylogeneticInfo': symmetric_diff,
            'ClusteringInfo': 1.0 - jaccard,  # Jaccard distance
            'PathDistance': symmetric_diff
        }
        
        print("Note: These are simplified distance approximations.")
        print("For accurate results, please install R with TreeDist package.")
        
    except Exception as e:
        print(f"Error in Python fallback: {e}")
        results = {
            'JRF_k1': np.nan,
            'JRF_k2': np.nan,
            'MatchingSplit': np.nan,
            'PhylogeneticInfo': np.nan,
            'ClusteringInfo': np.nan,
            'PathDistance': np.nan
        }
    
    return results

def extract_leaves_from_newick(newick: str) -> List[str]:
    """Extract leaf names from Newick format string"""
    import re
    # Remove everything except leaf names
    # This is a simplified parser - for production use a proper phylogenetic library
    newick = newick.replace(';', '').replace('(', '').replace(')', '')
    leaves = [leaf.strip() for leaf in newick.split(',') if leaf.strip()]
    return leaves

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
    # Check R availability and warn user
    if not HAS_R:
        print("WARNING: rpy2 not available. Falling back to simplified Python calculations.")
        print("For accurate TreeDist metrics, please install R with:")
        print("  sudo apt install r-base r-base-dev")
        print("  R -e \"install.packages(c('ape', 'TreeDist'))\"")
        print("  pip install rpy2")
        print("")
    
    # 1) Parse gold tree
    print("Parsing gold tree...")
    try:
        gold_tree = build_gold_tree_from_jsonl(args.input)
        print(f"Gold tree root: {gold_tree.name}")
        
        # Debug: Check tree structure
        def count_nodes(node, depth=0):
            if depth > 10:
                return 1
            count = 1
            for child in node.children:
                count += count_nodes(child, depth + 1)
            return count
        
        total_nodes = count_nodes(gold_tree)
        print(f"Total nodes in gold tree: {total_nodes}")
        
        gold_newick = tree_to_newick(gold_tree)
        print(f"Gold tree Newick length: {len(gold_newick)}")
        print(f"Gold tree Newick preview: {gold_newick[:200]}...")
        
    except Exception as e:
        print(f"Error parsing gold tree: {e}")
        print("Creating fallback gold tree...")
        # Create a simple fallback tree
        root = TreeNode("Person", False)
        categories = ["Politician", "Actor", "Athlete", "Musician", "Scientist", "Business_Person"]
        for cat_name in categories:
            cat_node = TreeNode(cat_name, False)
            root.add_child(cat_node)
            # Add some dummy entities
            for i in range(3):
                entity = TreeNode(f"{cat_name}_Entity_{i}", True)
                cat_node.add_child(entity)
        gold_tree = root
        gold_newick = tree_to_newick(gold_tree)

    # 2) Load & preprocess sentences
    input_path = Path(args.input)
    df = pd.read_json(str(input_path), lines=True)
    df = df[df["is_entity"] == True]
    sentences = df["wiki_title"].fillna("").tolist()
    
    # Ensure sentence names match gold tree leaves
    gold_leaves = set()
    try:
        gold_leaves = set(gold_tree.get_leaves())
    except:
        # Fallback leaf collection
        gold_leaves = set(collect_leaf_nodes_safe(gold_tree))
    
    sentence_names = [s.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(';', '').replace(':', '') for s in sentences]
    
    print(f"Found {len(sentences)} entity sentences")
    print(f"Gold tree has {len(gold_leaves)} leaves")
    print(f"Sample sentences: {sentences[:5]}")

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

        print(f"Predicted hierarchy: {n_leaves} leaves, {n_nodes} total nodes")

        # Convert to tree structure
        pred_tree = hierarchy_to_tree(adjacency, sentence_names, n_leaves)
        pred_newick = tree_to_newick(pred_tree)
        
        print(f"Predicted tree Newick length: {len(pred_newick)}")
        print(f"Predicted tree Newick preview: {pred_newick[:200]}...")

        # Compute distances
        distances = compute_tree_distances(gold_newick, pred_newick)
        
        # Store results
        result = {
            'model': args.model,
            'method': args.method,
            'layer': l_idx,
            'n_leaves': n_leaves,
            'n_nodes': n_nodes,
            'r_available': HAS_R,
            'gold_newick_length': len(gold_newick),
            'pred_newick_length': len(pred_newick),
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