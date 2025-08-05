#!/usr/bin/env python
# =============================================================================
# eval_tree.py - Tree Distance Evaluation
# =============================================================================
"""
Evaluate predicted trees against gold trees using Jaccard-Robinson-Foulds distance.

This script:
1. Reconstructs gold tree using only entity nodes (is_entity=True)
2. Builds predicted tree using the same approach as visualize_tree.py
3. Calculates Jaccard-Robinson-Foulds distance with k=1 and k=2
4. Provides debug visualizations of both trees
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
import pandas as pd
import numpy as np
from itertools import combinations

import util
from embeddings import EmbeddingConfig, EmbeddingModel
from hierarchy_node import HierarchyNode
from html_tree_encoding import HTMLTreeEncoding as TreeEncoding
from multibranch_tree_encoding import MultiBranchTreeEncoding
import template

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
        """Get all entity leaf nodes under this node."""
        if self.is_entity:
            return {self.name}
        
        entities = set()
        for child in self.children:
            entities.update(child.get_entity_leaves())
        return entities


def build_gold_tree(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str]]:
    """
    Build gold tree structure from JSONL data.
    
    Returns:
        - category to entities mapping
        - list of entity names (in consistent order) 
        - profession mapping
    """
    # Extract entity names and profession mapping
    entity_names = []
    profession_map = {}
    category_to_entities = {}
    
    for _, row in df.iterrows():
        name = row['wiki_title']
        is_entity = row['is_entity']
        
        if is_entity:
            entity_names.append(name)
            # Extract profession info
            if 'edges' in row and row['edges']:
                profession = row['edges'][0]['target_label']
                profession_map[name] = profession
                
                # Add to category mapping
                if profession not in category_to_entities:
                    category_to_entities[profession] = []
                category_to_entities[profession].append(name)
        else:
            # This is a category
            profession_map[name] = name
    
    entity_names.sort()  # Ensure consistent ordering
    
    return category_to_entities, entity_names, profession_map


def reconstruct_entity_tree_adjacency(df: pd.DataFrame, entity_names: List[str]) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """
    Reconstruct tree adjacency using actual hierarchical structure from the data.
    
    This creates a general tree where:
    - Leaf nodes are entities (is_entity=True, indices 0 to n-1)
    - Internal nodes are categories (is_entity=False)
    - Structure follows the actual parent-child relationships in the data
    - Works with arbitrary hierarchy depths and structures
    """
    entity_to_idx = {name: idx for idx, name in enumerate(entity_names)}
    n_entities = len(entity_names)
    
    # Build mappings and track relationships
    node_labels = {}
    node_name_to_id = {}
    next_internal_id = n_entities
    
    # Map entity names to their IDs and labels
    for idx, name in enumerate(entity_names):
        node_labels[idx] = name
        node_name_to_id[name] = idx
    
    # Create internal nodes for all categories
    category_nodes = df[df['is_entity'] == False]
    name_to_children = {}  # category_name -> list of child names
    
    # First pass: collect all category nodes and their children
    for _, row in category_nodes.iterrows():
        category_name = row['wiki_title']
        edges = row.get('edges', [])
        
        # Assign ID to this category if not already assigned
        if category_name not in node_name_to_id:
            node_name_to_id[category_name] = next_internal_id
            node_labels[next_internal_id] = category_name
            next_internal_id += 1
        
        # Collect children
        children_names = [edge['target_label'] for edge in edges]
        name_to_children[category_name] = children_names
    
    # Second pass: build adjacency relationships (avoid cycles)
    adjacency = {}
    
    for category_name, children_names in name_to_children.items():
        category_id = node_name_to_id[category_name]
        children_ids = []
        
        for child_name in children_names:
            if child_name in node_name_to_id:
                child_id = node_name_to_id[child_name]
                
                # Avoid adding parent as child (prevent cycles)
                # Don't add a node as child of its own descendant
                if not would_create_cycle(category_id, child_id, adjacency):
                    children_ids.append(child_id)
        
        if children_ids:
            adjacency[category_id] = children_ids
    
    return adjacency, node_labels


def would_create_cycle(parent_id: int, child_id: int, adjacency: Dict[int, List[int]]) -> bool:
    """Check if adding child_id as child of parent_id would create a cycle."""
    # If child_id is already an ancestor of parent_id, adding this edge would create a cycle
    visited = set()
    
    def is_ancestor(potential_ancestor: int, node: int) -> bool:
        if node in visited:
            return False
        visited.add(node)
        
        if potential_ancestor == node:
            return True
        
        if node in adjacency:
            for child in adjacency[node]:
                if is_ancestor(potential_ancestor, child):
                    return True
        return False
    
    return is_ancestor(parent_id, child_id)


def jaccard_robinson_foulds_distance(tree1_adj: Dict[int, List[int]], 
                                   tree2_adj: Dict[int, List[int]], 
                                   n_leaves: int, 
                                   k: int) -> float:
    """
    Calculate Jaccard-Robinson-Foulds distance between two trees.
    
    Args:
        tree1_adj: Adjacency list for tree 1
        tree2_adj: Adjacency list for tree 2
        n_leaves: Number of leaf nodes
        k: Parameter for JRF distance (1 or 2)
    
    Returns:
        JRF distance value
    """
    def get_all_splits(adj: Dict[int, List[int]], n_leaves: int) -> Set[frozenset]:
        """Get all splits (bipartitions) defined by internal nodes."""
        splits = set()
        visited = set()  # To avoid infinite recursion
        
        def get_leaves_under_node(node_id: int, path: Set[int] = None) -> Set[int]:
            """Get all leaf nodes under a given node with cycle detection."""
            if path is None:
                path = set()
            
            # Cycle detection
            if node_id in path:
                return set()
            
            if node_id < n_leaves:
                return {node_id}
            
            if node_id not in adj:
                return set()
            
            leaves = set()
            new_path = path | {node_id}
            
            for child in adj[node_id]:
                leaves.update(get_leaves_under_node(child, new_path))
            return leaves
        
        # For each internal node, create a split
        for node_id in adj:
            if node_id not in visited:
                leaves_under = get_leaves_under_node(node_id)
                visited.add(node_id)
                
                if len(leaves_under) > 1 and len(leaves_under) < n_leaves:
                    # Create bipartition
                    split = frozenset(leaves_under)
                    splits.add(split)
        
        return splits
    
    def get_k_subsets(splits: Set[frozenset], k: int) -> Set[frozenset]:
        """Get all k-subsets from splits."""
        k_subsets = set()
        for split in splits:
            if len(split) >= k:
                for subset in combinations(split, k):
                    k_subsets.add(frozenset(subset))
        return k_subsets
    
    # Get splits for both trees
    splits1 = get_all_splits(tree1_adj, n_leaves)
    splits2 = get_all_splits(tree2_adj, n_leaves)
    
    # Get k-subsets
    k_subsets1 = get_k_subsets(splits1, k)
    k_subsets2 = get_k_subsets(splits2, k)
    
    # Calculate Jaccard distance
    intersection = len(k_subsets1 & k_subsets2)
    union = len(k_subsets1 | k_subsets2)
    
    if union == 0:
        return 0.0
    
    jaccard_similarity = intersection / union
    jrf_distance = 1.0 - jaccard_similarity
    
    return jrf_distance


def build_predicted_tree(entity_names: List[str], 
                        model_type: str = "gpt2", 
                        method: str = "last_token",
                        layer: int = 0,
                        device: str = "cuda",
                        template_name: str = "entity_only") -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Build predicted tree using the same approach as visualize_tree.py.
    
    Returns:
        - adjacency dict
        - birth_time dict
    """
    # Apply template to entity names
    template_str = template.get_template(template_name)
    templated_texts = []
    for entity_name in entity_names:
        templated_text = template.apply_template(template_str, entity_name)
        templated_texts.append(templated_text)
    
    # Get embeddings
    cfg = EmbeddingConfig(
        model_type=model_type,
        method=method,
        layer=layer,
        device=device,
        verbose=False,
    )
    embedder = EmbeddingModel(cfg)
    embs = embedder.encode(templated_texts, entity_names)
    
    if embs.ndim == 3:  # (L, N, D)
        embs = embs[0]  # Take first layer
    
    # Build hierarchy
    hierarchy = HierarchyNode(embs)
    hierarchy.calculate_persistence()
    
    return hierarchy.h_nodes_adj, hierarchy.birth_time



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
    
    # Get colors
    node_colors = {}
    for idx, entity_name in enumerate(entity_names):
        profession = profession_map.get(entity_name, "Person")
        node_colors[idx] = PROFESSION_COLORS.get(profession, "#CCCCCC")
    
    # Create labels
    if is_gold_tree and gold_node_labels:
        labels = gold_node_labels
    else:
        labels = {idx: entity_name for idx, entity_name in enumerate(entity_names)}
    
    if is_gold_tree:
        # Use multi-branch tree encoder for gold tree
        tree_encoder = MultiBranchTreeEncoding(
            adjacency=adjacency,
            births=birth_time,
            n_leaves=n_leaves,
            n_nodes=max(adjacency.keys()) + 1 if adjacency else n_leaves,
            highlights=None,
            labels=labels,
            node_colors=node_colors,
            title=title,
            height_px=1000,
            width_pct=100,
            font_size=16,
            group_spacing_multiplier=group_spacing_multiplier,
            sibling_spacing_multiplier=sibling_spacing_multiplier
        )
    else:
        # Use binary tree encoder for predicted tree
        # Convert adjacency to binary format for TreeEncoding
        binary_adj = {}
        for parent, children in adjacency.items():
            if len(children) == 2:
                binary_adj[parent] = tuple(children)
            # Skip non-binary nodes (shouldn't happen for predicted trees)
        
        tree_encoder = TreeEncoding(
            adjacency=binary_adj,
            births=birth_time,
            n_leaves=n_leaves,
            n_nodes=max(adjacency.keys()) + 1 if adjacency else n_leaves,
            highlights=None,
            labels=labels,
            node_colors=node_colors,
            title=title,
            height_px=1000,
            width_pct=100,
            font_size=16
        )
    
    tree_encoder.draw(str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Evaluate tree distance between gold and predicted trees")
    parser.add_argument("--input", required=True, help="Input JSONL file (gold tree)")
    parser.add_argument("--output_dir", required=True, help="Output directory for results and visualizations")
    
    # Prediction model parameters
    parser.add_argument("--model", choices=["gpt2", "meta-llama/Meta-Llama-3-8B", "fasttext", "random_emb"], 
                       default="gpt2", help="Model for prediction")
    parser.add_argument("--method", choices=["average", "last_token"], default="last_token")
    parser.add_argument("--layer", type=int, default=0, help="Layer index for embeddings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--template", default="entity_only", help="Template for text generation")
    
    # Export options
    parser.add_argument("--export_visualizations", action="store_true", 
                       help="Export HTML visualizations of both trees")
    
    # Visualization spacing parameters
    parser.add_argument("--group_spacing_multiplier", type=float, default=10.0,
                       help="Multiplier for spacing between different category groups (default: 10.0)")
    parser.add_argument("--sibling_spacing_multiplier", type=float, default=0.8,
                       help="Multiplier for spacing between sibling nodes within same group (default: 0.8)")
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    df = pd.read_json(str(input_path), lines=True)
    
    print(f"Loaded {len(df)} records from {input_path}")
    
    # Build gold tree
    print("Building gold tree...")
    category_to_entities, entity_names, profession_map = build_gold_tree(df)
    gold_adjacency, gold_node_labels = reconstruct_entity_tree_adjacency(df, entity_names)
    
    print(f"Gold tree: {len(entity_names)} entities, {len(gold_adjacency)} internal nodes")
    
    # Build predicted tree
    print("Building predicted tree...")
    pred_adjacency, pred_birth_time = build_predicted_tree(
        entity_names=entity_names,
        model_type=args.model,
        method=args.method,
        layer=args.layer,
        device=args.device,
        template_name=args.template
    )
    
    print(f"Predicted tree: {len(entity_names)} entities, {len(pred_adjacency)} internal nodes")
    
    # Calculate JRF distances
    n_leaves = len(entity_names)
    
    jrf_k1 = jaccard_robinson_foulds_distance(gold_adjacency, pred_adjacency, n_leaves, k=1)
    jrf_k2 = jaccard_robinson_foulds_distance(gold_adjacency, pred_adjacency, n_leaves, k=2)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Dataset: {input_path.name}")
    print(f"Model: {args.model} (layer {args.layer})")
    print(f"Template: {args.template}")
    print(f"Number of entities: {n_leaves}")
    print(f"JRF Distance (k=1): {jrf_k1:.4f}")
    print(f"JRF Distance (k=2): {jrf_k2:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "dataset": str(input_path),
        "model": args.model,
        "layer": args.layer,
        "template": args.template,
        "n_entities": n_leaves,
        "jrf_k1": jrf_k1,
        "jrf_k2": jrf_k2,
        "gold_internal_nodes": len(gold_adjacency),
        "pred_internal_nodes": len(pred_adjacency)
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Export visualizations if requested
    if args.export_visualizations:
        print("\nExporting visualizations...")
        
        # Gold tree visualization
        # Create fake birth times for gold tree (based on tree depth)
        gold_birth_time = {}
        
        def assign_birth_times(node_id: int, depth: float = 0.0):
            gold_birth_time[node_id] = depth
            if node_id in gold_adjacency:
                for child in gold_adjacency[node_id]:
                    assign_birth_times(child, depth + 0.1)
        
        # Find root of gold tree (node that is not a child of any other node)
        all_children = set()
        for children in gold_adjacency.values():
            all_children.update(children)
        
        gold_root_id = None
        for node_id in gold_adjacency:
            if node_id not in all_children:
                gold_root_id = node_id
                break
        
        if gold_root_id is not None:
            assign_birth_times(gold_root_id)
        else:
            # If no clear root, assign birth times to all nodes
            for node_id in range(n_leaves):
                gold_birth_time[node_id] = 0.0
            for node_id in gold_adjacency:
                if node_id not in gold_birth_time:
                    gold_birth_time[node_id] = 0.1
        
        gold_viz_path = output_dir / "gold_tree.html"
        export_tree_visualization(
            gold_adjacency, gold_birth_time, entity_names, profession_map,
            gold_viz_path, f"Gold Tree - {input_path.name}", is_gold_tree=True,
            gold_node_labels=gold_node_labels,
            group_spacing_multiplier=args.group_spacing_multiplier,
            sibling_spacing_multiplier=args.sibling_spacing_multiplier
        )
        print(f"Gold tree visualization: {gold_viz_path}")
        
        # Predicted tree visualization
        pred_viz_path = output_dir / "predicted_tree.html"
        export_tree_visualization(
            pred_adjacency, pred_birth_time, entity_names, profession_map,
            pred_viz_path, f"Predicted Tree - {args.model} Layer {args.layer}", is_gold_tree=False
        )
        print(f"Predicted tree visualization: {pred_viz_path}")


if __name__ == "__main__":
    main()