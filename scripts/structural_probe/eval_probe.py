#!/usr/bin/env python3
"""
Evaluate a structural probe for hierarchical tree data (entity-level with centroid representations)
===========================================================================================

This script evaluates probes on taxonomy data where category nodes are represented
as centroids (averages) of their member entities' embeddings.

NEW: Uses separate JSONL gold data files for evaluation metrics.
NEW: Implements proper UUAS, DSpr, Root acc, and NSpr evaluation metrics.

Usage example:
    python eval_probe.py \
        --embedding_dir embeddings/ \
        --gold_data_path gold_data.jsonl \
        --layer 12 \
        --probe_type distance \
        --probe_rank 128 \
        --probe_path /path/to/probe_distance_layer12_rank128.pt \
        --vis_path probe_results/vis_layer12.html
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm
import unicodedata
import re


# ================================================================
# Probe definitions
# ================================================================

class TwoWordPSDProbe(nn.Module):
    """Squared L2 distance after low-rank projection"""
    def __init__(self, model_dim: int, probe_rank: int, device, dtype=torch.float32):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(model_dim, probe_rank, dtype=dtype))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.proj.dtype)
        x = torch.matmul(x, self.proj)                   # (B, N, R)
        x1 = x.unsqueeze(2)                              # (B, N, 1, R)
        dist = (x1 - x1.transpose(1, 2)).pow(2).sum(-1)  # (B, N, N)
        return dist


class OneWordPSDProbe(nn.Module):
    """Squared L2 norm after projection -> depth"""
    def __init__(self, model_dim: int, probe_rank: int, device, dtype=torch.float32):
        super().__init__()
        self.proj = nn.Parameter(torch.zeros(model_dim, probe_rank, dtype=dtype))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.proj.dtype)
        transformed = torch.matmul(batch, self.proj)  # (B, N, R)
        B, N, R = transformed.shape
        norms = torch.bmm(
            transformed.view(B*N, 1, R),
            transformed.view(B*N, R, 1)
        ).view(B, N)
        return norms  # (B, N)


# ================================================================
# Gold Data Loading and Processing
# ================================================================

def load_gold_structures(jsonl_path: str) -> Dict[int, Dict]:
    """
    Load gold hierarchical structures from JSONL file.
    
    Returns:
        Dict mapping tree_id to structure info:
        {
            'nodes': {qid: {'title': str, 'is_entity': bool}},
            'edges': [(parent_qid, child_qid), ...],
            'entity_nodes': [qid, ...],
            'category_nodes': [qid, ...],
            'root_node': qid
        }
    """
    print(f"Loading gold structures from {jsonl_path}")
    
    tree_structures = defaultdict(lambda: {
        'nodes': {},
        'edges': [],
        'entity_nodes': [],
        'category_nodes': [],
        'root_node': None
    })
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSONL"):
            data = json.loads(line.strip())
            tree_id = data['tree_id']
            qid = data['qid']
            title = data['wiki_title']
            is_entity = data['is_entity']
            
            # Store node info
            tree_structures[tree_id]['nodes'][qid] = {
                'title': title,
                'is_entity': is_entity
            }
            
            # Categorize nodes
            if is_entity:
                tree_structures[tree_id]['entity_nodes'].append(qid)
            else:
                tree_structures[tree_id]['category_nodes'].append(qid)
            
            # Store edges (parent -> child relationships)
            for edge in data.get('edges', []):
                parent_qid = qid  # Current node is parent
                child_qid = edge['target_qid']
                tree_structures[tree_id]['edges'].append((parent_qid, child_qid))
    
    # Remove cycles and build proper hierarchy
    for tree_id in tree_structures:
        tree_structures[tree_id] = build_acyclic_hierarchy(tree_structures[tree_id])
    
    print(f"Loaded {len(tree_structures)} tree structures")
    for tree_id in list(tree_structures.keys())[:3]:  # Show first 3 trees
        ts = tree_structures[tree_id]
        print(f"  Tree {tree_id}: {len(ts['nodes'])} nodes, {len(ts['edges'])} edges, "
              f"{len(ts['entity_nodes'])} entities, root={ts['root_node']}")
    
    return dict(tree_structures)


def build_acyclic_hierarchy(tree_structure: Dict) -> Dict:
    """
    Build an acyclic hierarchy by removing cycles and determining proper parent-child relationships.
    Uses entity count heuristic to resolve cycles between categories.
    """
    nodes = tree_structure['nodes']
    edges = tree_structure['edges']
    entity_nodes = set(tree_structure['entity_nodes'])
    category_nodes = set(tree_structure['category_nodes'])
    
    print(f"\n  Building hierarchy for tree with {len(nodes)} nodes")
    
    # Build adjacency lists
    adj = defaultdict(set)
    reverse_adj = defaultdict(set)
    for parent, child in edges:
        adj[parent].add(child)
        reverse_adj[child].add(parent)
    
    # Count direct entity children for each category
    entity_count = {}
    for cat in category_nodes:
        count = sum(1 for child in adj[cat] if child in entity_nodes)
        entity_count[cat] = count
        print(f"    Category {nodes[cat]['title']} ({cat}) has {count} direct entities")
    
    # Detect and resolve cycles between categories
    valid_edges = []
    removed_edges = []
    
    # First, keep all category->entity edges (these don't create cycles)
    for parent, child in edges:
        if parent in category_nodes and child in entity_nodes:
            valid_edges.append((parent, child))
    
    # Then handle category->category edges
    category_edges = [(p, c) for p, c in edges 
                      if p in category_nodes and c in category_nodes]
    
    # Build graph of only category nodes to detect cycles
    cat_graph = defaultdict(set)
    for parent, child in category_edges:
        cat_graph[parent].add(child)
    
    # Function to detect if adding an edge creates a cycle
    def creates_cycle(graph, parent, child):
        """Check if adding parent->child creates a cycle using DFS"""
        visited = set()
        stack = [child]
        while stack:
            node = stack.pop()
            if node == parent:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(graph.get(node, []))
        return False
    
    # Process category edges, checking for cycles
    clean_cat_graph = defaultdict(set)
    for parent, child in category_edges:
        if creates_cycle(clean_cat_graph, parent, child):
            # Cycle detected! Use entity count heuristic
            parent_entities = entity_count.get(parent, 0)
            child_entities = entity_count.get(child, 0)
            
            print(f"    Cycle detected: {nodes[parent]['title']} ({parent_entities} entities) <-> "
                  f"{nodes[child]['title']} ({child_entities} entities)")
            
            # Keep edge only if parent has more entities (or equal, to break ties consistently)
            if parent_entities >= child_entities:
                clean_cat_graph[parent].add(child)
                valid_edges.append((parent, child))
                print(f"      Keeping edge: {nodes[parent]['title']} -> {nodes[child]['title']}")
            else:
                removed_edges.append((parent, child))
                print(f"      Removing edge: {nodes[parent]['title']} -> {nodes[child]['title']}")
        else:
            # No cycle, keep the edge
            clean_cat_graph[parent].add(child)
            valid_edges.append((parent, child))
    
    # Find root node: category with no parents after cycle removal
    all_children = set()
    all_parents = set()
    for parent, child in valid_edges:
        if parent in category_nodes:
            all_parents.add(parent)
        if child in category_nodes:
            all_children.add(child)
    
    root_candidates = category_nodes - all_children
    
    if root_candidates:
        # Choose the candidate with most descendants
        best_root = None
        max_descendants = -1
        for candidate in root_candidates:
            # Count all descendants using BFS
            visited = set()
            queue = deque([candidate])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                for child in clean_cat_graph.get(node, []):
                    queue.append(child)
            descendant_count = len(visited) - 1
            if descendant_count > max_descendants:
                max_descendants = descendant_count
                best_root = candidate
        root_node = best_root
    else:
        # Fallback: pick the first category node
        root_node = list(category_nodes)[0] if category_nodes else None
    
    print(f"    Selected root: {nodes[root_node]['title'] if root_node else 'None'} ({root_node})")
    
    # Update structure
    tree_structure['edges'] = valid_edges
    tree_structure['root_node'] = root_node
    
    print(f"    Removed {len(removed_edges)} edges to eliminate cycles")
    print(f"    Final hierarchy: {len(valid_edges)} edges")
    
    return tree_structure


def compute_gold_distances(tree_structure: Dict) -> Dict[Tuple[str, str], int]:
    """
    Compute shortest path distances between all node pairs in the tree.
    
    Returns:
        Dict mapping (node1, node2) to shortest path distance
    """
    nodes = set(tree_structure['nodes'].keys())
    edges = tree_structure['edges']
    
    # Build adjacency list (undirected graph for distance calculation)
    adj = defaultdict(list)
    for parent, child in edges:
        if parent in nodes and child in nodes:
            adj[parent].append(child)
            adj[child].append(parent)
    
    distances = {}
    
    # BFS from each node to compute distances
    for start_node in nodes:
        dist = {start_node: 0}
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        
        # Store distances for this start node
        for end_node, d in dist.items():
            distances[(start_node, end_node)] = d
    
    return distances


def compute_gold_depths(tree_structure: Dict) -> Dict[str, int]:
    """
    Compute depth of each node from the root.
    
    Returns:
        Dict mapping node_id to depth (root has depth 0)
    """
    root = tree_structure['root_node']
    if not root:
        return {}
    
    nodes = set(tree_structure['nodes'].keys())
    edges = tree_structure['edges']
    
    # Build parent -> children mapping
    children = defaultdict(list)
    for parent, child in edges:
        if parent in nodes and child in nodes:
            children[parent].append(child)
    
    depths = {root: 0}
    queue = deque([root])
    
    while queue:
        current = queue.popleft()
        current_depth = depths[current]
        
        for child in children[current]:
            if child not in depths:
                depths[child] = current_depth + 1
                queue.append(child)
    
    return depths


# ================================================================
# Helper functions for centroid computation
# ================================================================

def build_node_title_lookup(nodes: List[str], node_titles: List[str], tree_structure: Dict) -> Dict[str, str]:
    """Build a robust node_id -> title mapping."""
    node_to_title: Dict[str, str] = {nid: title for nid, title in zip(nodes, node_titles)}
    ts_nodes = (tree_structure or {}).get('nodes', {}) or {}

    for nid, info in ts_nodes.items():
        if isinstance(info, dict):
            label = info.get('label') or info.get('title') or info.get('name')
            if isinstance(label, str) and label.strip():
                node_to_title[nid] = label
        elif isinstance(info, str):
            node_to_title[nid] = info

    return node_to_title


def compute_centroid_embeddings(embeddings: torch.Tensor, 
                               nodes: List[str], 
                               node_titles: List[str],
                               tree_structure: Dict,
                               gold_structure: Dict) -> Tuple[torch.Tensor, List[str], List[str], Dict]:
    """
    Compute centroid embeddings for category nodes using gold structure.
    Category nodes are represented as the average of their descendant leaf (entity) embeddings.
    """
    debug_info: Dict = {}
    
    # Use gold structure to determine entity vs category nodes
    entity_nodes = set(gold_structure.get('entity_nodes', []))
    category_nodes = set(gold_structure.get('category_nodes', []))
    
    # Build title lookup
    node_to_title = build_node_title_lookup(nodes, node_titles, tree_structure)
    
    # Map from node_id to its index in the provided `nodes` list
    node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes)}
    
    # Cache embeddings for entity nodes that exist in the current layer's embedding matrix
    node_embeddings: Dict[str, torch.Tensor] = {}
    for node_id in entity_nodes:
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            node_embeddings[node_id] = embeddings[idx]
    
    # Build parent-child relationships from gold structure
    parent_to_children: Dict[str, List[str]] = defaultdict(list)
    for parent, child in gold_structure['edges']:
        parent_to_children[parent].append(child)
    
    # Iteratively collect all descendant entity nodes for a given node (cycle-safe)
    def get_entity_descendants(node_id: str) -> List[str]:
        if node_id in entity_nodes:
            return [node_id]
        
        descendants = []
        visited = set()
        queue = deque([node_id])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            if current in entity_nodes:
                descendants.append(current)
            else:
                # Add children to queue
                for child in parent_to_children[current]:
                    if child not in visited:
                        queue.append(child)
        
        return descendants
    
    # Compute centroids for each category node from its descendant entities
    DEBUG_SAMPLE_N = 3
    for cat_node in category_nodes:
        descendants = get_entity_descendants(cat_node)
        valid_descendants = [d for d in descendants if d in node_to_idx]
        
        if valid_descendants:
            descendant_embeddings = [embeddings[node_to_idx[d]] for d in valid_descendants]
            centroid = torch.stack(descendant_embeddings).mean(dim=0)
            node_embeddings[cat_node] = centroid
            
            # Debug info
            cat_title = node_to_title.get(cat_node, cat_node)
            sample_list: List[str] = []
            for d in valid_descendants[:DEBUG_SAMPLE_N]:
                title = node_to_title.get(d, d)
                sample_list.append(f"{title} ({d})")
            if len(valid_descendants) > DEBUG_SAMPLE_N:
                sample_list.append(f"... and {len(valid_descendants) - DEBUG_SAMPLE_N} more")
            
            print(f"\n[DEBUG] Category '{cat_title}' (ID: {cat_node}):")
            print(f"  - Computed centroid from {len(valid_descendants)} entities")
            print(f"  - Sample entities: {sample_list}")
            
            debug_info[cat_node] = {
                'title': cat_title,
                'num_descendants': len(valid_descendants),
                'sample_descendants': sample_list
            }
    
    # Assemble outputs in the original node order, keeping only nodes we computed
    result_nodes: List[str] = []
    result_titles: List[str] = []
    result_embeddings: List[torch.Tensor] = []
    
    for node_id in nodes:
        if node_id in node_embeddings:
            result_nodes.append(node_id)
            result_titles.append(node_to_title.get(node_id, node_id))
            result_embeddings.append(node_embeddings[node_id])
    
    # Stack to tensor (K, D) or empty if nothing matched
    if result_embeddings:
        result_tensor = torch.stack(result_embeddings)
    else:
        result_tensor = torch.empty(0, embeddings.shape[1])
    
    print(f"\n[DEBUG] Centroid computation complete:")
    print(f"  - Original embeddings shape: {embeddings.shape}")
    print(f"  - Result embeddings shape: {result_tensor.shape}")
    print(f"  - Nodes with centroids: {len(result_nodes)}")
    
    return result_tensor, result_nodes, result_titles, debug_info


# ================================================================
# Data loading with gold structure integration
# ================================================================

def load_embeddings_with_gold(emb_dir: str, layer: int, probe_type: str, gold_structures: Dict) -> List[Dict]:
    """Load embeddings and compute centroids using gold structure"""
    
    paths = [os.path.join(emb_dir, f) for f in os.listdir(emb_dir) 
             if f.endswith("_embeddings.pt")]
    
    if not paths:
        print(f"ERROR: no *_embeddings.pt found in {emb_dir}", file=sys.stderr)
        sys.exit(1)
    
    paths.sort()
    print(f"Found {len(paths)} embedding files")
    
    data = []
    for p in tqdm(paths, desc="Loading embeddings"):
        try:
            blob = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(p, map_location="cpu")
        
        emb = blob["embeddings"]  # (N, L, D)
        if layer >= emb.shape[1]:
            raise ValueError(f"Layer {layer} out of range for {p}")
        
        md = blob["metadata"]
        tree_id = md["tree_id"]
        layer_emb = emb[:, layer, :]  # (N, D)
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(p)}")
        print(f"  Tree ID: {tree_id}")
        
        # Get corresponding gold structure
        if tree_id not in gold_structures:
            print(f"  WARNING: No gold structure found for tree_id {tree_id}, skipping")
            continue
        
        gold_structure = gold_structures[tree_id]
        
        # Compute centroid embeddings using gold structure
        centroid_emb, centroid_nodes, centroid_titles, debug_info = compute_centroid_embeddings(
            layer_emb, 
            md['nodes'], 
            md['node_titles'],
            md.get('tree_structure'),
            gold_structure
        )
        
        if centroid_emb.shape[0] == 0:
            print(f"  WARNING: No valid embeddings computed for tree {tree_id}, skipping")
            continue
        
        # Compute gold distances and depths
        gold_distances = compute_gold_distances(gold_structure)
        gold_depths = compute_gold_depths(gold_structure)
        
        # Create distance/depth matrices for the centroid nodes
        node_to_idx = {node_id: idx for idx, node_id in enumerate(centroid_nodes)}
        N = len(centroid_nodes)
        
        if probe_type == 'distance':
            # Build gold distance matrix
            distance_matrix = torch.full((N, N), float('inf'))
            for i, node_i in enumerate(centroid_nodes):
                for j, node_j in enumerate(centroid_nodes):
                    dist = gold_distances.get((node_i, node_j), float('inf'))
                    distance_matrix[i, j] = dist
            
            entry_distances = distance_matrix
        else:
            entry_distances = None
        
        if probe_type == 'depth':
            # Build gold depth vector
            depth_vector = torch.full((N,), -1.0)
            for i, node_id in enumerate(centroid_nodes):
                if node_id in gold_depths:
                    depth_vector[i] = gold_depths[node_id]
            
            entry_depths = depth_vector
        else:
            entry_depths = None
        
        entry = {
            "embeddings": centroid_emb,
            "nodes": centroid_nodes,
            "node_titles": centroid_titles,
            "tree_id": tree_id,
            "hidden_dim": layer_emb.shape[-1],
            "centroid_debug": debug_info,
            "gold_structure": gold_structure
        }
        
        if entry_distances is not None:
            entry['distances'] = entry_distances
        if entry_depths is not None:
            entry['depths'] = entry_depths
        
        data.append(entry)
        print(f"  Loaded with {len(centroid_nodes)} nodes and gold structure")
    
    return data


# ================================================================
# MST computation (from reporter.py)
# ================================================================

class UnionFind:
    """Naive UnionFind implementation for (slow) Prim's MST algorithm"""
    def __init__(self, n):
        self.parents = list(range(n))
        
    def union(self, i, j):
        if self.find(i) != self.find(j):
            i_parent = self.find(i)
            self.parents[i_parent] = j
            
    def find(self, i):
        i_parent = i
        while True:
            if i_parent != self.parents[i_parent]:
                i_parent = self.parents[i_parent]
            else:
                break
        return i_parent


def prims_matrix_to_edges(matrix):
    """
    Constructs a minimum spanning tree from the pairwise weights in matrix;
    returns the edges.
    
    Adapted from reporter.py but without POS tag filtering since we're working with entities.
    """
    pairs_to_distances = {}
    uf = UnionFind(len(matrix))
    
    for i_index in range(len(matrix)):
        for j_index in range(len(matrix)):
            if i_index != j_index:
                dist = matrix[i_index][j_index]
                if torch.isfinite(torch.tensor(dist)):
                    pairs_to_distances[(i_index, j_index)] = float(dist)
    
    edges = []
    for (i_index, j_index), distance in sorted(pairs_to_distances.items(), key=lambda x: x[1]):
        if uf.find(i_index) != uf.find(j_index):
            uf.union(i_index, j_index)
            edges.append((i_index, j_index))
    
    return edges


def get_argmin_excluding_invalid(prediction, valid_mask=None):
    """
    Gets the argmin of predictions, filtering out invalid entries.
    Adapted from get_nopunct_argmin in reporter.py.
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(prediction, dtype=torch.bool)
    
    # Find valid indices
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return torch.argmin(prediction).item()
    
    # Get argmin among valid indices
    valid_predictions = prediction[valid_indices]
    local_argmin = torch.argmin(valid_predictions)
    return valid_indices[local_argmin].item()


# ================================================================
# Evaluation metrics (based on reporter.py)
# ================================================================

def evaluate_distance_probe_correct(probe, dataset, device):
    """
    Evaluate distance probe with UUAS and DSpr metrics.
    Based on reporter.py implementation.
    """
    probe.eval()
    
    # For UUAS
    uspan_total = 0
    uspan_correct = 0
    
    # For DSpr (Distance Spearman)
    all_pred_distances = []
    all_gold_distances = []
    
    with torch.no_grad():
        for td in tqdm(dataset, desc="Evaluating distance probe"):
            emb = td["embeddings"].unsqueeze(0).to(device)  # (1, N, D)
            pred_dist = probe(emb).squeeze(0).cpu().float()  # (N, N)
            gold_dist = td["distances"].float()  # (N, N)
            
            N = gold_dist.size(0)
            
            # Collect distance pairs for DSpr calculation
            for i in range(N):
                for j in range(i + 1, N):  # Only upper triangle to avoid duplicates
                    if torch.isfinite(gold_dist[i, j]):
                        all_pred_distances.append(pred_dist[i, j].item())
                        all_gold_distances.append(gold_dist[i, j].item())
            
            # UUAS calculation using MST
            gold_edges = prims_matrix_to_edges(gold_dist.numpy())
            pred_edges = prims_matrix_to_edges(pred_dist.numpy())
            
            # Calculate edge overlap
            gold_edge_set = set([tuple(sorted(e)) for e in gold_edges])
            pred_edge_set = set([tuple(sorted(e)) for e in pred_edges])
            
            uspan_correct += len(gold_edge_set.intersection(pred_edge_set))
            uspan_total += len(gold_edges)
    
    # Calculate metrics
    uuas = uspan_correct / float(uspan_total) if uspan_total > 0 else 0.0
    
    if all_pred_distances and all_gold_distances:
        dspr, _ = spearmanr(all_pred_distances, all_gold_distances)
        dspr = float(dspr) if np.isfinite(dspr) else 0.0
    else:
        dspr = 0.0
    
    return {
        "uuas": uuas,
        "dspr": dspr,
        "num_edges": uspan_total,
        "num_distance_pairs": len(all_pred_distances)
    }


def evaluate_depth_probe_correct(probe, dataset, device):
    """
    Evaluate depth probe with Root accuracy and NSpr metrics.
    Based on reporter.py implementation.
    """
    probe.eval()
    
    # For Root accuracy
    total_trees = 0
    correct_root_predictions = 0
    
    # For NSpr (Norm Spearman)
    all_pred_depths = []
    all_gold_depths = []
    
    with torch.no_grad():
        for td in tqdm(dataset, desc="Evaluating depth probe"):
            emb = td["embeddings"].unsqueeze(0).to(device)  # (1, N, D)
            pred_depths = probe(emb).squeeze(0).cpu().float()  # (N,)
            gold_depths = td["depths"].float()  # (N,)
            
            # Filter valid depths (not -1)
            valid_mask = (gold_depths != -1)
            
            if valid_mask.any():
                valid_pred = pred_depths[valid_mask]
                valid_gold = gold_depths[valid_mask]
                
                # Collect for NSpr calculation
                all_pred_depths.extend(valid_pred.numpy().tolist())
                all_gold_depths.extend(valid_gold.numpy().tolist())
                
                # Root accuracy: node with minimum depth should be the gold root
                gold_root_idx = torch.argmin(valid_gold)
                pred_root_idx = torch.argmin(valid_pred)
                
                if gold_root_idx == pred_root_idx:
                    correct_root_predictions += 1
                
                total_trees += 1
    
    # Calculate metrics
    root_acc = correct_root_predictions / float(total_trees) if total_trees > 0 else 0.0
    
    if all_pred_depths and all_gold_depths:
        nspr, _ = spearmanr(all_pred_depths, all_gold_depths)
        nspr = float(nspr) if np.isfinite(nspr) else 0.0
    else:
        nspr = 0.0
    
    return {
        "root_acc": root_acc,
        "nspr": nspr,
        "num_trees": total_trees,
        "num_depth_points": len(all_pred_depths)
    }


# ================================================================
# Visualization (updated with better root positioning)
# ================================================================

def _canon_label(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return "Unknown"
    s = unicodedata.normalize("NFKC", s)  
    s = s.strip()
    s = re.sub(r"\s+", " ", s)            
    return s 


def _infer_leaf_occupations(gold_structure: Dict, node_to_title: Dict[str, str]) -> Dict[str, str]:
    """Infer a leaf->occupation mapping using gold structure."""
    entity_nodes = gold_structure.get('entity_nodes', [])
    
    # Build parent mapping
    child_to_parent: Dict[str, str] = {}
    for parent, child in gold_structure.get('edges', []):
        child_to_parent[child] = parent
    
    occ_by_leaf: Dict[str, str] = {}
    for leaf in entity_nodes:
        parent = child_to_parent.get(leaf)
        if parent:
            occ = node_to_title.get(parent, parent)
        else:
            occ = "Unknown"
        occ_by_leaf[leaf] = _canon_label(occ)
    
    return occ_by_leaf

def save_mst_html(probe, dataset, device, save_path: str, probe_type: str, model_name: str = "Unknown Model", layer: int = 0, rank: int = 0):
    """Save an interactive, hierarchical tree HTML with pan & zoom. Root node guaranteed at top."""
    probe_is_distance = (probe_type == 'distance')
    
    def _gold_edges_idx(gold_edges, id2idx):
        edges = []
        for p, c in gold_edges:
            if p in id2idx and c in id2idx:
                edges.append([id2idx[p], id2idx[c]])
        return edges
    
    def _find_gold_root(gold_structure: Dict, present_ids: set) -> Optional[str]:
        root = gold_structure.get('root_node')
        if root and root in present_ids:
            return root
        # Fallback
        for node_id in gold_structure.get('nodes', {}):
            if node_id in present_ids:
                return node_id
        return None
    
    out_records = []
    
    with torch.no_grad():
        for td in tqdm(dataset, desc="Generating hierarchical HTML data"):
            nodes = td["nodes"]
            titles = td["node_titles"]
            gold_structure = td["gold_structure"]
            
            node_to_title = {nid: title for nid, title in zip(nodes, titles)}
            occ_by_leaf = _infer_leaf_occupations(gold_structure, node_to_title)
            leaf_set = set(gold_structure.get('entity_nodes', []))
            
            id2idx = {nid: i for i, nid in enumerate(nodes)}
            
            # Gold directed edges & root
            gold_edges_idx = _gold_edges_idx(gold_structure['edges'], id2idx)
            gold_root_id = _find_gold_root(gold_structure, set(nodes))
            gold_root_idx = id2idx.get(gold_root_id, 0) if nodes else 0
            
            # Debug: Print root information
            print(f"Tree {td['tree_id']}: Root ID={gold_root_id}, Root Index={gold_root_idx}")
            
            # Predicted MST edges (undirected index pairs)
            pred_edges_idx = []
            if probe_is_distance:
                emb = td["embeddings"].unsqueeze(0).to(device)
                pred = probe(emb).squeeze(0).cpu().float().numpy()
                pred_edges_idx = prims_matrix_to_edges(pred)
            
            # Package nodes for the client
            vis_nodes = []
            for i, nid in enumerate(nodes):
                is_leaf = (nid in leaf_set)
                occ = occ_by_leaf.get(nid) if is_leaf else None
                vis_nodes.append({
                    "id": nid,
                    "title": node_to_title.get(nid, nid),
                    "index": i,
                    "is_leaf": is_leaf,
                    "occupation": occ,
                    "is_root": (i == gold_root_idx)  # Mark root node
                })
            
            out_records.append({
                "tree_id": td["tree_id"],
                "root_index": gold_root_idx,
                "root_title": node_to_title.get(gold_root_id, gold_root_id),
                "nodes": vis_nodes,
                "edges": {
                    "gold": gold_edges_idx,
                    "predicted": pred_edges_idx
                }
            })
    
    # HTML template with model info and guaranteed root-at-top layout
    probe_info = f"{probe_type.title()} Probe – {model_name} L{layer} rank{rank}"
    
    template = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8" />
<title>Hierarchical Tree Visualization</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
  .tree-card {{ border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px 16px 8px; margin-bottom: 24px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .hdr {{ display:flex; align-items:center; justify-content:space-between; gap: 12px; margin-bottom: 6px; }}
  .title {{ font-weight: 600; font-size: 16px; }}
  .controls label {{ margin-right: 12px; cursor: pointer; }}
  .legend {{ display:flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 4px; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; font-size: 12px; padding:2px 6px; border:1px solid #eee; border-radius: 999px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; display:inline-block; }}
  .wrap {{ position: relative; }}
  svg {{ width: 100%; border-radius: 12px; background: #ffffff; touch-action: pinch-zoom; }}
  .node circle {{ stroke: #333; stroke-width: 0.6px; }}
  .node text {{ font-size: 11px; pointer-events: none; }}
  .node.root text {{ font-weight: bold; }}
  .link {{ stroke: #848484; stroke-opacity: 0.85; fill: none; }}
  .tip {{ position: absolute; background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 10px; font-size: 12px; pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display:none; }}
  .probe-info {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 20px; font-weight: 500; color: #475569; }}
</style>
<body>
<h1 style="margin:0 0 16px">Hierarchical Tree Visualization</h1>
<div class="probe-info">{probe_info}</div>
<p style="margin:0 0 20px; color:#374151">Gold = taxonomy edges (parent→child). Predicted (if available) = MST rooted at the gold root. Click-drag to pan, wheel/pinch to zoom.</p>
<div id="app"></div>
<div class="tip" id="tooltip"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const DATA = __DATA__;
const app = document.getElementById('app');

const LEVEL_SEP = 120;
const NODE_SPACING = 100;
const MARGIN = {{top: 60, right: 60, bottom: 60, left: 60}};
const MIN_HEIGHT = 800;
const SCALE_EXTENT = [0.2, 4];

function makePalette(labels) {{
  const uniq = Array.from(new Set(labels.filter(Boolean)));
  const m = new Map();
  const n = Math.max(uniq.length, 1);
  uniq.forEach((lab, i) => {{
    const hue = Math.round(360 * i / n);
    m.set(lab, `hsl(${{hue}},65%,50%)`);
  }});
  m.set('Unknown', '#999999');
  return m;
}}

function orientUndirectedToRoot(edgesUndirected, rootIndex) {{
  const g = new Map();
  edgesUndirected.forEach(([a,b])=>{{
    if(!g.has(a)) g.set(a, new Set());
    if(!g.has(b)) g.set(b, new Set());
    g.get(a).add(b); g.get(b).add(a);
  }});
  
  const parent = new Map();
  const queue = [rootIndex];
  parent.set(rootIndex, -1);
  
  for(let i=0; i<queue.length; i++){{
    const u = queue[i];
    const neighbors = Array.from(g.get(u) || []);
    for(const v of neighbors){{
      if(!parent.has(v)){{
        parent.set(v, u);
        queue.push(v);
      }}
    }}
  }}
  
  const directed = [];
  for(const [child, par] of parent.entries()){{
    if(par !== -1) directed.push([par, child]);
  }}
  return directed;
}}

function buildHierarchy(nodes, directedEdges, rootIndex) {{
  console.log('Building hierarchy for root index:', rootIndex);
  console.log('Number of directed edges:', directedEdges.length);
  
  // Create node map
  const nodeMap = new Map();
  nodes.forEach(n => {{
    nodeMap.set(n.index, {{...n, children: []}});
  }});
  
  // Build parent-child relationships
  directedEdges.forEach(([parentIdx, childIdx]) => {{
    const parent = nodeMap.get(parentIdx);
    const child = nodeMap.get(childIdx);
    if (parent && child) {{
      parent.children.push(child);
    }}
  }});
  
  // Get root node
  const rootNode = nodeMap.get(rootIndex);
  if (!rootNode) {{
    console.error('Root node not found! Using first available node.');
    const firstNode = nodeMap.values().next().value;
    return d3.hierarchy(firstNode || {{title: 'Error', children: []}});
  }}
  
  console.log('Root node found:', rootNode.title);
  return d3.hierarchy(rootNode, d => d.children || []);
}}

function drawCard(record) {{
  console.log('Drawing card for tree:', record.tree_id, 'Root:', record.root_title);
  
  const card = document.createElement('div');
  card.className = 'tree-card';
  app.appendChild(card);

  // Header
  const hdr = document.createElement('div');
  hdr.className = 'hdr';
  const title = document.createElement('div');
  title.className = 'title';
  title.textContent = `Tree: ${{record.tree_id}} (Root: ${{record.root_title}})`;
  hdr.appendChild(title);

  const controls = document.createElement('div');
  controls.className = 'controls';
  const hasPred = (record.edges.predicted && record.edges.predicted.length);
  const modes = hasPred ? ['gold','predicted'] : ['gold'];
  modes.forEach((m,idx)=>{{
    const label = document.createElement('label');
    label.innerHTML = `<input type="radio" name="mode-${{record.tree_id}}" value="${{m}}" ${{idx===0?'checked':''}}> ${{m.toUpperCase()}}`;
    controls.appendChild(label);
  }});
  hdr.appendChild(controls);
  card.appendChild(hdr);

  // Legend
  const occs = record.nodes.filter(n=>n.is_leaf).map(n=>n.occupation||'Unknown');
  const pal = makePalette(occs);
  const legend = document.createElement('div');
  legend.className = 'legend';
  Array.from(new Set(occs)).forEach(o=>{{
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<span class="swatch" style="background:${{pal.get(o)||'#999'}}"></span><span>${{o||'Unknown'}}</span>`;
    legend.appendChild(item);
  }});
  card.appendChild(legend);

  // SVG setup
  const wrap = d3.select(card).append('div').attr('class','wrap');
  const svg = wrap.append('svg');
  const bg = svg.append('rect').attr('fill','transparent').attr('pointer-events','all');
  const container = svg.append('g').attr('class','container');
  const gLinks = container.append('g').attr('class','links');
  const gNodes = container.append('g').attr('class','nodes');

  // Tooltip
  const tip = document.getElementById('tooltip');
  function showTip(html, x, y) {{ 
    tip.style.left = (x+10)+'px'; 
    tip.style.top = (y+10)+'px'; 
    tip.innerHTML = html; 
    tip.style.display = 'block'; 
  }}
  function hideTip() {{ tip.style.display = 'none'; }}

  // Zoom behavior
  const zoom = d3.zoom()
    .scaleExtent(SCALE_EXTENT)
    .on('zoom', (event) => {{ container.attr('transform', event.transform); }});
  svg.call(zoom).on('dblclick.zoom', null);

  function edgesFor(mode){{
    if(mode==='predicted'){{
      return orientUndirectedToRoot(record.edges.predicted, record.root_index);
    }}
    return record.edges.gold;
  }}

  function render(mode){{
    console.log('Rendering mode:', mode);
    
    const directed = edgesFor(mode);
    const hierarchy = buildHierarchy(record.nodes, directed, record.root_index);

    // Create tree layout - FORCE vertical orientation
    const treeLayout = d3.tree()
      .nodeSize([NODE_SPACING, LEVEL_SEP]);  // Width between siblings, height between levels
    
    treeLayout(hierarchy);

    const nodes = hierarchy.descendants();
    const links = hierarchy.links();
    
    console.log('Tree nodes count:', nodes.length);
    console.log('Root position:', {{x: hierarchy.x, y: hierarchy.y}});

    // Calculate bounds to determine SVG size
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    nodes.forEach(d => {{
      minX = Math.min(minX, d.x);
      maxX = Math.max(maxX, d.x);
      minY = Math.min(minY, d.y);
      maxY = Math.max(maxY, d.y);
    }});

    const width = Math.max(1000, (maxX - minX) + MARGIN.left + MARGIN.right + 100);
    const height = Math.max(MIN_HEIGHT, (maxY - minY) + MARGIN.top + MARGIN.bottom + 100);

    svg.attr('width', width).attr('height', height);
    bg.attr('width', width).attr('height', height);

    // Transform to center the tree and ensure root is at top
    const centerX = width / 2;
    const startY = MARGIN.top;
    
    // Apply translation to center horizontally and position root at top
    nodes.forEach(d => {{
      d.displayX = d.x + centerX;
      d.displayY = d.y - minY + startY;  // Normalize so root is at startY
    }});

    // Draw links using Bezier curves
    const linkGenerator = d3.linkVertical()
      .x(d => d.displayX)
      .y(d => d.displayY);

    gLinks.selectAll('path').data(links)
      .join('path')
      .attr('class', 'link')
      .attr('d', linkGenerator)
      .attr('stroke-width', 1.5);

    // Draw nodes
    const nodeSelection = gNodes.selectAll('g.node').data(nodes, d => d.data.index);
    
    const nodeEnter = nodeSelection.enter()
      .append('g')
      .attr('class', d => `node ${{d.data.is_root ? 'root' : ''}}`);

    nodeEnter.append('circle')
      .attr('r', d => d.data.is_leaf ? 6 : (d.data.is_root ? 10 : 8))
      .attr('fill', d => {{
        if (d.data.is_root) return '#ff6b6b';  // Red for root
        return d.data.is_leaf ? (pal.get(d.data.occupation) || '#999') : '#444';
      }})
      .attr('stroke-width', d => d.data.is_root ? 2 : 0.6)
      .on('mousemove', (event, d) => {{
        const depth = d.depth;
        const html = `<b>${{d.data.title}}</b><br/>ID: ${{d.data.id}}<br/>Depth: ${{depth}}${{d.data.is_leaf ? `<br/>Occupation: ${{d.data.occupation||'Unknown'}}` : ''}}${{d.data.is_root ? '<br/><b style="color:#ff6b6b">ROOT NODE</b>' : ''}}`;
        showTip(html, event.pageX, event.pageY);
      }})
      .on('mouseout', hideTip);

    nodeEnter.append('text')
      .attr('dy', d => d.data.is_leaf ? 20 : -14)
      .attr('text-anchor', 'middle')
      .text(d => d.data.title)
      .attr('fill', d => d.data.is_root ? '#ff6b6b' : '#222')
      .attr('font-size', d => d.data.is_root ? '14px' : (d.data.is_leaf ? '10px' : '11px'))
      .attr('font-weight', d => d.data.is_root ? 'bold' : (d.data.is_leaf ? 'normal' : '600'));

    nodeSelection.merge(nodeEnter)
      .attr('transform', d => `translate(${{d.displayX}}, ${{d.displayY}})`);

    nodeSelection.exit().remove();

    // Auto-fit view with some padding
    setTimeout(() => {{
      try {{
        const bbox = container.node().getBBox();
        if (bbox && bbox.width && bbox.height) {{
          const padding = 50;
          const scale = Math.min(
            (width - padding * 2) / bbox.width,
            (height - padding * 2) / bbox.height,
            1.5  // Don't zoom in too much
          );
          const translateX = (width - bbox.width * scale) / 2 - bbox.x * scale;
          const translateY = padding;  // Keep some top padding
          
          const transform = d3.zoomIdentity.translate(translateX, translateY).scale(scale);
          svg.transition().duration(500).call(zoom.transform, transform);
        }}
      }} catch(e) {{
        console.warn('Auto-fit failed:', e);
      }}
    }}, 100);
  }}

  // Event listeners
  controls.querySelectorAll('input[type=radio]').forEach(r => {{
    r.addEventListener('change', () => render(r.value));
  }});
  
  // Initial render
  const initialMode = controls.querySelector('input[type=radio]:checked').value;
  render(initialMode);
}}

// Render all trees
DATA.forEach(record => drawCard(record));
</script>
</body>
</html>"""

    html = template.replace("__DATA__", json.dumps(out_records))

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✓ Hierarchical Tree HTML saved to {save_path}")

# ================================================================
# Probe loading
# ================================================================

def load_probe(path: str, probe_type: str, hidden_dim: int, rank: int, device):
    """Load pre-trained probe from file"""
    state_dict = torch.load(path, map_location=device)
    
    if probe_type == 'distance':
        probe = TwoWordPSDProbe(hidden_dim, rank, device)
    else:
        probe = OneWordPSDProbe(hidden_dim, rank, device)
    
    probe.load_state_dict(state_dict)
    probe.eval()
    
    return probe


# ================================================================
# CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate structural probe with centroid representations and gold data")
    parser.add_argument("--embedding_dir", required=True, help="Directory containing embedding files")
    parser.add_argument("--gold_data_path", required=True, help="Path to JSONL gold data file")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to use")
    parser.add_argument("--probe_type", choices=['distance', 'depth'], required=True, help="Type of probe")
    parser.add_argument("--probe_rank", type=int, default=128, help="Probe rank")
    parser.add_argument("--probe_path", required=True, help="Path to pre-trained probe")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--save_dir", default="probe_results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--vis_path", default=None, help="Path to save MST HTML (e.g., results/vis.html)")
    parser.add_argument("--model_name", default=None, help="Model name for display (e.g., 'meta-llama/Meta-Llama-3-8B')")
    return parser.parse_args()


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Infer model name if not provided
    model_name = args.model_name
    if not model_name:
        # Try to infer from embedding directory path
        if 'gpt2' in args.embedding_dir.lower():
            model_name = 'gpt2'
        elif 'llama' in args.embedding_dir.lower():
            if '3-8b' in args.embedding_dir.lower():
                model_name = 'meta-llama/Meta-Llama-3-8B'
            elif '3-70b' in args.embedding_dir.lower():
                model_name = 'meta-llama/Meta-Llama-3-70B'
            else:
                model_name = 'meta-llama/Llama'
        elif 'bert' in args.embedding_dir.lower():
            model_name = 'bert-base-uncased'
        else:
            model_name = 'Unknown Model'
    
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Probe type: {args.probe_type}")
    print(f"Probe path: {args.probe_path}")
    print(f"Gold data: {args.gold_data_path}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load gold structures
    gold_structures = load_gold_structures(args.gold_data_path)
    
    # Load data with gold structure integration
    print(f"\nLoading embeddings from {args.embedding_dir}")
    test_data = load_embeddings_with_gold(args.embedding_dir, args.layer, args.probe_type, gold_structures)
    
    if not test_data:
        print("ERROR: No data loaded", file=sys.stderr)
        sys.exit(1)
    
    hidden_dim = test_data[0]["hidden_dim"]
    print(f"\nEvaluation setup:")
    print(f"  - Model: {model_name}")
    print(f"  - Layer: {args.layer}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Probe rank: {args.probe_rank}")
    print(f"  - Test trees: {len(test_data)}")
    print(f"  - Gold structures: {len(gold_structures)}")
    
    # Load probe
    probe = load_probe(args.probe_path, args.probe_type, hidden_dim, args.probe_rank, device)
    print(f"  - Loaded {args.probe_type} probe from {args.probe_path}")
    
    # Evaluate with correct metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    if args.probe_type == 'distance':
        metrics = evaluate_distance_probe_correct(probe, test_data, device)
        print(f"\nDistance Probe Metrics:")
        print(f"  - UUAS (Undirected Unlabeled Attachment Score): {metrics['uuas']:.4f}")
        print(f"  - DSpr (Distance Spearman Correlation): {metrics['dspr']:.4f}")
        print(f"  - Total edges evaluated: {metrics['num_edges']}")
        print(f"  - Total distance pairs: {metrics['num_distance_pairs']}")
    else:
        metrics = evaluate_depth_probe_correct(probe, test_data, device)
        print(f"\nDepth Probe Metrics:")
        print(f"  - Root Accuracy: {metrics['root_acc']*100:.2f}%")
        print(f"  - NSpr (Norm Spearman Correlation): {metrics['nspr']:.4f}")
        print(f"  - Total trees evaluated: {metrics['num_trees']}")
        print(f"  - Total depth points: {metrics['num_depth_points']}")
    
    # Save metrics
    metrics_path = os.path.join(args.save_dir, f"metrics", f"{args.probe_type}", f"n_comp{args.probe_rank}", f"layer{args.layer}.json")
    metrics_dir = os.path.dirname(metrics_path)
    os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Save MST HTML visualization if requested
    if args.vis_path:
        save_mst_html(probe, test_data, device, args.vis_path, args.probe_type, model_name, args.layer, args.probe_rank)
    
    print("\n✓ Evaluation complete")


if __name__ == "__main__":
    main()