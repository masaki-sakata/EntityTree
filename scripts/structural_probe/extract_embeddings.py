#!/usr/bin/env python3
"""
Extract hidden states from language models for hierarchical taxonomy data

This script extracts hidden states from all layers of a language model
for entities in a taxonomy tree, and uses centroids for category representations.

Usage:
    python extract_embeddings.py --data_path data.jsonl --output_dir embeddings/ --model_name meta-llama/Meta-Llama-3-8B
    python extract_embeddings.py --data_path data.jsonl --output_dir embeddings/ --model_name meta-llama/Meta-Llama-3-8B --split_tree
"""

import json
import argparse
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm
import itertools
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Extract hidden states / fastText vectors for taxonomy data')
    parser.add_argument('--data_path',  type=str, required=True,
                        help='Path to the JSONL data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the embeddings')

    # New: explicit run mode
    parser.add_argument('--run_mode', choices=['train', 'test'], required=True,
                        help='Execution mode: train or test')

    # Model configuration
    parser.add_argument('--model_type', choices=['hf', 'fasttext'], default='hf',
                        help='hf: HuggingFace Transformer\n'
                             'fasttext: fastText vectors')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B',
                        help='HuggingFace model name (used when --model_type hf)')
    parser.add_argument('--vector_path', type=str, default=None,
                        help='Path to fastText .bin file (required when --model_type fasttext)')

    # Tree splitting option (train only)
    parser.add_argument('--split_tree', action='store_true',
                        help='Split the tree into multiple trees with one entity per category (train mode only)')
    parser.add_argument('--num_splits', type=int, default=None,
                        help='Number of tree splits to generate (default: all possible combinations for small trees)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for tree splitting')

    # New: centroid recomputation source for test mode
    parser.add_argument('--centroid_data_path', type=str, default=None,
                        help='Path to the GOLD (train) JSONL used to recompute category/root centroids in test mode')

    # New: policy for missing categories in test vs. train centroids
    parser.add_argument('--missing_category_policy', type=str, choices=['error', 'root', 'skip'], default='error',
                        help='How to handle test categories missing from train centroids: error | root | skip')

    # Other settings
    parser.add_argument('--max_length', type=int, default=500,
                        help='Maximum sequence length (for hf)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (currently unused)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--text_mode', type=str, default='title',
                        choices=['title', 'title_desc_title'],
                        help='Text construction mode (only title available for fastText)')
    parser.add_argument('--is_verbose', action='store_true',
                        help='Enable verbose output for debugging')

    return parser.parse_args()



def load_data(data_path):
    """Load JSONL data and organize by tree_id"""
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist!")
        sys.exit(1)
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {data_path}: {e}")
        sys.exit(1)
    
    # Parse JSONL
    lines = content.strip().split('\n')
    data = []
    
    for i, line in enumerate(lines):
        if line.strip():
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i+1}: {e}")
                sys.exit(1)
    
    if not data:
        print("Error: No valid data found!")
        sys.exit(1)
    
    # Group by tree_id
    tree_groups = defaultdict(list)
    for item in data:
        tree_groups[item['tree_id']].append(item)
    
    print(f"Loaded {len(data)} nodes across {len(tree_groups)} trees")
    for tree_id, items in tree_groups.items():
        print(f"  Tree {tree_id}: {len(items)} nodes")
    
    return tree_groups


def build_taxonomy_structure(tree_data, text_mode, is_verbose=False):
    """Build taxonomy tree structure from data"""
    nodes = {}
    edges = []
    
    # First pass: collect all node information
    for item in tree_data:
        node_info = {
            'qid': item['qid'],
            'title': item['wiki_title'],
            'is_entity': item.get('is_entity', False)
        }
        
        if text_mode == 'title_desc_title':
            node_info['description'] = item.get('description', '')
        
        nodes[item['qid']] = node_info
    
    if is_verbose:
        print(f"\n[DEBUG] Collected {len(nodes)} nodes")
        entities = [qid for qid, n in nodes.items() if n.get('is_entity', False)]
        non_entities = [qid for qid, n in nodes.items() if not n.get('is_entity', False)]
        print(f"  Entities: {len(entities)}")
        print(f"  Non-entities: {len(non_entities)}")
    
    # Second pass: process edges with complete node information
    for item in tree_data:
        for edge in item['edges']:
            target_qid = edge['target_qid']
            
            # Skip if target node doesn't exist in our tree
            if target_qid not in nodes:
                if is_verbose:
                    print(f"  Skipping edge from {item['qid']} to {target_qid} (target not in tree)")
                continue
            
            # Determine edge direction based on node types
            if item.get('is_entity', False):
                # Entity -> Category: Category is parent of Entity
                edges.append((target_qid, item['qid']))
                if is_verbose and len(edges) <= 50:
                    print(f"  Entity→Cat edge: {target_qid} → {item['qid']} ({nodes[target_qid]['title']} → {nodes[item['qid']]['title']})")
            else:
                # Non-entity (Category or Root)
                target_is_entity = nodes[target_qid].get('is_entity', False)
                
                if target_is_entity:
                    # Category -> Entity: Category is parent of Entity
                    edges.append((item['qid'], target_qid))
                    if is_verbose and len(edges) <= 50:
                        print(f"  Cat→Entity edge: {item['qid']} → {target_qid} ({nodes[item['qid']]['title']} → {nodes[target_qid]['title']})")
                else:
                    # Category -> Category/Root: determine by context
                    # In taxonomy, usually the broader category is the parent
                    # Person (root) should be parent of occupation categories
                    if item['wiki_title'] == 'Person':
                        # Person -> Category
                        edges.append((item['qid'], target_qid))
                        if is_verbose and len(edges) <= 50:
                            print(f"  Root→Cat edge: {item['qid']} → {target_qid} ({nodes[item['qid']]['title']} → {nodes[target_qid]['title']})")
                    else:
                        # Category -> Person (or other broader category)
                        edges.append((target_qid, item['qid']))
                        if is_verbose and len(edges) <= 50:
                            print(f"  Cat→Root edge: {target_qid} → {item['qid']} ({nodes[target_qid]['title']} → {nodes[item['qid']]['title']})")
    
    if is_verbose:
        print(f"  Total edges created: {len(edges)}")
    
    return {'nodes': nodes, 'edges': edges}


def classify_nodes(tree_structure):
    """Classify nodes into root, categories, and entities"""
    nodes = tree_structure['nodes']
    edges = tree_structure['edges']
    
    # Build parent-child relationships
    children = defaultdict(set)
    parents = defaultdict(set)
    
    for parent, child in edges:
        children[parent].add(child)
        parents[child].add(parent)
    
    # Identify node types based on is_entity flag
    root_nodes = []
    category_nodes = []
    entity_nodes = []
    
    for qid, node_info in nodes.items():
        if node_info.get('is_entity', False):
            entity_nodes.append(qid)
        elif len(parents[qid]) == 0:
            # No parents = root node
            root_nodes.append(qid)
        else:
            # Has parents and not an entity = category
            category_nodes.append(qid)
    
    # Find category-entity relationships
    category_entities = defaultdict(list)
    for cat in category_nodes:
        for child in children[cat]:
            if child in entity_nodes:
                category_entities[cat].append(child)
    
    # Debug: Check for categories without entities
    empty_categories = [cat for cat in category_nodes if cat not in category_entities or len(category_entities[cat]) == 0]
    if empty_categories:
        print(f"  Warning: Categories without entities: {empty_categories}")
        # Try to find entities for these categories by checking all entities' parents
        for entity in entity_nodes:
            for parent in parents[entity]:
                if parent in category_nodes and parent not in category_entities:
                    category_entities[parent] = []
                if parent in category_nodes:
                    if entity not in category_entities[parent]:
                        category_entities[parent].append(entity)
    
    return {
        'root': root_nodes,
        'categories': category_nodes,
        'entities': entity_nodes,
        'category_entities': category_entities,
        'children': children,
        'parents': parents
    }


def build_input_text(node, text_mode):
    """Build input text for a node"""
    title = node['title'].strip()
    desc  = (node.get('description') or '').strip()

    if title == '':
        raise ValueError(f"Title is empty for node {node}")

    if text_mode == 'title':
        return title
    elif text_mode == 'title_desc_title':
        if desc:
            return f"{title}. {desc}. {title}"
        else:
            raise ValueError("Description is empty, cannot use 'title_desc_title' mode without a description.")
    else:
        raise ValueError(f"Unknown text_mode: {text_mode}")


def calculate_tree_distances(tree_structure):
    """Calculate tree distance matrix using Floyd-Warshall"""
    nodes = list(tree_structure['nodes'].keys())
    n = len(nodes)
    
    # Initialize with infinity
    distances = np.full((n, n), np.inf)
    
    # Node to index mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Diagonal elements are 0
    for i in range(n):
        distances[i, i] = 0
    
    # Direct edges have distance 1
    for parent, child in tree_structure['edges']:
        if parent in node_to_idx and child in node_to_idx:
            i, j = node_to_idx[parent], node_to_idx[child]
            distances[i, j] = 1
            distances[j, i] = 1  # Treat as undirected
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distances[i, k] + distances[k, j] < distances[i, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
    
    return distances, nodes


def load_model_and_tokenizer(args, device):
    """Load HuggingFace or fastText model"""
    if args.model_type == 'hf':
        try:
            print(f"Loading HF model: {args.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoModel.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                output_hidden_states=True
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model.to(device).eval()
            print(f"✓ HF model loaded on {device} (layers={model.config.num_hidden_layers})")
            return tokenizer, model

        except Exception as e:
            print(f"[ERROR] HuggingFace model load failed: {e}")
            sys.exit(1)

    elif args.model_type == 'fasttext':
        if args.vector_path is None:
            print("[ERROR] --model_type fasttext requires --vector_path")
            sys.exit(1)

        try:
            import fasttext
            print(f"Loading fastText vectors from {args.vector_path}")
            ft_model = fasttext.load_model(args.vector_path)
            print(f"✓ fastText model loaded (dim={ft_model.get_dimension()})")
            return None, ft_model
        except Exception as e:
            print(f"[ERROR] fastText model load failed: {e}")
            sys.exit(1)

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")


def _extract_hidden_states_hf(texts, tokenizer, model, device, max_length=500, debug=False, debug_max_print=5):
    """Extract hidden states from HuggingFace model"""
    all_hidden_states = []
    printed = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting embeddings"):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            # Get last non-PAD token index
            attn = inputs['attention_mask'][0]
            last_idx = attn.nonzero()[-1].item()

            # Debug output
            if debug and printed < debug_max_print:
                ids = inputs['input_ids'][0]
                id_list = ids.tolist()
                tokens = tokenizer.convert_ids_to_tokens(id_list)
                last_id = id_list[last_idx]
                last_tok = tokens[last_idx]
                last_str = tokenizer.decode([last_id]).strip()

                print("\n" + "=" * 72)
                print(f"TEXT             : {text}")
                print(f"SEQ LEN          : {len(tokens)} (max={max_length})")
                print(f"LAST IDX         : {last_idx}")
                print(f"  └ token id     : {last_id}")
                print(f"  └ token (subwd): {last_tok}")
                print(f"  └ decoded str  : \"{last_str}\"")
                print("ATTN MASK tail   :", attn.tolist()[-10:])
                print("=" * 72)
                printed += 1

            # Collect per-layer vectors
            vecs = []
            for layer_h in hidden_states:
                vecs.append(layer_h[0, last_idx, :].cpu())

            all_hidden_states.append(torch.stack(vecs))

    return torch.stack(all_hidden_states)


def _extract_fasttext_embeddings(texts, ft_model, debug=False, debug_max_print=5):
    """Extract fastText embeddings (token average)"""
    dim = ft_model.get_dimension()
    vec_list = []
    printed = 0

    for text in tqdm(texts, desc="[fastText] Averaging vectors"):
        tokens = text.strip().split()

        if len(tokens) == 0:
            vec = np.zeros(dim, dtype=np.float32)
        else:
            vecs = [ft_model.get_word_vector(tok) for tok in tokens]
            vec = np.mean(vecs, axis=0)

        # Debug output
        if debug and printed < debug_max_print:
            print("\n" + "-" * 70)
            print(f"[fastText-DEBUG] TEXT        : {text}")
            print(f"[fastText-DEBUG] TOKENS ({len(tokens)}) : {tokens}")
            print(f"[fastText-DEBUG] VEC L2-norm : {np.linalg.norm(vec):.4f}")
            print("-" * 70)
            printed += 1

        vec_list.append(torch.tensor(vec, dtype=torch.float32))

    return torch.stack(vec_list).unsqueeze(1)


def extract_hidden_states(texts, args, tokenizer, model, device):
    """Wrapper for extracting hidden states"""
    if args.model_type == 'hf':
        return _extract_hidden_states_hf(
            texts,
            tokenizer,
            model,
            device,
            max_length=args.max_length,
            debug=args.is_verbose
        )
    elif args.model_type == 'fasttext':
        return _extract_fasttext_embeddings(
            texts,
            model,
            debug=args.is_verbose
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

def recompute_centroids_from_data(centroid_data_path, args, tokenizer, model, device, is_verbose=False):
    """
    Recompute category/root centroids from GOLD (train) data on-the-fly.
    This function does NOT use split; it directly aggregates entity embeddings per category.

    Returns:
        category_centroids: dict[qid] -> tensor [num_layers, hidden_dim]
        root_centroid: tensor [num_layers, hidden_dim]
        stats: dict with counts and coverage info
    """
    if centroid_data_path is None or not os.path.exists(centroid_data_path):
        print("[ERROR] --centroid_data_path must be provided and exist in test mode")
        sys.exit(1)

    # Load GOLD data grouped by tree
    gold_groups = load_data(centroid_data_path)

    # Aggregators for per-category sum and counts
    cat_sum = defaultdict(lambda: None)   # qid -> tensor sum
    cat_cnt = defaultdict(int)            # qid -> count of entity embeddings included

    # For root centroid (global)
    root_sum = None
    root_cnt = 0

    total_gold_entities = 0
    total_gold_categories = set()

    for tree_id, gold_tree in gold_groups.items():
        # Build structure and classify
        gold_struct = build_taxonomy_structure(gold_tree, args.text_mode, is_verbose)
        gold_cls = classify_nodes(gold_struct)

        # Collect entity texts and qids
        gold_entity_texts, gold_entity_qids = [], []
        for qid in gold_cls['entities']:
            if qid in gold_struct['nodes']:
                gold_entity_texts.append(build_input_text(gold_struct['nodes'][qid], args.text_mode))
                gold_entity_qids.append(qid)

        # Extract embeddings for GOLD entities
        if gold_entity_texts:
            gold_entity_tensor = extract_hidden_states(
                gold_entity_texts,
                args=args,
                tokenizer=tokenizer,
                model=model,
                device=device
            )
            # Map qid -> embedding [num_layers, hidden_dim]
            gold_entity_embeds = {qid: gold_entity_tensor[i] for i, qid in enumerate(gold_entity_qids)}
        else:
            gold_entity_embeds = {}

        # Aggregate per-category by summing entity embeddings (weighted by entity count)
        for cat_qid, ent_list in gold_cls['category_entities'].items():
            total_gold_categories.add(cat_qid)
            for ent_qid in ent_list:
                if ent_qid in gold_entity_embeds:
                    emb = gold_entity_embeds[ent_qid]
                    if cat_sum[cat_qid] is None:
                        cat_sum[cat_qid] = emb.clone()
                    else:
                        cat_sum[cat_qid] += emb
                    cat_cnt[cat_qid] += 1

        # Aggregate root
        for ent_qid in gold_cls['entities']:
            if ent_qid in gold_entity_embeds:
                emb = gold_entity_embeds[ent_qid]
                if root_sum is None:
                    root_sum = emb.clone()
                else:
                    root_sum += emb
                root_cnt += 1

        total_gold_entities += len(gold_cls['entities'])

    # Build centroids by dividing sum by count
    category_centroids = {}
    for cat_qid, s in cat_sum.items():
        if s is not None and cat_cnt[cat_qid] > 0:
            category_centroids[cat_qid] = s / cat_cnt[cat_qid]

    if root_sum is None or root_cnt == 0:
        root_centroid = None
        print("[WARNING] No root centroid could be computed from GOLD data (no entities available).")
    else:
        root_centroid = root_sum / root_cnt

    stats = {
        'gold_trees': len(gold_groups),
        'gold_total_entities': total_gold_entities,
        'gold_total_categories': len(total_gold_categories),
        'centroid_categories': len(category_centroids),
        'has_root_centroid': root_centroid is not None
    }

    if is_verbose:
        print("\n[DEBUG] Recomputed centroids from GOLD data:")
        print(f"  GOLD trees: {stats['gold_trees']}")
        print(f"  GOLD entities: {stats['gold_total_entities']}")
        print(f"  GOLD categories (unique): {stats['gold_total_categories']}")
        print(f"  Centroid categories built: {stats['centroid_categories']}")
        print(f"  Root centroid available: {stats['has_root_centroid']}")

    return category_centroids, root_centroid, stats



def compute_centroids(entity_embeddings, entity_qids, category_entities):
    """
    Compute centroid embeddings for categories based on their entities
    
    Args:
        entity_embeddings: dict mapping qid to embedding tensor [num_layers, hidden_dim]
        entity_qids: list of all entity qids
        category_entities: dict mapping category qid to list of entity qids
    
    Returns:
        dict mapping category qid to centroid embedding
    """
    centroids = {}
    
    for category_qid, entity_list in category_entities.items():
        # Collect embeddings for entities in this category
        entity_embeds = []
        for entity_qid in entity_list:
            if entity_qid in entity_embeddings:
                entity_embeds.append(entity_embeddings[entity_qid])
        
        if entity_embeds:
            # Compute mean across entities
            centroid = torch.stack(entity_embeds).mean(dim=0)
            centroids[category_qid] = centroid
    
    # Compute root centroid (average of ALL entities)
    all_entity_embeds = [entity_embeddings[qid] for qid in entity_qids if qid in entity_embeddings]
    if all_entity_embeds:
        root_centroid = torch.stack(all_entity_embeds).mean(dim=0)
    else:
        root_centroid = None
    
    return centroids, root_centroid


def split_taxonomy_tree(tree_structure, node_classification, num_splits=None, random_seed=42):
    """
    Split a taxonomy tree into multiple trees with one entity per category
    
    Args:
        tree_structure: original tree structure
        node_classification: classification of nodes
        num_splits: number of splits to generate (None = all combinations for small trees)
        random_seed: random seed for sampling
    
    Returns:
        list of split tree structures
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    category_entities = node_classification['category_entities']
    
    # Get all possible combinations
    all_combinations = []
    categories = list(category_entities.keys())
    
    if all(len(category_entities[cat]) > 0 for cat in categories):
        # Generate combinations
        entity_lists = [category_entities[cat] for cat in categories]
        
        # Check total number of combinations
        total_combinations = 1
        for entity_list in entity_lists:
            total_combinations *= len(entity_list)
        
        print(f"Total possible combinations: {total_combinations}")
        if total_combinations <= 10000 and num_splits is None:
            # Generate all combinations for small trees
            all_combinations = list(itertools.product(*entity_lists))
        else:
            # Sample random combinations for large trees (without replacement)
            if num_splits is None:
                num_splits = min(1000, total_combinations)
            # Clip num_splits so it does not exceed the total number of combinations
            num_splits = min(num_splits, total_combinations)

            # Candidate counts (radices) per category
            bases = [len(lst) for lst in entity_lists]

            # multipliers[i] = product of bases[i+1:]
            multipliers = [1] * len(bases)
            for i in range(len(bases) - 2, -1, -1):
                multipliers[i] = multipliers[i + 1] * bases[i + 1]

            def index_to_combination(idx: int):
                combo = []
                for base, mult, choices in zip(bases, multipliers, entity_lists):
                    pos = (idx // mult) % base
                    combo.append(choices[pos])
                return tuple(combo)

            # Sample indices without replacement
            sampled_indices = random.sample(range(total_combinations), num_splits)
            all_combinations = [index_to_combination(i) for i in sampled_indices]

    
    # Create split trees
    split_trees = []
    
    for i, combination in enumerate(all_combinations):
        # Build new tree with selected entities
        selected_entities = set(combination)
        new_nodes = {}
        new_edges = []
        
        # Include root, categories, and selected entities
        for qid, node_info in tree_structure['nodes'].items():
            if (qid in node_classification['root'] or 
                qid in node_classification['categories'] or 
                qid in selected_entities):
                new_nodes[qid] = node_info.copy()
        
        # Include relevant edges
        for parent, child in tree_structure['edges']:
            if parent in new_nodes and child in new_nodes:
                new_edges.append((parent, child))
        
        split_tree = {
            'nodes': new_nodes,
            'edges': new_edges,
            'split_id': i,
            'selected_entities': list(selected_entities)
        }
        split_trees.append(split_tree)
    
    return split_trees

def build_final_embeddings(tree_structure, node_classification, entity_embeddings, 
                          category_centroids, root_centroid, selected_entities=None,
                          missing_category_policy='error'):
    """
    Build final embedding tensor for a tree using entities and (precomputed) centroids.

    Args:
        tree_structure: tree structure
        node_classification: node classification
        entity_embeddings: dict of entity embeddings
        category_centroids: dict of category centroids (from GOLD/train)
        root_centroid: root centroid (from GOLD/train)
        selected_entities: optional entity subset (for split trees)
        missing_category_policy: 'error' | 'root' | 'skip'

    Returns:
        (embeddings tensor, ordered node list)
    """
    ordered_nodes = []
    ordered_embeddings = []

    # Decide which nodes to include
    nodes_to_include = set()
    nodes_to_include.update(node_classification['root'])
    nodes_to_include.update(node_classification['categories'])

    if selected_entities is not None:
        nodes_to_include.update(selected_entities)
    else:
        nodes_to_include.update(node_classification['entities'])

    # Iterate with deterministic order
    for qid in sorted(nodes_to_include):
        if qid not in tree_structure['nodes']:
            continue

        # Root nodes
        if qid in node_classification['root']:
            if root_centroid is None:
                raise ValueError("Root centroid is None; cannot assign root embedding.")
            ordered_nodes.append(qid)
            ordered_embeddings.append(root_centroid)
            continue

        # Category nodes (use precomputed centroids from GOLD/train)
        if qid in node_classification['categories']:
            if qid in category_centroids:
                ordered_nodes.append(qid)
                ordered_embeddings.append(category_centroids[qid])
            else:
                # Handle missing categories according to policy
                if missing_category_policy == 'error':
                    raise ValueError(f"Missing centroid for category {qid} in test mode.")
                elif missing_category_policy == 'root':
                    if root_centroid is None:
                        raise ValueError(f"Missing centroid for category {qid} and root centroid is None.")
                    ordered_nodes.append(qid)
                    ordered_embeddings.append(root_centroid)
                elif missing_category_policy == 'skip':
                    # Do not append this category
                    continue
                else:
                    raise ValueError(f"Unknown missing_category_policy: {missing_category_policy}")
            continue

        # Entity nodes (use test entity embeddings)
        if qid in entity_embeddings:
            ordered_nodes.append(qid)
            ordered_embeddings.append(entity_embeddings[qid])

    # Stack embeddings
    if ordered_embeddings:
        final_embeddings = torch.stack(ordered_embeddings)
    else:
        final_embeddings = torch.empty(0)

    return final_embeddings, ordered_nodes



def save_embeddings(embeddings_tensor, metadata, output_path):
    """Save embeddings and metadata using torch.save"""
    data = {
        'embeddings': embeddings_tensor,
        'metadata': metadata,
        'embedding_shape': tuple(embeddings_tensor.shape),
        'torch_version': torch.__version__
    }
    
    # Use torch.save for efficient tensor storage
    torch.save(data, output_path)
    
    print(f"✓ Embeddings saved to {output_path}")
    print(f"  Tensor shape: {embeddings_tensor.shape}")
    print(f"  Tensor dtype: {embeddings_tensor.dtype}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def main():
    args = parse_args()

    # Device configuration
    if args.device == 'auto':
        device = torch.device('cuda' if (torch.cuda.is_available() and args.model_type == 'hf') else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device} (model_type={args.model_type})")

    # FastText validation
    if args.model_type == 'fasttext' and args.text_mode != 'title':
        print("[ERROR] fastText only supports text_mode='title'")
        sys.exit(1)

    # Output directory
    output_root = os.path.join(args.output_dir, args.text_mode)
    os.makedirs(output_root, exist_ok=True)
    print(f"Embeddings will be saved under: {output_root}")

    # Load model/tokenizer ONCE (shared by train/test and centroid recomputation)
    tokenizer, model = load_model_and_tokenizer(args, device)

    # ---------------------------
    # RUN MODE: TRAIN (unchanged)
    # ---------------------------
    if args.run_mode == 'train':
        # Load data
        tree_groups = load_data(args.data_path)

        # Process each tree
        for tree_id, tree_data in tree_groups.items():
            print(f"\n{'='*20} Processing Tree {tree_id} {'='*20}")

            # Build structure
            tree_structure = build_taxonomy_structure(tree_data, args.text_mode, args.is_verbose)

            # Classify nodes
            node_classification = classify_nodes(tree_structure)

            print(f"Tree structure:")
            print(f"  Root nodes: {len(node_classification['root'])}")
            print(f"  Category nodes: {len(node_classification['categories'])}")
            print(f"  Entity nodes: {len(node_classification['entities'])}")

            # Phase 1: entity embeddings
            print("\nPhase 1: Extracting entity embeddings...")
            entity_texts, entity_qids = [], []
            for qid in node_classification['entities']:
                if qid in tree_structure['nodes']:
                    entity_texts.append(build_input_text(tree_structure['nodes'][qid], args.text_mode))
                    entity_qids.append(qid)

            if entity_texts:
                entity_tensor = extract_hidden_states(entity_texts, args=args, tokenizer=tokenizer, model=model, device=device)
                entity_embeddings = {qid: entity_tensor[i] for i, qid in enumerate(entity_qids)}
            else:
                entity_embeddings = {}

            # Phase 2: category/root centroids (train-side)
            print("\nPhase 2: Computing category centroids...")
            category_centroids, root_centroid = compute_centroids(
                entity_embeddings,
                entity_qids,
                node_classification['category_entities']
            )
            print(f"  Computed centroids for {len(category_centroids)} categories")

            # Phase 3: split or complete
            if args.split_tree:
                print("\nPhase 3: Splitting tree...")
                split_trees = split_taxonomy_tree(
                    tree_structure,
                    node_classification,
                    num_splits=args.num_splits,
                    random_seed=args.random_seed
                )
                print(f"  Generated {len(split_trees)} tree splits")

                for split_tree in split_trees:
                    split_id = split_tree['split_id']
                    print(f"\n  Processing split {split_id}...")

                    embeddings, ordered_nodes = build_final_embeddings(
                        split_tree,
                        node_classification,
                        entity_embeddings,
                        category_centroids,
                        root_centroid,
                        selected_entities=set(split_tree['selected_entities']),
                        missing_category_policy='error'  # train side shouldn't miss
                    )

                    distances, _ = calculate_tree_distances(split_tree)
                    metadata = {
                        'tree_id': tree_id,
                        'split_id': split_id,
                        'nodes': ordered_nodes,
                        'node_titles': [split_tree['nodes'][n]['title'] for n in ordered_nodes],
                        'selected_entities': split_tree['selected_entities'],
                        'text_mode': args.text_mode,
                        'tree_structure': split_tree,
                        'distances': distances,
                        'model_type': args.model_type,
                        'model_name': args.model_name if args.model_type == 'hf' else args.vector_path,
                        'num_layers': embeddings.shape[1],
                        'used_centroids': True,
                        'full_category_entities': node_classification['category_entities'],
                        'run_mode': 'train'
                    }
                    out_path = os.path.join(output_root, f'tree_{tree_id}_split_{split_id}_embeddings.pt')
                    save_embeddings(embeddings, metadata, out_path)

            else:
                print("\nPhase 3: Building complete tree embeddings...")
                embeddings, ordered_nodes = build_final_embeddings(
                    tree_structure,
                    node_classification,
                    entity_embeddings,
                    category_centroids,
                    root_centroid,
                    missing_category_policy='error'
                )

                distances, _ = calculate_tree_distances(tree_structure)
                metadata = {
                    'tree_id': tree_id,
                    'nodes': ordered_nodes,
                    'node_titles': [tree_structure['nodes'][n]['title'] for n in ordered_nodes],
                    'text_mode': args.text_mode,
                    'tree_structure': tree_structure,
                    'distances': distances,
                    'model_type': args.model_type,
                    'model_name': args.model_name if args.model_type == 'hf' else args.vector_path,
                    'num_layers': embeddings.shape[1],
                    'used_centroids': True,
                    'node_classification': node_classification,
                    'run_mode': 'train'
                }
                out_path = os.path.join(output_root, f'tree_{tree_id}_embeddings.pt')
                save_embeddings(embeddings, metadata, out_path)

        print(f"\n✓ All embeddings extracted and saved into {output_root}")
        return

    # -------------------------
    # RUN MODE: TEST (new path)
    # -------------------------
    if args.run_mode == 'test':
        # Force-disable split in test mode
        if args.split_tree:
            print("[INFO] --split_tree is ignored in test mode.")

        # Validate centroid_data_path
        if args.centroid_data_path is None:
            print("[ERROR] In test mode, --centroid_data_path must be provided.")
            sys.exit(1)

        # Load TEST data
        test_groups = load_data(args.data_path)

        # Recompute centroids from GOLD (train) data on-the-fly
        print("\n[TEST] Recomputing centroids from GOLD data...")
        pre_cat_centroids, pre_root_centroid, gold_stats = recompute_centroids_from_data(
            args.centroid_data_path, args, tokenizer, model, device, is_verbose=args.is_verbose
        )

        # Process each TEST tree
        for tree_id, tree_data in test_groups.items():
            print(f"\n{'='*20} Processing TEST Tree {tree_id} {'='*20}")

            # Build structure and classify
            test_struct = build_taxonomy_structure(tree_data, args.text_mode, args.is_verbose)
            test_cls = classify_nodes(test_struct)

            print(f"TEST Tree structure:")
            print(f"  Root nodes: {len(test_cls['root'])}")
            print(f"  Category nodes: {len(test_cls['categories'])}")
            print(f"  Entity nodes: {len(test_cls['entities'])}")

            # Extract TEST entity embeddings
            print("\n[TEST] Phase 1: Extracting TEST entity embeddings...")
            test_entity_texts, test_entity_qids = [], []
            for qid in test_cls['entities']:
                if qid in test_struct['nodes']:
                    test_entity_texts.append(build_input_text(test_struct['nodes'][qid], args.text_mode))
                    test_entity_qids.append(qid)

            if test_entity_texts:
                test_entity_tensor = extract_hidden_states(
                    test_entity_texts, args=args, tokenizer=tokenizer, model=model, device=device
                )
                test_entity_embeddings = {qid: test_entity_tensor[i] for i, qid in enumerate(test_entity_qids)}
            else:
                test_entity_embeddings = {}

            # Coverage inspection for categories
            test_cats = set(test_cls['categories'])
            available_cats = set(pre_cat_centroids.keys())
            missing_cats = sorted(list(test_cats - available_cats))
            found_cats = len(test_cats) - len(missing_cats)

            if missing_cats:
                action = args.missing_category_policy
                print(f"[TEST] Missing category centroids: {len(missing_cats)} (policy={action})")
                if action == 'error':
                    # Fail fast with details
                    print("[ERROR] The following categories are missing from GOLD centroids:")
                    for q in missing_cats[:20]:
                        title = test_struct['nodes'][q]['title'] if q in test_struct['nodes'] else q
                        print(f"  - {title} ({q})")
                    if len(missing_cats) > 20:
                        print(f"  ... and {len(missing_cats)-20} more")
                    sys.exit(1)
                # For 'root' and 'skip', build_final_embeddings() will handle assignment/skip.

            # Build final embeddings (categories/root use precomputed from GOLD)
            print("\n[TEST] Phase 2: Building final TEST embeddings...")
            embeddings, ordered_nodes = build_final_embeddings(
                test_struct,
                test_cls,
                test_entity_embeddings,
                pre_cat_centroids,
                pre_root_centroid,
                selected_entities=None,
                missing_category_policy=args.missing_category_policy
            )

            # Distances on TEST tree (unchanged)
            distances, _ = calculate_tree_distances(test_struct)

            # Metadata with provenance
            metadata = {
                'tree_id': tree_id,
                'nodes': ordered_nodes,
                'node_titles': [test_struct['nodes'][n]['title'] for n in ordered_nodes],
                'text_mode': args.text_mode,
                'tree_structure': test_struct,
                'distances': distances,
                'model_type': args.model_type,
                'model_name': args.model_name if args.model_type == 'hf' else args.vector_path,
                'num_layers': embeddings.shape[1],
                'used_centroids': True,
                'centroid_source': 'recomputed_from_centroid_data',
                'centroid_data_path': args.centroid_data_path,
                'centroid_coverage': {
                    'test_categories': len(test_cats),
                    'found': found_cats,
                    'missing': len(missing_cats),
                    'policy': args.missing_category_policy,
                    'missing_list': missing_cats
                },
                'gold_stats': gold_stats,
                'run_mode': 'test'
            }

            out_path = os.path.join(output_root, f'tree_{tree_id}_test_embeddings.pt')
            save_embeddings(embeddings, metadata, out_path)

        print(f"\n✓ All TEST embeddings extracted and saved into {output_root}")

if __name__ == "__main__":
    main()