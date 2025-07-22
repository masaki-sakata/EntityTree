#!/usr/bin/env python
# =============================================================================
# visualize_tree.py (Improved Version)
# =============================================================================
"""
Build Tree HTML visualizations with profession-based coloring and PNG export.

New features:
- Display model name and layer number on HTML
- Color entities by profession (parent node category)
- Export to PNG format in addition to HTML
- Larger font sizes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Dict, Optional
import pandas as pd

import numpy as np

import util
from embeddings import EmbeddingConfig, EmbeddingModel
from hierarchy_node import HierarchyNode
from html_tree_encoding import HTMLTreeEncoding as TreeEncoding
import summarization

# Optional dependencies for PNG export
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

# Profession color mapping
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
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Tree and export HTML/PNG")
    p.add_argument("--input", required=True, help="Input JSONL file")
    p.add_argument("--output_dir", required=True, help="Output directory for HTML/PNG files")
    p.add_argument("--output_file_name", required=True, help="Output file name (without extension)")
    p.add_argument("--export_png", action="store_true", help="Export PNG files (requires selenium)")

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
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #

def build_output_path(base: Path, layer_idx: int | None, file_name: str, extension: str = "html") -> Path:
    """Generate an output file path for the given layer index."""
    if base.suffix.lower() == f".{extension}":
        # User passed a file path directly → reuse as‑is
        return base

    # Directory was supplied → create per‑layer sub‑directory
    layer_dir = base / f"layer{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    return layer_dir / f"{file_name}.{extension}"


def extract_profession_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Extract profession mapping for each entity from JSONL data."""
    profession_map = {}
    
    for _, row in df.iterrows():
        if row['is_entity'] and 'edges' in row and row['edges']:
            entity_name = row['wiki_title']
            # Get the profession from the first edge (should be the parent category)
            profession = row['edges'][0]['target_label']
            profession_map[entity_name] = profession
        elif not row['is_entity']:
            # This is a category node
            category_name = row['wiki_title']
            profession_map[category_name] = category_name
    
    return profession_map


def get_node_colors(sentences: list[str], profession_map: Dict[str, str]) -> Dict[int, str]:
    """Map node indices to colors based on profession."""
    colors = {}
    for idx, sentence in enumerate(sentences):
        profession = profession_map.get(sentence, "Person")
        colors[idx] = PROFESSION_COLORS.get(profession, "#CCCCCC")  # Default gray
    return colors


def export_to_png(html_path: Path, png_path: Path) -> bool:
    """Export HTML file to PNG using selenium."""
    if not HAS_SELENIUM:
        print("Warning: selenium not available. Skipping PNG export.")
        print("Install with: pip install selenium")
        return False
    
    try:
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Initialize webdriver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Load HTML file
        driver.get(f"file://{html_path.absolute()}")
        
        # Wait for the network to render
        time.sleep(3)
        
        # Take screenshot
        driver.save_screenshot(str(png_path))
        driver.quit()
        
        print(f"PNG exported: {png_path}")
        return True
        
    except Exception as e:
        print(f"Error exporting PNG: {e}")
        if 'driver' in locals():
            driver.quit()
        return False


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
    """Embed sentences and return (embeddings, target_layers)."""
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
    # 1) Load & preprocess ----------------------------------------------------
    input_path = Path(args.input)
    
    # Load the JSONL file into a DataFrame
    df = pd.read_json(str(input_path), lines=True)
    
    # Extract profession mapping for all entities and categories
    profession_map = extract_profession_mapping(df)
    
    # Filter entity rows and extract sentences
    entity_df = df[df["is_entity"] == True]
    sentences = entity_df["wiki_title"].fillna("").tolist()
    
    # Get color mapping for nodes
    node_colors = get_node_colors(sentences, profession_map)

    print(f"Loaded {len(sentences)} entities from {len(df)} total records")
    print(f"Profession distribution: {pd.Series([profession_map.get(s, 'Unknown') for s in sentences]).value_counts().to_dict()}")

    # 2) Embed sentences (single forward pass when possible) -----------------
    all_embs, target_layers = _encode_sentences(
        sentences=sentences,
        model_type=args.model,
        method=args.method,
        layer=args.layer,
        device=args.device,
    )

    base_out = Path(args.output_dir)

    # 3) Build & export Tree per layer ---------------------------------------
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

        # Create title with model and layer info
        model_display = args.model.split("/")[-1] if "/" in args.model else args.model
        title = f"{model_display} - Layer {l_idx} - Hierarchical Clustering"

        # Encode tree to HTML with enhanced options
        tree_encoder = TreeEncoding(
            adjacency=adjacency,
            births=hierarchy.birth_time,
            n_leaves=n_leaves,
            n_nodes=n_nodes,
            highlights=highlights,
            labels={idx: sent for idx, sent in enumerate(sentences)},
            node_colors=node_colors,
            title=title,
            height_px=1000,
            width_pct=100,
            font_size=16  # Larger font size
        )

        # 4) Output HTML ---------------------------------------------------------
        html_path = build_output_path(base_out, l_idx, args.output_file_name, "html")
        tree_encoder.draw(str(html_path))
        print(f"[DONE] layer {l_idx}: wrote {html_path}")

        # 5) Export PNG if requested --------------------------------------------
        if args.export_png:
            png_path = build_output_path(base_out, l_idx, args.output_file_name, "png")
            export_to_png(html_path, png_path)


# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Check PNG export dependencies
    if args.export_png and not HAS_SELENIUM:
        print("Warning: --export_png specified but selenium not available.")
        print("Install with: pip install selenium")
        print("Also ensure you have Chrome/Chromium and chromedriver installed.")
    
    print("Arguments:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    run(args)