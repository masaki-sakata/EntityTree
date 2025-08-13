#!/usr/bin/env python
# =============================================================================
# visualize_tree.py (Template-enabled Version with Verbose Debug)
# =============================================================================
"""
Build Tree HTML visualizations with template-based text generation.

New features:
- Template system for generating contextual text from entity names
- Support for various question formats and professional contexts
- Maintains profession-based coloring and PNG export capabilities
- Verbose mode for debugging token usage
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
import template  # Import our new template module
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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
    p = argparse.ArgumentParser(description="Build Tree and export HTML/PNG with templates")
    p.add_argument("--input", required=True, help="Input JSONL file")
    p.add_argument("--output_dir", required=True, help="Output directory for HTML/PNG files")
    p.add_argument("--output_file_name", required=True, help="Output file name (without extension)")
    p.add_argument("--export_png", action="store_true", help="Export PNG files (requires selenium)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose debug output for token inspection")

    # Template params ----------------------------------------------------------
    p.add_argument("--template", type=str, default="entity_only",
                   help=f"Template to use for text generation. Available: {', '.join(template.list_templates())}")
    p.add_argument("--list_templates", action="store_true",
                   help="List all available templates and exit")

    # Embedding params --------------------------------------------------------
    p.add_argument(
        "--model",
        choices=["gpt2", "meta-llama/Meta-Llama-3-8B", "fasttext", "random_emb"],
        default="gpt2",
    )
    p.add_argument("--method", choices=["average", "last_token"], default="last_token")
    p.add_argument(
        "--layer",
        default="all",
        help='Transformer hidden layer index (0-based). Use "all" for every layer.',
    )
    p.add_argument("--device", default="cuda")

    # Random embedding specific params ----------------------------------------
    p.add_argument("--random_dim", type=int, default=768, 
                   help="Dimension for random embeddings (only used with model=random_emb)")
    p.add_argument("--random_std", type=float, default=1.0,
                   help="Standard deviation for random embeddings (only used with model=random_emb)")
    p.add_argument("--random_seed", type=int, default=42,
                   help="Random seed for reproducible embeddings (only used with model=random_emb)")

    # PCA plots ---------------------------------------------------------------
    p.add_argument("--pca", action="store_true",
                   help="Also export PCA plots (2D PNG and 3D HTML)")
    p.add_argument("--pca_label2d", action="store_true",
                   help="Annotate labels in 2D PCA plot (may clutter)")

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


def get_node_colors(entity_names: list[str], profession_map: Dict[str, str]) -> Dict[int, str]:
    """Map node indices to colors based on profession."""
    colors = {}
    for idx, entity_name in enumerate(entity_names):
        profession = profession_map.get(entity_name, "Person")
        colors[idx] = PROFESSION_COLORS.get(profession, "#CCCCCC")  # Default gray
    return colors


def apply_template_to_entities(entity_names: list[str], template_name: str) -> tuple[list[str], str]:
    """Apply template to all entity names and return templated texts."""
    template_str = template.get_template(template_name)
    templated_texts = []
    
    for entity_name in entity_names:
        templated_text = template.apply_template(template_str, entity_name)
        templated_texts.append(templated_text)
    
    return templated_texts, template_str


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


def _compute_pca(X: np.ndarray, max_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    PCAで主成分に射影した座標と寄与率を返す。
    次元数/サンプル数に応じて自動で成分数を調整。
    """
    n_samples, n_features = X.shape
    n_components = min(max_components, n_samples, n_features)
    pca = PCA(n_components=n_components, random_state=0)
    comps = pca.fit_transform(X)  # (N, n_components)
    evr = pca.explained_variance_ratio_  # (n_components,)
    return comps, evr


def _group_indices_by_profession(entity_names: list[str], profession_map: Dict[str, str]) -> Dict[str, list[int]]:
    groups: Dict[str, list[int]] = {}
    for i, name in enumerate(entity_names):
        prof = profession_map.get(name, "Person")
        groups.setdefault(prof, []).append(i)
    return groups


def _save_pca_2d_png(
    comps: np.ndarray,
    evr: np.ndarray,
    entity_names: list[str],
    profession_map: Dict[str, str],
    base_out: Path,
    layer_idx: int,
    file_name: str,
    annotate: bool = False,
) -> Path:
    """
    職業ごとに色分けして2D散布図をPNG出力。
    """
    # 2成分に満たない場合はゼロ埋め
    if comps.shape[1] < 2:
        pad = np.zeros((comps.shape[0], 2 - comps.shape[1]))
        comps2 = np.hstack([comps, pad])
    else:
        comps2 = comps[:, :2]

    x, y = comps2[:, 0], comps2[:, 1]

    groups = _group_indices_by_profession(entity_names, profession_map)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    for prof, idxs in groups.items():
        color = PROFESSION_COLORS.get(prof, "#CCCCCC")
        ax.scatter(x[idxs], y[idxs], s=28, alpha=0.9, linewidths=0, label=prof, c=color)

    if annotate:
        for i, name in enumerate(entity_names):
            ax.annotate(name, (x[i], y[i]), fontsize=7, alpha=0.8)

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({(evr[1]*100 if len(evr)>1 else 0):.1f}%)")
    ax.set_title("PCA 2D")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=8, frameon=False, ncols=2)
    plt.tight_layout()

    png_path = build_output_path(base_out, layer_idx, f"{file_name}_pca2d", "png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"[PCA] 2D PNG exported: {png_path}")
    return png_path


def _save_pca_3d_html(
    comps: np.ndarray,
    evr: np.ndarray,
    entity_names: list[str],
    profession_map: Dict[str, str],
    base_out: Path,
    layer_idx: int,
    file_name: str,
) -> Path:
    """
    職業ごとにトレースを分けた3D散布図をPlotlyでHTML出力。
    """
    # 3成分に満たない場合はゼロ埋め
    if comps.shape[1] < 3:
        pad = np.zeros((comps.shape[0], 3 - comps.shape[1]))
        comps3 = np.hstack([comps, pad])
    else:
        comps3 = comps[:, :3]

    x, y, z = comps3[:, 0], comps3[:, 1], comps3[:, 2]

    groups = _group_indices_by_profession(entity_names, profession_map)
    traces = []
    for prof, idxs in groups.items():
        color = PROFESSION_COLORS.get(prof, "#CCCCCC")
        hover_text = [
            f"<b>{entity_names[i]}</b><br>Profession: {prof}<br>"
            f"PC1: {x[i]:.3f}<br>PC2: {y[i]:.3f}<br>PC3: {z[i]:.3f}"
            for i in idxs
        ]
        traces.append(
            go.Scatter3d(
                x=x[idxs], y=y[idxs], z=z[idxs],
                mode="markers",
                name=prof,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=5, color=color, opacity=0.9),
            )
        )

    title = "PCA 3D — EVR: [" + ", ".join([f"{v*100:.1f}%" for v in evr[:3]]) + "]"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({evr[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({(evr[1]*100 if len(evr)>1 else 0):.1f}%)",
            zaxis_title=f"PC3 ({(evr[2]*100 if len(evr)>2 else 0):.1f}%)",
        ),
        legend=dict(itemsizing="trace", font=dict(size=10)),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    html_path = build_output_path(base_out, layer_idx, f"{file_name}_pca3d", "html")
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    print(f"[PCA] 3D HTML exported: {html_path}")
    return html_path


# --------------------------------------------------------------------------- #
# Embedding utility                                                           #
# --------------------------------------------------------------------------- #

def _encode_sentences(
    sentences: Sequence[str],
    entity_names: Sequence[str],  # 必須引数として定義
    model_type: str,
    method: str,
    layer: str | int,
    device: str,
    verbose: bool = False,
    # Random embedding parameters
    random_dim: int = 768,
    random_std: float = 1.0,
    random_seed: int = 42,
) -> tuple[np.ndarray, Sequence[int]]:
    """Embed sentences and return (embeddings, target_layers)."""
    
    if len(sentences) != len(entity_names):
        raise ValueError(f"sentences と entity_names の長さが一致しません: {len(sentences)} != {len(entity_names)}")
    
    # Handle random embeddings through EmbeddingModel
    if model_type == "random_emb":
        cfg = EmbeddingConfig(
            model_type=model_type,
            method=method,
            layer=0,  # Random embeddings are always single layer (layer 0)
            device=device,
            verbose=verbose,
            random_dim=random_dim,
            random_std=random_std,
            random_seed=random_seed,
        )
        embedder = EmbeddingModel(cfg)
        embs = embedder.encode(list(sentences), list(entity_names))
        embs = embs[None]  # → (1, N, D)
        target_layers = [0]
        
        return embs, target_layers
    
    # Handle other model types (existing code)
    cfg = EmbeddingConfig(
        model_type=model_type,
        method=method,
        layer=layer,
        device=device,
        verbose=verbose,
    )
    embedder = EmbeddingModel(cfg)
    embs = embedder.encode(list(sentences), list(entity_names))

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
    # Handle list templates request
    if args.list_templates:
        print("Available templates:")
        print("=" * 50)
        template.show_template_examples()
        return

    # 1) Load & preprocess ----------------------------------------------------
    input_path = Path(args.input)
    
    # Load the JSONL file into a DataFrame
    df = pd.read_json(str(input_path), lines=True)
    
    # Extract profession mapping for all entities and categories
    profession_map = extract_profession_mapping(df)
    
    # Filter entity rows and extract entity names
    entity_df = df[df["is_entity"] == True]
    entity_names = entity_df["wiki_title"].fillna("").tolist()
    
    # Apply template to entity names to create actual input texts
    templated_texts, template_str = apply_template_to_entities(entity_names, args.template)
    
    # Get color mapping for nodes (based on original entity names)
    node_colors = get_node_colors(entity_names, profession_map)

    print(f"Loaded {len(entity_names)} entities from {len(df)} total records")
    print(f"Template used: '{args.template}' -> '{template_str}'")
    print(f"Example templated text: '{templated_texts[0] if templated_texts else 'N/A'}'")
    print(f"Profession distribution: {pd.Series([profession_map.get(s, 'Unknown') for s in entity_names]).value_counts().to_dict()}")

    if args.verbose:
        print("\n" + "="*60)
        print("VERBOSE MODE: Debugging token usage...")
        print("="*60)

    # 2) Embed templated sentences (single forward pass when possible) -------
    all_embs, target_layers = _encode_sentences(
        sentences=templated_texts,  
        entity_names=entity_names,  
        model_type=args.model,
        method=args.method,
        layer=args.layer,
        device=args.device,
        verbose=args.verbose,
        random_dim=args.random_dim,
        random_std=args.random_std,
        random_seed=args.random_seed,
    )

    if args.verbose:
        print("="*60)
        print("Token debugging complete.")
        print("="*60 + "\n")

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
        # trimming_summary, _, important = summarization.get_hierarchy_summary_ids(embs)
        # kcenter_summary = summarization.get_k_center_summary_ids(
        #     summary_length=len(trimming_summary),
        #     embs=embs,
        # )
        # highlights = None  # set(trimming_summary) | set(kcenter_summary) | set(important)

        # Create title with model, layer, and template info
        if args.model == "random_emb":
            model_display = f"Random Emb (dim={args.random_dim}, std={args.random_std}, seed={args.random_seed})"
        else:
            model_display = args.model.split("/")[-1] if "/" in args.model else args.model
        
        title = f"{model_display} - Layer {l_idx} - Template: {args.template}"

        # Encode tree to HTML with enhanced options
        # Use original entity names for labels, but embeddings come from templated texts
        tree_encoder = TreeEncoding(
            adjacency=adjacency,
            births=hierarchy.birth_time,
            n_leaves=n_leaves,
            n_nodes=n_nodes,
            highlights=None,
            labels={idx: entity_name for idx, entity_name in enumerate(entity_names)},  # Show original entity names
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
            
        # --- PCA plots --------------------------------------------------------
        if args.pca:
            try:
                comps, evr = _compute_pca(embs, max_components=3)

                _save_pca_2d_png(
                    comps=comps,
                    evr=evr,
                    entity_names=entity_names,
                    profession_map=profession_map,
                    base_out=base_out,
                    layer_idx=l_idx,
                    file_name=args.output_file_name,
                    annotate=args.pca_label2d,
                )

                _save_pca_3d_html(
                    comps=comps,
                    evr=evr,
                    entity_names=entity_names,
                    profession_map=profession_map,
                    base_out=base_out,
                    layer_idx=l_idx,
                    file_name=args.output_file_name,
                )
            except Exception as e:
                print(f"[PCA] Error while generating PCA plots: {e}")



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
    
    # Validate random embedding parameters
    if args.model == "random_emb":
        print(f"Using random embeddings with:")
        print(f"  Dimension: {args.random_dim}")
        print(f"  Standard deviation: {args.random_std}")
        print(f"  Random seed: {args.random_seed}")
        print(f"  Layer: 0 (single layer only)")
    
    print("Arguments:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    run(args)