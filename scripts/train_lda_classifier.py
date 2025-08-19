#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

uv run python3 train_lda_classifier.py \
  --input /home/masaki/hierarchical-repr/EntityTree/input/300people/train250.jsonl \
  --outdir ../models/lda \
  --model meta-llama/Meta-Llama-3-8B \
  --method last_token \
  --template entity_only \
  --layer all \
  --device cuda \
  --solver svd \
  --n_components 5 \
  --shrinkage none \
  --verbose

"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Sequence
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import sklearn
import joblib

from embeddings import EmbeddingConfig, EmbeddingModel
import template

ALLOWED_CLASSES = {
    "Politician", "Actor", "Athlete", "Musician", "Scientist", "Business Person",
}

def vprint(enabled: bool, *args, **kwargs):
    if enabled: print(*args, **kwargs)

def choose_label_from_edges(edges: list, allowed: set) -> tuple[str | None, bool]:
    if not edges: return None, False
    first = edges[0].get("target_label")
    if first in allowed: return first, False
    for e in edges[1:]:
        cand = e.get("target_label")
        if cand in allowed: return cand, True
    return None, False

def load_entities_and_labels(jsonl_path: Path, verbose: bool=False) -> tuple[list[str], list[str], dict]:
    df = pd.read_json(str(jsonl_path), lines=True)
    total_rows = len(df)
    ent_df = df[df["is_entity"] == True].copy()

    entity_names, labels = [], []
    counts = {"missing_edges": 0, "with_edges": 0, "selected": 0, "fallback_used": 0}
    invalid_first_edge = Counter()
    mapping_examples, fallback_examples = [], []

    for _, row in ent_df.iterrows():
        name = row.get("wiki_title", "")
        edges = row.get("edges", [])
        if not name: continue
        if not edges:
            counts["missing_edges"] += 1
            continue
        counts["with_edges"] += 1
        first = edges[0].get("target_label")
        if first and first not in ALLOWED_CLASSES:
            invalid_first_edge[first] += 1
        label, used_fallback = choose_label_from_edges(edges, ALLOWED_CLASSES)
        if label is None:  # 非対象クラス
            continue
        if used_fallback and len(fallback_examples) < 5:
            counts["fallback_used"] += 1
            fallback_examples.append((name, label))
        entity_names.append(str(name))
        labels.append(str(label))
        if len(mapping_examples) < 10:
            mapping_examples.append((name, label))

    class_counts = Counter(labels)
    dbg = {
        "total_rows": total_rows,
        "num_entities": int(ent_df.shape[0]),
        "num_categories": int(total_rows - ent_df.shape[0]),
        "counts": counts,
        "invalid_first_edge_top": invalid_first_edge.most_common(10),
        "class_counts": dict(class_counts),
        "mapping_examples": mapping_examples,
        "fallback_examples": fallback_examples,
    }

    if verbose:
        vprint(True, "[DATA] total rows:", total_rows)
        vprint(True, "[DATA] is_entity=True:", dbg["num_entities"], "| is_entity=False:", dbg["num_categories"])
        vprint(True, "[DATA] missing_edges:", counts["missing_edges"], "| with_edges:", counts["with_edges"])
        vprint(True, "[FILTER] selected entities:", len(entity_names), "| fallback_used:", counts["fallback_used"])
        vprint(True, "[CLASS] distribution:", dict(class_counts))
        if mapping_examples:
            vprint(True, "[SAMPLE] entity -> class (first 10):")
            for n, c in mapping_examples: vprint(True, f"  - {n} -> {c}")

    if not entity_names:
        raise RuntimeError("is_entity=True かつ 6クラスに該当する行が見つかりませんでした。")

    return entity_names, labels, dbg

def apply_template_to_entities(entity_names: Sequence[str], template_name: str, verbose: bool=False) -> list[str]:
    t = template.get_template(template_name)
    templated = [template.apply_template(t, n) for n in entity_names]
    if verbose and templated:
        vprint(True, f"[TEMPLATE] '{template_name}': example -> {templated[0]!r}")
    return templated

def build_embedder(model_type: str, method: str, layer: str, device: str, verbose: bool) -> EmbeddingModel:
    cfg = EmbeddingConfig(model_type=model_type, method=method, layer=layer, device=device, verbose=verbose)
    return EmbeddingModel(cfg)

def compute_embeddings(embedder: EmbeddingModel, texts: Sequence[str], entity_names: Sequence[str], verbose: bool=False):
    embs = embedder.encode(list(texts), list(entity_names))
    if verbose:
        vprint(True, "[EMB] raw type:", type(embs))
        if isinstance(embs, np.ndarray): vprint(True, "[EMB] raw shape:", embs.shape, "dtype:", embs.dtype)
    return embs

def ensure_3d(embs: np.ndarray, verbose: bool=False) -> tuple[np.ndarray, list[int]]:
    if embs.ndim == 2:
        embs3, layers = embs[None, ...], [0]
    elif embs.ndim == 3:
        embs3, layers = embs, list(range(embs.shape[0]))
    else:
        raise ValueError(f"Unexpected embedding shape: {embs.shape}")
    if verbose:
        vprint(True, f"[EMB] normalized to 3D: {embs3.shape} (L,N,D)=({embs3.shape[0]},{embs3.shape[1]},{embs3.shape[2]})")
    return embs3, layers

def parse_shrinkage(arg: str | None) -> float | str | None:
    if arg is None: return None
    if isinstance(arg, str):
        s = arg.lower()
        if s in ("none", "null", "false"): return None
        if s == "auto": return "auto"
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"--shrinkage は 'auto' / 'none' / <float> で指定してください: {arg}")
    return arg  # float

def make_lda(solver: str, shrinkage, n_components: int | None, verbose: bool=False) -> LinearDiscriminantAnalysis:
    solver = solver.lower()
    if solver not in ("eigen", "svd", "lsqr"):
        raise ValueError("--solver は eigen / svd / lsqr から選んでください。")
    # 互換性チェック
    if solver == "svd" and shrinkage is not None:
        vprint(True, "[WARN] solver='svd' は shrinkage 非対応のため無視します。")
        shrinkage = None
    lda = LinearDiscriminantAnalysis(
        solver=solver,
        shrinkage=shrinkage,
        n_components=(None if solver=="lsqr" else n_components)
    )
    if verbose:
        vprint(True, "[LDA] init params:", {"solver": solver, "shrinkage": shrinkage, "n_components": n_components})
    return lda

def main():
    p = argparse.ArgumentParser(description="Train and save LDA classifier with optional dimensionality reduction.")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="gpt2",
                   help='gpt2 | fasttext | meta-llama/Meta-Llama-3-8B (aliases: llama3, llama-3-8b, "Llama-3 8B")')
    p.add_argument("--method", default="last_token", choices=["average", "last_token"])
    p.add_argument("--layer", default="all", help='"all" or integer index')
    p.add_argument("--device", default="cuda")
    p.add_argument("--template", default="entity_only")
    p.add_argument("--solver", default="svd", choices=["eigen", "svd", "lsqr"],
                   help="次元削減を保存したい場合は eigenか svd を使用。lsqr は transform 不可。")
    p.add_argument("--shrinkage", default="auto",
                   help="'auto' | 'none' | <float>  (svd では無視されます)")
    p.add_argument("--n_components", type=int, default=None,
                   help="判別成分の数（省略時は自動で <= n_classes-1）")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no_report", action="store_true")
    args = p.parse_args()
    verbose = args.verbose

    # モデル名の正規化
    model_map = {"llama-3-8b": "meta-llama/Meta-Llama-3-8B", "Llama-3 8B": "meta-llama/Meta-Llama-3-8B"}
    model_type = model_map.get(args.model, args.model)

    if verbose:
        vprint(True, "=== Arguments ===")
        vprint(True, json.dumps(vars(args), indent=2, ensure_ascii=False))

    # 1) データ読み込み（is_entity=True のみ）
    entity_names, labels_str, dbg = load_entities_and_labels(Path(args.input), verbose=verbose)

    # 2) テンプレート適用
    texts = apply_template_to_entities(entity_names, args.template, verbose=verbose)

    # 3) 埋め込み
    embedder = build_embedder(model_type=model_type, method=args.method, layer=args.layer, device=args.device, verbose=verbose)
    embs = compute_embeddings(embedder, texts, entity_names, verbose=verbose)
    embs_3d, layer_indices = ensure_3d(embs, verbose=verbose)

    N = embs_3d.shape[1]
    if N != len(labels_str):
        raise RuntimeError(f"Embedding count mismatch: {N} vs labels {len(labels_str)}")

    # 4) ラベル符号化
    le = LabelEncoder()
    y_int = le.fit_transform(labels_str)
    if verbose:
        vprint(True, "[LABEL] class->id mapping:", {cls: int(i) for i, cls in enumerate(le.classes_)})

    # 5) LDA 学習（レイヤー毎）
    shrinkage = parse_shrinkage(args.shrinkage)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    base_stub = f"lda_{model_type.split('/')[-1]}_{args.method}_{args.template.replace('/', '-')}"
    outdir = Path(args.outdir, args.model, f"comp_{args.n_components}")

    for out_idx, l_idx in enumerate(layer_indices):
        X = embs_3d[out_idx].astype(np.float64, copy=False)
        vprint(verbose, f"[TRAIN][layer {l_idx}] X.shape={X.shape}")

        lda = make_lda(args.solver, shrinkage, args.n_components, verbose=verbose)
        lda.fit(X, y_int)

        # transform 可否 & 次元数
        n_comp_eff = None
        try:
            T = lda.transform(X)  # 次元削減（solver='eigen' または 'svd' のみ）
            n_comp_eff = int(T.shape[1])
            vprint(verbose, f"[DIMRED][layer {l_idx}] transform(X) -> {T.shape}")
            if hasattr(lda, "explained_variance_ratio_"):
                evr = getattr(lda, "explained_variance_ratio_")
                vprint(verbose, f"[DIMRED][layer {l_idx}] explained_variance_ratio_ (len={len(evr)}): {np.round(evr, 4)}")
        except Exception as e:
            vprint(True, f"[DIMRED][layer {l_idx}] transform not available ({args.solver}): {e}")

        if not args.no_report:
            acc = lda.score(X, y_int)
            vprint(True, f"[RESULT][layer {l_idx}] Train accuracy: {acc:.4f}")
            try:
                y_pred = lda.predict(X)
                vprint(True, "[REPORT]\n" + classification_report(y_int, y_pred, target_names=le.classes_))
                vprint(True, "[CONFUSION]\n" + np.array2string(confusion_matrix(y_int, y_pred)))
            except Exception as e:
                vprint(True, f"[WARN] report failed: {e}")

        meta = {
            "timestamp_utc": ts,
            "input_jsonl": str(args.input),
            "n_samples": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "classes": le.classes_.tolist(),
            "class_counts": dbg.get("class_counts", {}),
            "embedding": {
                "model_type": model_type,
                "method": args.method,
                "layer_requested": args.layer,
                "layer_index_trained": int(l_idx),
                "device": args.device,
            },
            "template": args.template,
            "sklearn_version": sklearn.__version__,
            "lda_params": {
                "solver": args.solver,
                "shrinkage": None if shrinkage is None else ("auto" if shrinkage == "auto" else float(shrinkage)),
                "n_components_arg": args.n_components,
            },
            "n_components_effective": n_comp_eff,  # transform できた場合の実効次元
        }

        # 保存
        outdir.mkdir(parents=True, exist_ok=True)
        model_path = outdir / f"{base_stub}_layer{l_idx}.joblib"
        meta_path  = outdir / f"{base_stub}_layer{l_idx}.meta.json"
        joblib.dump({"lda": lda, "label_encoder": le, "meta": meta}, model_path)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        vprint(True, "[SAVED]", model_path)
        vprint(True, "[SAVED]", meta_path)

if __name__ == "__main__":
    main()
