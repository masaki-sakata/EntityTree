#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tda_persim.py
- Gold の理想 0次PD
- 各(モデル, 層)の 0次PD
- Gold との Bottleneck/Wasserstein/Betti0-L2
- 0次バーコード & Betti-0 曲線（PNG+CSV）を gold / 各層で保存
- 層間・モデル間の PD0 距離行列
- 長寿特徴の MST 解釈（上位kエッジを切ったクラスタの職業混合など）

依存:
  pip install ripser persim matplotlib numpy pandas networkx
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from ripser import ripser
from persim import plot_diagrams, bottleneck, wasserstein

# 既存リポジトリの関数たちを再利用
from embeddings import EmbeddingConfig, EmbeddingModel
import template
from eval_tree import build_gold_tree, reconstruct_entity_tree_adjacency

# -----------------------------
# Gold: ウルトラメトリック生成
# -----------------------------
def _build_parent_map(adj: Dict[int, List[int]]) -> Dict[int, Optional[int]]:
    parent = {}
    for p, cs in adj.items():
        parent.setdefault(p, None)
        for c in cs:
            parent[c] = p
    return parent

def _find_roots(adj: Dict[int, List[int]], n_leaves: int) -> List[int]:
    parents = set(adj.keys())
    children = {c for cs in adj.values() for c in cs}
    roots = list(parents - children)
    for i in range(n_leaves):
        if (i not in parents) and (i not in children):
            roots.append(i)
    return roots

def _compute_depth_map(adj: Dict[int, List[int]], n_leaves: int):
    from collections import deque
    parent = _build_parent_map(adj)
    roots = _find_roots(adj, n_leaves)
    depth = {}
    q = deque((r,0) for r in roots)
    while q:
        u,d = q.popleft()
        if u in depth: continue
        depth[u] = d
        for v in adj.get(u, []):
            q.append((v, d+1))
    all_nodes = set(parent.keys()) | {c for cs in adj.values() for c in cs} | set(range(n_leaves))
    for u in all_nodes:
        depth.setdefault(u, 0)
        parent.setdefault(u, None)
    return depth, parent, roots

def _lca(u, v, parent, depth):
    uu, vv = u, v
    du, dv = depth.get(uu,0), depth.get(vv,0)
    while du > dv and uu is not None:
        uu = parent.get(uu, None); du -= 1
    while dv > du and vv is not None:
        vv = parent.get(vv, None); dv -= 1
    while uu != vv:
        uu = parent.get(uu, None) if uu is not None else None
        vv = parent.get(vv, None) if vv is not None else None
        if uu is None and vv is None:
            return None
    return uu

def gold_ultrametric_from_jsonl(jsonl_path: Union[str, Path]) -> Tuple[np.ndarray, List[str], Dict[str,str]]:
    df = pd.read_json(jsonl_path, lines=True)
    _, entity_names, prof_map = build_gold_tree(df)
    gold_adj, _ = reconstruct_entity_tree_adjacency(df, entity_names)
    n = len(entity_names)
    depth, parent, _ = _compute_depth_map(gold_adj, n)
    max_depth = max(depth.values()) if depth else 1

    DM = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            anc = _lca(i, j, parent, depth)
            h = (max_depth - depth.get(anc, 0)) if anc is not None else (max_depth + 1.0)
            DM[i,j] = DM[j,i] = float(h)
    tri = DM[np.triu_indices_from(DM, 1)]
    mn, mx = (float(tri.min()) if tri.size else 0.0), (float(tri.max()) if tri.size else 1.0)
    if mx > mn:
        DM = (DM - mn) / (mx - mn)
    np.fill_diagonal(DM, 0.0)
    return DM, entity_names, prof_map

# -----------------------------
# 埋め込み & 距離
# -----------------------------
def texts_from_entities(ents: List[str], template_name="entity_only")->List[str]:
    t = template.get_template(template_name)
    return [template.apply_template(t, n) for n in ents]

def get_embeddings(entity_names: List[str], model: str, method: str, layer: Union[int,str],
                   device: str, template_name: str, verbose=False,
                   random_dim=768, random_std=1.0, random_seed=42) -> Union[np.ndarray, np.ndarray]:
    texts = texts_from_entities(entity_names, template_name)
    cfg = EmbeddingConfig(model_type=model, method=method, layer=0 if model=="random_emb" else layer,
                          device=device, verbose=verbose,
                          random_dim=random_dim, random_std=random_std, random_seed=random_seed)
    embs = EmbeddingModel(cfg).encode(texts, entity_names)
    if hasattr(embs, "detach"): embs = embs.detach().cpu().numpy()
    return embs

def cosine_DM(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True)+1e-12)
    S = Xn @ Xn.T
    D = 1.0 - np.clip(S, -1.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D

def euclid_DM(X: np.ndarray) -> np.ndarray:
    XX = (X*X).sum(1, keepdims=True)
    D2 = XX + XX.T - 2*(X@X.T)
    D2 = np.maximum(D2, 0)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D

def auto_thresh(D: np.ndarray, q=0.9) -> float:
    tri = D[np.triu_indices_from(D, 1)]
    return float(np.quantile(tri, q)) if tri.size else 1.0

def pd0_from_DM(D: np.ndarray, thresh="auto"):
    thr = auto_thresh(D) if thresh=="auto" else float(thresh)
    res = ripser(D, maxdim=0, thresh=thr, distance_matrix=True)
    dgm0 = res["dgms"][0] if len(res["dgms"])>0 else np.zeros((0,2))
    return {0: dgm0}, float(thr)

# -----------------------------
# 可視化 & 保存（PD点図 / バーコード / Betti0）
# -----------------------------
def save_pd_png(dgms: Dict[int,np.ndarray], path: Path, title="PD"):
    path.parent.mkdir(parents=True, exist_ok=True) 
    plt.figure(figsize=(5.5,4.5), dpi=150)
    plot_diagrams([dgms.get(k, np.zeros((0,2))) for k in sorted(dgms)], show=False, legend=True)
    plt.title(title); plt.tight_layout(); plt.savefig(path); plt.close()

def save_barcode0_png(dgm0: np.ndarray, path: Path, title="Barcode H0", r_max: Optional[float]=None):
    """
    H0 のバーコードをプロット。death=inf は描画しない（注記のみ）。
    xlim の右端は有限 death の最大値を使い、無ければ安全な既定値(=1.0)。
    """
    path.parent.mkdir(parents=True, exist_ok=True) 

    plt.figure(figsize=(7, max(2.8, 0.12*max(6, dgm0.shape[0]))), dpi=140)

    # 既定の上限
    SAFE_DEFAULT = 1.0

    # 入力が空なら素直に既定範囲
    if dgm0.size == 0:
        r = SAFE_DEFAULT if (r_max is None or not np.isfinite(r_max) or r_max <= 0) else float(r_max)
        plt.xlim(0, max(r * 1.02, 1e-6))
        plt.ylim(-1, 5)
        plt.xlabel("radius r"); plt.ylabel("H0 intervals (sorted)")
        plt.title(title); plt.tight_layout(); plt.savefig(path); plt.close()
        return

    D = dgm0.copy()
    finite = np.isfinite(D[:, 1])
    Df = D[finite]

    # r_max が未指定 or 非有限なら、有限 death の最大を採用。なければ既定値。
    if r_max is None or not np.isfinite(r_max) or r_max <= 0:
        if Df.size:
            r_max = float(np.max(Df[:, 1]))
        else:
            r_max = SAFE_DEFAULT

    # 念のため最終ガード
    if not np.isfinite(r_max) or r_max <= 0:
        r_max = SAFE_DEFAULT

    # バーコード描画（finite only）
    y = 0
    for b, d in Df:
        plt.hlines(y=y, xmin=float(b), xmax=min(float(d), r_max), linewidth=2)
        y += 1

    # ∞ バーの注記
    inf_count = int((~finite).sum())
    if inf_count > 0:
        plt.text(0.02, y + 0.5, f"{inf_count} bars with death=inf (not drawn)", fontsize=9)

    plt.ylim(-1, max(5, y) + 1)
    plt.xlim(0, max(r_max * 1.02, 1e-6))
    plt.xlabel("radius r"); plt.ylabel("H0 intervals (sorted)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()


def betti0_curve(dgm0: np.ndarray, r_min: float, r_max: float, num=400) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.linspace(r_min, r_max, num=num)
    if dgm0.size == 0:
        return rs, np.ones_like(rs, dtype=int)
    b = dgm0[:,0][:,None]
    d = dgm0[:,1][:,None]
    mask = (b <= rs[None,:]) & (rs[None,:] < d)
    beta = mask.sum(axis=0)
    return rs, beta

def save_line(xs, ys, path: Path, xlabel, ylabel, title):
    plt.figure(figsize=(6,4), dpi=150)
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

def write_pd0_csv(dgm0: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True) 
    # birth, death（inf は文字列 'inf'）
    if dgm0.size:
        arr = []
        for b,d in dgm0:
            arr.append((float(b), ("inf" if not np.isfinite(d) else float(d))))
        pd.DataFrame(arr, columns=["birth","death"]).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=["birth","death"]).to_csv(path, index=False)

def write_betti_csv(rs: np.ndarray, beta: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True) 
    pd.DataFrame({"r": rs, "betti0": beta}).to_csv(path, index=False)

# -----------------------------
# MST 解釈（長寿特徴）
# -----------------------------
def mst_top_edges(D: np.ndarray, k: int=3) -> List[Tuple[int,int,float]]:
    G = nx.Graph(); n = D.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            w = float(D[i,j])
            if np.isfinite(w): G.add_edge(i, j, weight=w)
    T = nx.minimum_spanning_tree(G, algorithm="kruskal")
    edges = [(u,v,T[u][v]["weight"]) for u,v in T.edges()]
    return sorted(edges, key=lambda x: x[2], reverse=True)[:k]

def partition_by_cut(D: np.ndarray, cut_edge: Tuple[int,int]) -> List[List[int]]:
    G = nx.Graph(); n = D.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=float(D[i,j]))
    T = nx.minimum_spanning_tree(G, algorithm="kruskal")
    u,v = cut_edge
    if T.has_edge(u,v): T.remove_edge(u,v)
    comps = list(nx.connected_components(T))
    return [sorted(list(c)) for c in comps]

def describe_partition(clusters: List[List[int]], entity_names: List[str], profession_map: Dict[str,str]) -> List[Dict]:
    out=[]
    for cid, idxs in enumerate(clusters):
        names=[entity_names[i] for i in idxs]
        profs=[profession_map.get(n, "Person") for n in names]
        vals=pd.Series(profs).value_counts().to_dict()
        maj = max(vals.items(), key=lambda x:x[1])
        out.append(dict(cluster_id=cid, size=len(idxs), majority_label=maj[0], majority_count=int(maj[1]),
                        label_hist=vals, members=names))
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Gold vs PD(0D) + Barcodes + Betti0 curves")
    ap.add_argument("--input", required=True, help="Gold JSONL")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--models", default="gpt2", help="カンマ区切り: gpt2,fasttext,meta-llama/Meta-Llama-3-8B,random_emb")
    ap.add_argument("--method", default="last_token", choices=["average","last_token"])
    ap.add_argument("--layer", default="all", help="整数 or 'all'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--template", default="entity_only")
    ap.add_argument("--metric", default="cosine", choices=["cosine","euclidean"])
    ap.add_argument("--thresh", default="auto")
    ap.add_argument("--topk_mst", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--random_dim", type=int, default=768)
    ap.add_argument("--random_std", type=float, default=1.0)
    ap.add_argument("--random_seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # --- Gold ---
    DM_gold, entity_names, prof_map = gold_ultrametric_from_jsonl(args.input)
    dgm_gold, thr_gold = pd0_from_DM(DM_gold, thresh="auto")

    if dgm_gold[0].size:
        finite_gold = dgm_gold[0][:, 1][np.isfinite(dgm_gold[0][:, 1])]
        if finite_gold.size:
            rmax_gold = float(np.max(finite_gold))
        else:
            rmax_gold = float(thr_gold) if np.isfinite(thr_gold) else 1.0
    else:
        rmax_gold = 1.0

    save_pd_png(dgm_gold, out/"gold_pd0.png", "Gold (ideal) PD0")
    write_pd0_csv(dgm_gold[0], out/"gold_pd0.csv")

    # Gold バーコード & Betti0（r 範囲は後で全体最大に合わせて再出力も可）
    rmax_gold = float(np.nanmax(dgm_gold[0][:,1])) if dgm_gold[0].size else 1.0
    save_barcode0_png(dgm_gold[0], out/"gold_barcode0.png", "Gold Barcode H0", r_max=rmax_gold)
    rs_g, beta_g = betti0_curve(dgm_gold[0], 0.0, max(1.0, thr_gold, rmax_gold), num=400)
    write_betti_csv(rs_g, beta_g, out/"gold_betti0.csv")

    # --- 各(モデル,層) ---
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    per_layer_rows = []
    all_pd0 = {}   # key=(model,layer_idx) -> np.ndarray (PD0)
    all_DM  = {}
    max_death = rmax_gold

    for model in model_list:
        embs = get_embeddings(
            entity_names, model, args.method, args.layer, args.device,
            args.template, args.verbose, args.random_dim, args.random_std, args.random_seed
        )

        # ===== Guard: layer='all' でも 2D 埋め込み（fasttext / random_emb 等）を扱えるように =====
        if isinstance(args.layer, str) and args.layer.lower() == "all":
            # 期待形状は (L, N, D)。2D の場合は (1, N, D) に持ち上げる
            if embs.ndim == 2:
                embs = embs[None, ...]  # (N, D) → (1, N, D)
            elif embs.ndim != 3:
                raise ValueError(f"Unexpected embeddings shape for layer='all': {getattr(embs, 'shape', None)}")

            L, N, D = embs.shape
            for li in range(L):
                X = embs[li]  # (N, D)

                DM = cosine_DM(X) if args.metric == "cosine" else euclid_DM(X)
                dgm, thr = pd0_from_DM(DM, thresh=args.thresh)
                d0 = dgm.get(0, np.zeros((0, 2)))
                all_pd0[(model, li)] = d0
                all_DM[(model, li)] = DM

                if d0.size:
                    finite = d0[:, 1][np.isfinite(d0[:, 1])]
                    if finite.size:
                        max_death = max(max_death, float(np.max(finite)))

                # Gold との距離
                bn = bottleneck(d0, dgm_gold[0])
                w1 = w1 = wasserstein(d0, dgm_gold[0])  

                # Betti-0 L2（Gold の r 範囲に合わせる）
                rs_l, beta_l = betti0_curve(d0, rs_g.min(), rs_g.max(), num=len(rs_g))
                beta_l2 = float(np.sqrt(np.mean((beta_l - beta_g) ** 2)))

                per_layer_rows.append(dict(
                    model=model, layer=li, n_entities=N, metric=args.metric,
                    thresh_layer=thr, bottleneck=float(bn),
                    wasserstein1=float(w1), betti0_L2=float(beta_l2)
                ))

        else:
            # ===== 単層モード（layer が整数）=====
            if embs.ndim == 3:
                # まれに (1, N, D) が返る実装に備える
                embs = embs[0]
            elif embs.ndim != 2:
                raise ValueError(f"Unexpected embeddings shape for single layer: {getattr(embs, 'shape', None)}")

            X = embs  # (N, D)
            N = X.shape[0]

            DM = cosine_DM(X) if args.metric == "cosine" else euclid_DM(X)
            dgm, thr = pd0_from_DM(DM, thresh=args.thresh)
            d0 = dgm.get(0, np.zeros((0, 2)))

            layer_idx = int(args.layer)
            all_pd0[(model, layer_idx)] = d0
            all_DM[(model, layer_idx)] = DM

            if d0.size:
                finite = d0[:, 1][np.isfinite(d0[:, 1])]
                if finite.size:
                    max_death = max(max_death, float(np.max(finite)))

            bn = bottleneck(d0, dgm_gold[0])
            w1 = w1 = wasserstein(d0, dgm_gold[0])  

            rs_l, beta_l = betti0_curve(d0, rs_g.min(), rs_g.max(), num=len(rs_g))
            beta_l2 = float(np.sqrt(np.mean((beta_l - beta_g) ** 2)))

            per_layer_rows.append(dict(
                model=model, layer=layer_idx, n_entities=N, metric=args.metric,
                thresh_layer=thr, bottleneck=float(bn),
                wasserstein1=float(w1), betti0_L2=float(beta_l2)
            ))

    # --- 保存: gold 一致度テーブル ---
    df = pd.DataFrame(per_layer_rows)
    if not df.empty:
        df.sort_values(["model","layer"]).to_csv(out/"gold_consistency_per_layer.csv", index=False)

    # --- 共通 r 範囲で、バーコード & Betti0 を全保存 ---
    r_max = max(1.0, max_death)
    for (model, layer) in sorted(all_pd0.keys(), key=lambda x:(x[0], x[1])):
        d0 = all_pd0[(model, layer)]
        # PD 点図
        save_pd_png({0:d0}, out/f"{model}_L{layer}_pd0.png", f"{model} L{layer} – PD0")
        write_pd0_csv(d0, out/f"{model}_L{layer}_pd0.csv")
        # バーコード
        save_barcode0_png(d0, out/f"{model}_L{layer}_barcode0.png",
                          f"{model} L{layer} – Barcode H0", r_max=r_max)
        # Betti0（共通 r）
        rs, beta = betti0_curve(d0, 0.0, r_max, num=400)
        write_betti_csv(rs, beta, out/f"{model}_L{layer}_betti0.csv")
        # ついでに Gold との重ね描き
        plt.figure(figsize=(6,4), dpi=150)
        plt.plot(*betti0_curve(dgm_gold[0], 0.0, r_max, num=400), label="Gold β0")
        plt.plot(rs, beta, label=f"{model} L{layer} β0")
        plt.xlabel("radius r"); plt.ylabel("Betti0"); plt.legend(); plt.title(f"Betti0: Gold vs {model} L{layer}")
        plt.tight_layout(); plt.savefig(out/f"{model}_L{layer}_betti0_vs_gold.png"); plt.close()

    # --- モデル・層間 PD0 距離行列 ---
    keys = sorted(all_pd0.keys(), key=lambda x:(x[0], x[1]))
    if len(keys) >= 2:
        nK = len(keys)
        M_bn = np.zeros((nK, nK)); M_w1 = np.zeros_like(M_bn)
        for i in range(nK):
            for j in range(i, nK):
                d_bn = bottleneck(all_pd0[keys[i]], all_pd0[keys[j]])
                d_w1 = wasserstein(all_pd0[keys[i]], all_pd0[keys[j]], matching=False, order=1, internal_p=2.0)
                M_bn[i,j]=M_bn[j,i]=float(d_bn); M_w1[i,j]=M_w1[j,i]=float(d_w1)
        lab = [f"{m}-L{l}" for (m,l) in keys]
        pd.DataFrame(M_bn, index=lab, columns=lab).to_csv(out/"pd0_bottleneck_matrix.csv")
        pd.DataFrame(M_w1, index=lab, columns=lab).to_csv(out/"pd0_wasserstein1_matrix.csv")

    # --- MST による長寿特徴の解釈（ベスト層で） ---
    if not df.empty:
        best = df.sort_values("bottleneck", ascending=True).iloc[0]
        mkey = (best["model"], int(best["layer"]))
        DM_best = all_DM[mkey]
        top_edges = mst_top_edges(DM_best, k=int(args.topk_mst))
        report=[]
        for (u,v,w) in top_edges:
            parts = partition_by_cut(DM_best, (u,v))
            desc = describe_partition(parts, entity_names, prof_map)
            report.append(dict(edge=(int(u),int(v)), weight=float(w), partition=desc))
        (out/"long_lived_features.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
