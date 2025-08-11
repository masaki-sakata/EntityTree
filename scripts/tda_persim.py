# tda_min_persim.py
import numpy as np, pandas as pd
from pathlib import Path
from ripser import ripser
from persim import plot_diagrams, bottleneck, wasserstein
import matplotlib.pyplot as plt

from embeddings import EmbeddingConfig, EmbeddingModel
import template

def _cosine_dm(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = X @ X.T
    D = 1.0 - np.clip(S, -1.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D

def _euclid_dm(X):
    XX = np.sum(X*X, axis=1, keepdims=True)
    D2 = XX + XX.T - 2*(X@X.T)
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2); np.fill_diagonal(D, 0.0); return D

def _auto_thresh(D, q=0.9):
    tri = D[np.triu_indices_from(D, 1)]
    return float(np.quantile(tri, q)) if tri.size else 1.0

def get_entity_names(jsonl_path):
    df = pd.read_json(jsonl_path, lines=True)
    return sorted(df[df["is_entity"]==True]["wiki_title"].tolist())

def get_embeddings(entity_names, model="gpt2", method="last_token", layer=0, device="cuda",
                   template_name="entity_only", random_dim=768, random_std=1.0, random_seed=42):
    t = template.get_template(template_name)
    texts = [template.apply_template(t, n) for n in entity_names]
    cfg = EmbeddingConfig(model_type=model, method=method, layer=layer, device=device,
                          random_dim=random_dim, random_std=random_std, random_seed=random_seed)
    embs = EmbeddingModel(cfg).encode(texts, entity_names)
    if hasattr(embs, "detach"): embs = embs.detach().cpu().numpy()
    if embs.ndim == 3: embs = embs[0]   # (1,N,D) → (N,D)
    return embs  # (N,D)

def persistent_diagrams_from_embeddings(X, metric="cosine", maxdim=1, thresh="auto"):
    D = _cosine_dm(X) if metric=="cosine" else _euclid_dm(X)
    if thresh=="auto": thresh = _auto_thresh(D, 0.9)
    res = ripser(D, maxdim=maxdim, thresh=thresh, distance_matrix=True)
    dgms = {k: res["dgms"][k] for k in range(min(len(res["dgms"]), maxdim+1))}
    return dgms, float(thresh)

def save_pd_png(dgms, out_png, title="PD"):
    plt.figure(figsize=(5.5,4.5), dpi=150)
    plot_diagrams([dgms.get(k, np.zeros((0,2))) for k in sorted(dgms)], show=False, legend=True)
    plt.title(title); plt.tight_layout(); plt.savefig(out_png); plt.close()

def summarize_pd(dgms):
    def summary(D):
        if D.size==0: return dict(count=0, total=0.0, mean=0.0, entropy=0.0)
        life = np.maximum(D[:,1]-D[:,0],0.0)
        total = float(life.sum()); mean = float(life.mean())
        if total>0: p=life/total; ent=float(-(p*np.log(p+1e-12)).sum())
        else: ent=0.0
        return dict(count=int(life.size), total=total, mean=mean, entropy=ent)
    out={}
    for k in sorted(dgms):
        s=summary(dgms[k])
        for key,val in s.items(): out[f"H{k}_{key}"]=val
    return out


if __name__=="__main__":
    import argparse, json
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--method", default="last_token")
    ap.add_argument("--layer", default="0")
    ap.add_argument("--metric", default="cosine", choices=["cosine","euclidean"])
    ap.add_argument("--maxdim", type=int, default=1)
    ap.add_argument("--thresh", default="auto")
    args=ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ents = get_entity_names(args.input)
    layer = 0 if args.layer!="all" else 0  # 最小版は単層のみ
    X = get_embeddings(ents, model=args.model, method=args.method, layer=layer)

    dgms, thr = persistent_diagrams_from_embeddings(X, metric=args.metric,
                                                    maxdim=args.maxdim, thresh=args.thresh)
    save_pd_png(dgms, out/"pd.png", f"{args.model} L{layer} (N={len(ents)})")
    summ = summarize_pd(dgms); summ.update(dict(thresh=thr, n=len(ents)))
    (out/"tda_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("[OK] PD/summary saved:", out)
