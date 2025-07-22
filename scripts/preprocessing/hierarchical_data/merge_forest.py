#!/usr/bin/env python3
"""
merge_forest.py

・複数の JSONL で表現されたツリーを QID ベースで完全マージ
・連結成分ごとに新しい tree_id を振り直し
・NaN / None の値では既存属性を上書きしない（pop 欄が NaN になる問題を回避）
・tree_id 昇順で JSONL 出力

USAGE:
    poetry run python3 merge_forest.py --input /work03/masaki/data/taxonomy/taxonomy_from_popQA.jsonl --output /work03/masaki/data/taxonomy/taxonomy_from_popQA_merged.jsonl

"""

import argparse
import json
import math
from collections import defaultdict
from itertools import count
from pathlib import Path
import sys

import pandas as pd

###############################################################################
# 1. argparse ── コマンドライン引数
###############################################################################
parser = argparse.ArgumentParser(description="Merge multiple JSONL trees into one forest")
parser.add_argument("-i", "--input",  required=True, help="Input JSONL file")
parser.add_argument("-o", "--output", default="-",
                    help="Output JSONL file (default: stdout)")
args = parser.parse_args()

in_path  = Path(args.input)
out_fh   = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")

###############################################################################
# 2. JSONL 読み込み & ノード / エッジ統合
###############################################################################
df = pd.read_json(in_path, lines=True)

nodes_by_qid: dict[str, dict] = {}        # QID → 属性 dict
out_edges: defaultdict[str, set] = defaultdict(set)  # QID → {(property, target_qid)}
in_edges:  defaultdict[str, set] = defaultdict(set)  # QID → {source_qid}

def is_valid(v):
    """NaN / None を False とする判定"""
    return not (v is None or (isinstance(v, float) and math.isnan(v)))

for _, row in df.iterrows():
    node = row.to_dict()
    qid  = node["qid"]

    # ---------- 属性マージ（NaN/None では上書きしない） ----------
    base = nodes_by_qid.setdefault(qid, {})
    for k, v in node.items():
        if k == "edges":
            continue
        if is_valid(v):
            base[k] = v

    # ---------- エッジ集合を統合 ----------
    for e in node.get("edges", []):
        prop, tgt = e["property"], e["target_qid"]
        out_edges[qid].add((prop, tgt))
        in_edges[tgt].add(qid)

###############################################################################
# 3. Union-Find で連結成分 → tree_id 割り当て
###############################################################################
parent: dict[str, str] = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for src, es in out_edges.items():
    for _, tgt in es:
        if tgt in nodes_by_qid:          # 未知ノードは無視
            union(src, tgt)

tid_counter = count(1)
root2tid: dict[str, int] = {}
def tree_id(qid):
    r = find(qid)
    return root2tid.setdefault(r, next(tid_counter))

###############################################################################
# 4. ノード再構築
###############################################################################
def build_node(qid):
    node = nodes_by_qid[qid].copy()
    node["tree_id"] = tree_id(qid)

    # エッジをラベル付きで整形
    edges = [
        {"property": p,
         "target_qid": t,
         "target_label": nodes_by_qid.get(t, {}).get("wiki_title", "")}
        for p, t in sorted(out_edges[qid])
    ]
    node.update({
        "edges": edges,
        "source_props": sorted({p for p, _ in out_edges[qid]}),
        "num_edges": len(edges),
    })
    return node

result_nodes = [build_node(q) for q in nodes_by_qid]
result_nodes.sort(key=lambda n: n["tree_id"])   # tree_id で昇順ソート

###############################################################################
# 5. JSONL 書き出し
###############################################################################
for n in result_nodes:
    print(json.dumps(n, ensure_ascii=False), file=out_fh)

if out_fh is not sys.stdout:
    out_fh.close()
