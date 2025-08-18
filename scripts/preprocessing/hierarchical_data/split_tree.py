#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
木構造のJSONLを、人数（is_entity=true）ベースの比率で
学習用／評価用に分割するスクリプト。

- 比率指定: 8:2, 9:1, 0.8, 80%, 4/1 など（学習側の比率）
- 人数ベースで分割（is_entity=true の件数に対して）
- 既定はカテゴリ(P106)ごとの層化分割（ON）
- 非エンティティノードは両方に含め、カテゴリ側edgesは分割に合わせて人物のみを残す
- 乱数シード指定可

uv run python3 split_tree.py --input /home/masaki/hierarchical-repr/EntityTree/input/tree_yago_300people.jsonl --ratio 8:2 \
  --train-out /home/masaki/hierarchical-repr/EntityTree/input/300people/tr80_te20/train.jsonl --eval-out /home/masaki/hierarchical-repr/EntityTree/input/300people/tr80_te20/test.jsonl --seed 42


"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, DefaultDict
from collections import defaultdict

def parse_ratio(text: str) -> float:
    """
    比率文字列を学習側の小数 (0<r<1) に変換。
    許容: '8:2', '4/1', '0.8', '80%', '0.80'
    """
    s = text.strip()
    if ":" in s or "/" in s:
        delim = ":" if ":" in s else "/"
        a, b = s.split(delim, 1)
        a = float(a.strip())
        b = float(b.strip())
        if a < 0 or b < 0 or (a + b) == 0:
            raise ValueError(f"Invalid ratio: {text}")
        r = a / (a + b)
    else:
        s2 = s.replace("%", "")
        v = float(s2)
        # 80 or 80.0 → 0.8, 0.8 → 0.8
        r = v / 100.0 if "%" in s or v > 1.0 else v
    if not (0.0 < r < 1.0):
        raise ValueError(f"Ratio must be between 0 and 1 (exclusive): {r}")
    return r

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def pick_primary_category_qid(person_node: Dict[str, Any]) -> str:
    """
    人物ノードの主カテゴリ（P106）の target_qid を返す。
    複数あれば最初のもの。なければ 'UNKNOWN'。
    """
    edges = person_node.get("edges", []) or []
    p106 = [e for e in edges if e.get("property") == "P106"]
    if not p106:
        return "UNKNOWN"
    # “最初のもの”に安定性を持たせたいので、target_qid があればそれで昇順ソートして先頭
    # （入力順に依存したくない場合）
    def keyer(e):
        q = e.get("target_qid")
        return "" if q is None else q
    p106_sorted = sorted(p106, key=keyer)
    return p106_sorted[0].get("target_qid", "UNKNOWN") or "UNKNOWN"

def compute_group_counts(
    group_sizes: Dict[str, int],
    train_ratio: float,
    total_people: int
) -> Dict[str, int]:
    """
    各グループの学習数を決める。
    - まず floor(n_i * r) をとり、全体の target = round(N * r) に合わせて
      端数の大きいグループから +1 して調整。
    """
    desired_total = int(round(total_people * train_ratio))
    # 1人以上いる場合は両セットが空にならないよう矯正（任意）
    if total_people >= 2:
        desired_total = max(1, min(desired_total, total_people - 1))

    base = {}
    fracs: List[Tuple[str, float]] = []
    s = 0
    for g, n in group_sizes.items():
        w = n * train_ratio
        b = math.floor(w)
        base[g] = b
        s += b
        fracs.append((g, w - b))

    need = desired_total - s
    # 端数が大きい順に +1
    fracs.sort(key=lambda x: x[1], reverse=True)
    for i in range(max(0, need)):
        g = fracs[i % len(fracs)][0]
        base[g] += 1

    return base

def stratified_split(
    persons: List[Dict[str, Any]],
    train_ratio: float,
    seed: int
) -> Tuple[Set[str], Set[str]]:
    """
    主カテゴリ(P106)ごとに層化して分割。戻り値は (train_qids, eval_qids)。
    """
    # グルーピング
    groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in persons:
        g = pick_primary_category_qid(p)
        groups[g].append(p)

    # 各グループの学習数を算出
    group_sizes = {g: len(lst) for g, lst in groups.items()}
    total = sum(group_sizes.values())
    train_counts = compute_group_counts(group_sizes, train_ratio, total)

    rnd = random.Random(seed)
    train_qids: Set[str] = set()
    eval_qids: Set[str] = set()

    for g, lst in groups.items():
        # 安定性のために qid で整列後、乱択
        lst_sorted = sorted(lst, key=lambda x: x.get("qid", ""))
        rnd.shuffle(lst_sorted)
        k = train_counts.get(g, 0)
        train_nodes = lst_sorted[:k]
        eval_nodes = lst_sorted[k:]
        train_qids.update([n.get("qid") for n in train_nodes if n.get("qid")])
        eval_qids.update([n.get("qid") for n in eval_nodes if n.get("qid")])

    return train_qids, eval_qids

def non_stratified_split(
    persons: List[Dict[str, Any]],
    train_ratio: float,
    seed: int
) -> Tuple[Set[str], Set[str]]:
    """
    層化しない単純ランダム分割。戻り値は (train_qids, eval_qids)。
    """
    total = len(persons)
    desired_total = int(round(total * train_ratio))
    if total >= 2:
        desired_total = max(1, min(desired_total, total - 1))
    rnd = random.Random(seed)
    lst_sorted = sorted(persons, key=lambda x: x.get("qid", ""))
    rnd.shuffle(lst_sorted)
    train_nodes = lst_sorted[:desired_total]
    eval_nodes = lst_sorted[desired_total:]
    return (
        set([n.get("qid") for n in train_nodes if n.get("qid")]),
        set([n.get("qid") for n in eval_nodes if n.get("qid")]),
    )

def filter_category_edges_for_split(
    node: Dict[str, Any],
    person_qids_in_split: Set[str],
    qid_is_entity: Dict[str, bool],
) -> Dict[str, Any]:
    """
    非エンティティ（カテゴリ）ノードの edges から、
    分割に含まれない人物（is_entity=true かつ not in person_qids_in_split）を除く。
    それ以外（カテゴリ→カテゴリ等）は保持。num_edges を更新。
    """
    edges = node.get("edges", []) or []
    new_edges = []
    for e in edges:
        tgt = e.get("target_qid")
        if tgt is None:
            new_edges.append(e)
            continue
        if qid_is_entity.get(tgt, False):
            # 人物
            if tgt in person_qids_in_split:
                new_edges.append(e)
        else:
            # カテゴリ等
            new_edges.append(e)

    out = dict(node)
    out["edges"] = new_edges
    out["num_edges"] = len(new_edges)
    return out

def build_outputs(
    all_nodes: List[Dict[str, Any]],
    train_qids: Set[str],
    eval_qids: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    出力JSONL（学習／評価）の行リストを構築。
    - 非エンティティは両方に含め、edgesを分割ごとにフィルタ
    - 人物は該当分割にのみ含める（内容はそのまま）
    """
    # qid→node / qid→is_entity
    qid_to_node: Dict[str, Dict[str, Any]] = {}
    qid_is_entity: Dict[str, bool] = {}
    for n in all_nodes:
        q = n.get("qid")
        if q:
            qid_to_node[q] = n
            qid_is_entity[q] = bool(n.get("is_entity", False))

    train_lines: List[Dict[str, Any]] = []
    eval_lines: List[Dict[str, Any]] = []

    # 1) 非エンティティ（カテゴリ等）は両方に入れる（edgesを各分割でフィルタ）
    for n in all_nodes:
        if not n.get("is_entity", False):
            train_lines.append(
                filter_category_edges_for_split(n, train_qids, qid_is_entity)
            )
            eval_lines.append(
                filter_category_edges_for_split(n, eval_qids, qid_is_entity)
            )

    # 2) 人物は各分割にのみ入れる（内容は原文のまま）
    for q in sorted(train_qids):
        node = qid_to_node.get(q)
        if node:
            train_lines.append(node)
    for q in sorted(eval_qids):
        node = qid_to_node.get(q)
        if node:
            eval_lines.append(node)

    return train_lines, eval_lines

def main():
    ap = argparse.ArgumentParser(description="人数ベースの比率でJSONL木データを学習／評価に分割")
    ap.add_argument("--input", type=str, help="入力JSONLのパス")
    ap.add_argument("--ratio", type=str, required=True,
                    help="学習側の比率。例: '8:2', '9:1', '0.8', '80%', '4/1'")
    ap.add_argument("--train-out", type=str, default=None, help="学習用JSONLの出力パス")
    ap.add_argument("--eval-out", type=str, default=None, help="評価用JSONLの出力パス")
    ap.add_argument("--seed", type=int, default=42, help="乱数シード（既定: 42）")
    ap.add_argument("--no-stratify", action="store_true",
                    help="層化分割を無効化（カテゴリ無視の単純ランダム）")
    args = ap.parse_args()

    inp = Path(args.input)
    if args.train_out is None:
        args.train_out = str(inp.with_suffix("")) + ".train.jsonl"
    if args.eval_out is None:
        args.eval_out = str(inp.with_suffix("")) + ".eval.jsonl"

    train_ratio = parse_ratio(args.ratio)
    nodes = load_jsonl(inp)

    # 人物ノードの抽出
    persons = [n for n in nodes if n.get("is_entity", False)]
    if len(persons) == 0:
        raise RuntimeError("人物（is_entity=true）が見つかりません。")

    # 分割（層化 or 非層化）
    if args.no_stratify:
        train_qids, eval_qids = non_stratified_split(persons, train_ratio, args.seed)
    else:
        train_qids, eval_qids = stratified_split(persons, train_ratio, args.seed)

    # 出力構築
    train_lines, eval_lines = build_outputs(nodes, train_qids, eval_qids)

    # 書き出し
    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.eval_out), exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as f:
        for obj in train_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(args.eval_out, "w", encoding="utf-8") as f:
        for obj in eval_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # レポート
    print("=== Split Report ===")
    print(f"Input:         {inp}")
    print(f"Train out:     {args.train_out}")
    print(f"Eval out:      {args.eval_out}")
    print(f"Ratio (train): {train_ratio:.4f}")
    print(f"Seed:          {args.seed}")
    print(f"Stratified:    {not args.no_stratify}")
    print(f"People total:  {len(persons)}")
    print(f"Train people:  {len(train_qids)}")
    print(f"Eval people:   {len(eval_qids)}")

if __name__ == "__main__":
    main()
