#!/usr/bin/env python
# =============================================================================
# eval_tree.py - Tree Distance Evaluation (JRF-compliant)
# =============================================================================
"""
Evaluate predicted trees against gold trees using *true* Jaccard-Robinson-Foulds
distance (Robinson & Foulds 1981, Pompei et al. 2012).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from itertools import combinations
from math import comb  

from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr


import util
from embeddings import EmbeddingConfig, EmbeddingModel
from hierarchy_node import HierarchyNode
from html_tree_encoding import HTMLTreeEncoding as TreeEncoding
from multibranch_tree_encoding import MultiBranchTreeEncoding
import template

from IPython import embed  # for debugging

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

from datetime import datetime
import time
from contextlib import contextmanager

@contextmanager
def timed(label: str):
    start_wall = datetime.now()
    start_perf = time.perf_counter()
    print(f"[{label}] start: {start_wall:%Y-%m-%d %H:%M:%S.%f}")
    try:
        yield
    finally:
        end_wall = datetime.now()
        elapsed = time.perf_counter() - start_perf
        print(f"[{label}] end  : {end_wall:%Y-%m-%d %H:%M:%S.%f} (elapsed: {elapsed:.3f}s)\n")


# --------------------------------------------------------------------------- #
# 1. Gold tree utilities（元コードから変更なし）
# --------------------------------------------------------------------------- #
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
        if self.is_entity:
            return {self.name}
        entities = set()
        for child in self.children:
            entities.update(child.get_entity_leaves())
        return entities


def build_gold_tree(df: pd.DataFrame
                    ) -> Tuple[Dict[str, List[str]], List[str], Dict[str, str]]:
    """Build gold tree structure from JSONL data."""
    entity_names: List[str] = []
    profession_map: Dict[str, str] = {}
    category_to_entities: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        name = row['wiki_title']
        is_entity = row['is_entity']

        if is_entity:
            entity_names.append(name)
            if 'edges' in row and row['edges']:
                profession = row['edges'][0]['target_label']
                profession_map[name] = profession
                category_to_entities.setdefault(profession, []).append(name)
        else:
            profession_map[name] = name

    entity_names.sort()
    return category_to_entities, entity_names, profession_map


def reconstruct_entity_tree_adjacency(
        df: pd.DataFrame, entity_names: List[str]
) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """Reconstruct gold tree adjacency preserving original hierarchy."""
    entity_to_idx = {n: i for i, n in enumerate(entity_names)}
    n_entities = len(entity_names)
    node_labels: Dict[int, str] = {}
    node_name_to_id: Dict[str, int] = {}
    next_internal_id = n_entities

    for idx, name in enumerate(entity_names):
        node_labels[idx] = name
        node_name_to_id[name] = idx

    category_nodes = df[df['is_entity'] == False]
    name_to_children: Dict[str, List[str]] = {}

    for _, row in category_nodes.iterrows():
        cat = row['wiki_title']
        edges = row.get('edges', [])
        if cat not in node_name_to_id:
            node_name_to_id[cat] = next_internal_id
            node_labels[next_internal_id] = cat
            next_internal_id += 1
        name_to_children[cat] = [e['target_label'] for e in edges]

    adjacency: Dict[int, List[int]] = {}
    def would_create_cycle(parent: int, child: int) -> bool:
        stack = [child]
        while stack:
            cur = stack.pop()
            if cur == parent:
                return True
            stack.extend(adjacency.get(cur, []))
        return False

    for cat, children in name_to_children.items():
        pid = node_name_to_id[cat]
        adjacency[pid] = []
        for child in children:
            if child in node_name_to_id:
                cid = node_name_to_id[child]
                if not would_create_cycle(pid, cid):
                    adjacency[pid].append(cid)
        if not adjacency[pid]:
            del adjacency[pid]  # remove empty internal node

    return adjacency, node_labels

# --------------------------------------------------------------------------- #
# 2. **True JRF distance implementation**
# --------------------------------------------------------------------------- #
def _split_jaccard(a: frozenset, b: frozenset, n_leaves: int) -> float:
    # 片側しか入っていないので補集合も試す
    comp_a = frozenset(range(n_leaves)) - a
    comp_b = frozenset(range(n_leaves)) - b
    j1 = len(a & b) / len(a | b)
    j2 = len(a & comp_b) / len(a | comp_b)
    # もう片方（comp_a と …）は左右入れ替えなので同じ値
    return max(j1, j2)

def _collect_splits(adj, n_leaves):
    splits = set()
    def leaves_under(node, visited):
        if node in visited: return set()
        visited.add(node)
        if node < n_leaves: return {node}
        leaves = set()
        for ch in adj.get(node, []):
            leaves |= leaves_under(ch, visited.copy())
        return leaves

    for parent, children in adj.items():
        for ch in children:                       # ← 子ごとに edge-split
            subset = leaves_under(ch, set())
            if 0 < len(subset) < n_leaves:
                # 小さい側を代表にして正規化
                rep = frozenset(subset if len(subset) <= n_leaves/2
                                else set(range(n_leaves)) - subset)
                splits.add(rep)
    return list(splits)



def jaccard_robinson_foulds_distance(
        tree1_adj: Dict[int, List[int]],
        tree2_adj: Dict[int, List[int]],
        n_leaves: int,
        k: int = 1
) -> float:
    """
    Compute JRF distance (Pompei et al. 2012) between two trees.

    Parameters
    ----------
    tree1_adj, tree2_adj : adjacency dicts
    n_leaves             : number of leaf nodes
    k                    : JRF parameter (typically 1 or 2)

    Returns
    -------
    distance : float   (0 = identical, 2·m = maximally different,
                        m = max(#splits1, #splits2))
    """
    splits1 = _collect_splits(tree1_adj, n_leaves)
    splits2 = _collect_splits(tree2_adj, n_leaves)

    m, n = len(splits1), len(splits2)
    size = max(m, n)                       # square matrix for Hungarian
    cost = np.ones((size, size))           # default cost 1 (no match)

    for i, s1 in enumerate(splits1):
        for j, s2 in enumerate(splits2):
            sim = _split_jaccard(s1, s2, n_leaves) ** k
            cost[i, j] = 1.0 - sim         # turn similarity into cost

    row_ind, col_ind = linear_sum_assignment(cost)
    total_sim = (1.0 - cost[row_ind, col_ind]).sum()

    max_pairs = size                       # = max(m, n)
    # JRF distance = 2 * (max_pairs − Σ similarity)
    return 2.0 * (max_pairs - total_sim)

# --------------------------------------------------------------------------- #
# 2.5. Quartet distance (supports multifurcating gold; pred is binary)
# --------------------------------------------------------------------------- #
def _compute_leaf_sets(adjacency: Dict[int, List[int]], n_leaves: int) -> Dict[int, Set[int]]:
    """
    Compute and memoize the leaf set under each node.
    Leaves are [0 .. n_leaves-1]. Non-listed nodes simply have empty children.
    """
    memo: Dict[int, Set[int]] = {}

    # Collect all nodes that appear anywhere
    nodes = set(adjacency.keys()) | {c for cs in adjacency.values() for c in cs} | set(range(n_leaves))

    def dfs(u: int) -> Set[int]:
        if u in memo:
            return memo[u]
        if u < n_leaves:
            memo[u] = {u}
            return memo[u]
        s: Set[int] = set()
        for v in adjacency.get(u, []):
            s |= dfs(v)
        memo[u] = s
        return s

    for node in nodes:
        dfs(node)
    return memo



def _normalize_split_bitset(subset_bits: int, k: int, n_leaves: int) -> int:
    """
    無向 split の正規化：小さい側を代表に。k == n-k（真っ二つ）のときは
    subset/complement のビット表現のうち整数として小さい方を代表にする。
    """
    universe = (1 << n_leaves) - 1
    comp_bits = universe ^ subset_bits
    if k < (n_leaves - k):
        return subset_bits
    if k > (n_leaves - k):
        return comp_bits
    # k == n-k（タイ）
    return min(subset_bits, comp_bits)


def _collect_unique_split_bitsets(adjacency: Dict[int, List[int]], n_leaves: int) -> Set[int]:
    """
    木のユニークな無向 split をビットセット（int）で収集。
    各 edge-split（親→子）の子側にある葉集合を取り、小さい側（タイは上記関数）を代表に。
    """
    if n_leaves <= 1:
        return set()

    # 既存の leaf-set メモ化を利用
    leaf_sets = _compute_leaf_sets(adjacency, n_leaves)
    universe = set(range(n_leaves))
    splits: Set[int] = set()

    def to_bits(leaves: Set[int]) -> int:
        b = 0
        for i in leaves:
            b |= (1 << i)
        return b

    for parent, children in adjacency.items():
        for ch in children:
            subset = leaf_sets.get(ch, set())
            k = len(subset)
            if 0 < k < n_leaves:
                rep_bits = _normalize_split_bitset(to_bits(subset), k, n_leaves)
                splits.add(rep_bits)
    return splits


def _resolve_quartet_topology(splits: Set[int], n_leaves: int, a: int, b: int, c: int, d: int) -> int:
    if not splits:
        return 0

    mask_all = (1 << n_leaves) - 1

    def pair_mask(x: int, y: int) -> int:
        return (1 << x) | (1 << y)

    ab = pair_mask(a, b); cd = pair_mask(c, d)
    ac = pair_mask(a, c); bd = pair_mask(b, d)
    ad = pair_mask(a, d); bc = pair_mask(b, c)

    for s in splits:
        sc = (~s) & mask_all

        # ab|cd
        if ((s & ab) == ab and (sc & cd) == cd) or ((s & cd) == cd and (sc & ab) == ab):
            return 1
        # ac|bd
        if ((s & ac) == ac and (sc & bd) == bd) or ((s & bd) == bd and (sc & ac) == ac):
            return 2
        # ad|bc
        if ((s & ad) == ad and (sc & bc) == bc) or ((s & bc) == bc and (sc & ad) == ad):
            return 3

    return 0


# def quartet_distance(
#     tree_gold: Dict[int, List[int]],
#     tree_pred: Dict[int, List[int]],
#     n_leaves: int,
#     normalize: bool = True
# ) -> Tuple[float, int, int, int]:
#     """
#     標準 Quartet Distance（両木のトポロジー一致=0、不一致=1。両方未解像は0にカウント）
#     戻り値: (qd, total_quartets, gold_resolved, pred_resolved)
#     """
#     if n_leaves < 4:
#         return 0.0, 0, 0, 0

#     total_quartets = comb(n_leaves, 4)
#     Sg = _collect_unique_split_bitsets(tree_gold, n_leaves)
#     Sp = _collect_unique_split_bitsets(tree_pred, n_leaves)

#     mismatches = 0
#     gold_resolved = 0
#     pred_resolved = 0

#     # 4点組の直接評価
#     for a, b, c, d in combinations(range(n_leaves), 4):
#         tg = _resolve_quartet_topology(Sg, n_leaves, a, b, c, d)
#         tp = _resolve_quartet_topology(Sp, n_leaves, a, b, c, d)
#         if tg != 0:
#             gold_resolved += 1
#         if tp != 0:
#             pred_resolved += 1
#         # 少なくとも片方が解像していて、トポロジーが異なるなら不一致
#         if (tg != 0 or tp != 0) and (tg != tp):
#             mismatches += 1

#     qd_raw = mismatches
#     qd = (qd_raw / total_quartets) if normalize else float(qd_raw)
#     return qd, total_quartets, gold_resolved, pred_resolved


# def generalized_quartet_distance(
#     tree_gold: Dict[int, List[int]],
#     tree_pred: Dict[int, List[int]],
#     n_leaves: int
# ) -> Tuple[float, int, int, int, int]:
#     """
#     Generalized Quartet Distance（GQD）
#       gold で解像している quartet に限定し、同じトポロジーでないものを分子に。
#       戻り値: (gqd, resolved_in_gold, resolved_in_pred, shared_resolved, total_quartets)
#     """
#     if n_leaves < 4:
#         return 0.0, 0, 0, 0, 0

#     total_quartets = comb(n_leaves, 4)
#     Sg = _collect_unique_split_bitsets(tree_gold, n_leaves)
#     Sp = _collect_unique_split_bitsets(tree_pred, n_leaves)

#     Rg = 0     # gold resolved
#     Rp = 0     # pred resolved
#     shared = 0 # 両方解像かつ同じトポロジー

#     for a, b, c, d in combinations(range(n_leaves), 4):
#         tg = _resolve_quartet_topology(Sg, n_leaves, a, b, c, d)
#         tp = _resolve_quartet_topology(Sp, n_leaves, a, b, c, d)
#         if tg != 0:
#             Rg += 1
#             if tp != 0:
#                 Rp += 1
#                 if tg == tp:
#                     shared += 1
#         else:
#             if tp != 0:
#                 Rp += 1

#     if Rg == 0:
#         # 定義上 0 とする（規約）。NaN 回避。
#         return 0.0, Rg, Rp, shared, total_quartets

#     gqd = 1.0 - (shared / Rg)
#     return gqd, Rg, Rp, shared, total_quartets



# --------------- helpers ---------------------------

def _c2(x: int) -> int:
    return (x * (x - 1)) // 2

def _c4(n: int) -> int:
    return (n * (n - 1) * (n - 2) * (n - 3)) // 24

def _popcount(x: int) -> int:
    try:
        return x.bit_count()  # Python 3.8+
    except AttributeError:
        return bin(x).count("1")

def _parent_map(adjacency: Dict[int, List[int]]) -> Dict[int, int]:
    return {c: p for p, cs in adjacency.items() for c in cs}

def _leafsets_bits(adjacency: Dict[int, List[int]], n_leaves: int) -> Dict[int, int]:
    """各ノード配下の葉集合をintビット集合で返す（正しいポストオーダー集計）。"""
    # すべてのノードを収集
    nodes = set(adjacency.keys()) | {c for cs in adjacency.values() for c in cs} | set(range(n_leaves))
    children = adjacency

    # トポロジカル順（根→葉）
    order = []
    indeg = {u: 0 for u in nodes}
    for p, cs in children.items():
        for c in cs:
            indeg[c] = indeg.get(c, 0) + 1
    from collections import deque
    q = deque([u for u in nodes if indeg.get(u, 0) == 0])
    while q:
        u = q.popleft()
        order.append(u)
        for v in children.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # 逆順（葉→根）で集計
    bits: Dict[int, int] = {}
    for u in reversed(order):
        if u < n_leaves:
            bits[u] = 1 << u
        else:
            b = 0
            for v in children.get(u, []):
                b |= bits.get(v, 0)
            bits[u] = b
    return bits

# --------------- components utilities ---------------------------

def _components_around_each_node(
    adjacency: Dict[int, List[int]],
    n: int,
    bits: Dict[int, int],
    parent: Dict[int, int],
) -> Dict[int, List[int]]:
    """
    各内部ノード v に接続する“成分”をビット集合で列挙する。
      - 子側成分: 各子 c の bits[c]
      - 親側成分: root 以外は U ^ bits[v]
    """
    U = (1 << n) - 1
    comps: Dict[int, List[int]] = {}
    for v, cs in adjacency.items():
        lst = []
        for c in cs:
            b = bits[c]
            if b:
                lst.append(b)
        if v in parent:  # 親側成分
            up = U ^ bits[v]
            if up:
                lst.append(up)
        comps[v] = lst
    return comps

def _count_star_quartets(
    adjacency: Dict[int, List[int]],
    n: int,
    bits: Dict[int, int],
    parent: Dict[int, int],
) -> int:
    """star quartets の総数：各ノード周りで4成分から1枚ずつ選ぶ積の総和。"""
    comps = _components_around_each_node(adjacency, n, bits, parent)
    total = 0
    for v, lst in comps.items():
        sizes = [_popcount(b) for b in lst if _popcount(b) > 0]
        m = len(sizes)
        if m < 4:
            continue
        # Σ_{i<j<k<l} s_i s_j s_k s_l
        # 小 m 向けに四重ループで十分（木の次数は小さいことが多い）
        for a in range(m):
            sa = sizes[a]
            if sa == 0:  # 念のため
                continue
            for b in range(a + 1, m):
                sb = sizes[b]
                if sb == 0:
                    continue
                for c in range(b + 1, m):
                    sc = sizes[c]
                    if sc == 0:
                        continue
                    for d in range(c + 1, m):
                        sd = sizes[d]
                        if sd == 0:
                            continue
                        total += sa * sb * sc * sd
    return total

# --------------- GQD: gold-reference version ---------------------------

def generalized_quartet_distance(
    tree_gold: Dict[int, List[int]],
    tree_pred: Dict[int, List[int]],
    n_leaves: int
) -> Tuple[float, int, int, int, int]:
    """
    GQD（Generalized Quartet Distance, gold基準）。
    定義：gqd = 1 - ( shared_butterflies / q_gold )
      - q_gold: gold 木における butterfly quartets 総数
                = C(n,4) - （gold の star quartets 総数）
      - shared_butterflies: gold/pred の両方で同一トポロジーの butterfly に属する四葉組の数
    返り値: (gqd, q_gold, q_pred, shared_butterflies, C(n,4))
    """
    if n_leaves < 4:
        return 0.0, 0, 0, 0, 0

    # 準備
    bits_g = _leafsets_bits(tree_gold, n_leaves)
    bits_p = _leafsets_bits(tree_pred, n_leaves)
    parent_g = _parent_map(tree_gold)
    parent_p = _parent_map(tree_pred)
    U4 = comb(n_leaves, 4)

    # 分母 q_gold（正）
    stars_g = _count_star_quartets(tree_gold, n_leaves, bits_g, parent_g)
    q_gold = U4 - stars_g
    if q_gold <= 0:
        # gold が完全スター（解像なし）の場合
        return 0.0, 0, 0, 0, U4

    # 参考情報として pred 側の butterfly 総数も出す（Rp 相当）
    stars_p = _count_star_quartets(tree_pred, n_leaves, bits_p, parent_p)
    q_pred = U4 - stars_p

    # 共有 butterfly 数（同一トポロジー）
    # ノード×ノードの成分交差行列 M_{i,j} を使う
    comps_g = _components_around_each_node(tree_gold, n_leaves, bits_g, parent_g)
    comps_p = _components_around_each_node(tree_pred, n_leaves, bits_p, parent_p)

    shared = 0
    for v, Sv in comps_g.items():
        # 0 サイズ成分は除外
        Sv = [b for b in Sv if _popcount(b) > 0]
        if len(Sv) < 2:
            continue
        for w, Tw in comps_p.items():
            Tw = [b for b in Tw if _popcount(b) > 0]
            if len(Tw) < 2:
                continue

            # 交差行列 M[i][j] = |Sv[i] ∩ Tw[j]|
            m = len(Sv)
            n = len(Tw)
            M = [[0] * n for _ in range(m)]
            for i in range(m):
                row_mask = Sv[i]
                for j in range(n):
                    M[i][j] = _popcount(row_mask & Tw[j])

            # Σ_{i<j} Σ_{p<q} [ C(M_ip,2)C(M_jq,2) + C(M_iq,2)C(M_jp,2) ]
            for i in range(m):
                for j in range(i + 1, m):
                    Mi = M[i]
                    Mj = M[j]
                    for p in range(n):
                        mip = Mi[p]
                        if mip >= 2:
                            c_mip = _c2(mip)
                        else:
                            c_mip = 0
                        for q in range(p + 1, n):
                            miq = Mi[q]
                            mjp = Mj[p]
                            mjq = Mj[q]

                            term1 = (_c2(mip) if mip >= 2 else 0) * (_c2(mjq) if mjq >= 2 else 0)
                            term2 = (_c2(miq) if miq >= 2 else 0) * (_c2(mjp) if mjp >= 2 else 0)
                            if term1 or term2:
                                shared += term1 + term2

    # gqd は [0,1] に収まる
    gqd = 1.0 if shared == 0 else 1.0 - (shared / q_gold)
    # 浮動小数の丸め誤差対策でクリップ
    if gqd < 0.0:
        gqd = 0.0
    elif gqd > 1.0:
        gqd = 1.0

    return gqd, q_gold, q_pred, shared, U4

# --------------- GTD: 高速実装（あなたの式＋バグ修正） ---------------

def _cluster_info(adjacency: Dict[int, List[int]], n_leaves: int):
    """
    各内部ノード v について
      - bits_v: 配下葉のビット集合
      - child_bits: 各子の葉ビット集合
      - sz_v, sum_child_sq: |L(v)| と Σ |L(child)|^2
    """
    bits = _leafsets_bits(adjacency, n_leaves)
    info = []
    for v, cs in adjacency.items():
        if not cs:
            continue
        bv = bits[v]
        sz = _popcount(bv)
        if sz < 2:
            continue
        child_bits = [bits[c] for c in cs]
        sum_sq = sum((_popcount(b)) ** 2 for b in child_bits)
        info.append((v, bv, sz, child_bits, sum_sq))
    return info  # list of tuples

def generalized_triplet_distance(
    tree_gold: Dict[int, List[int]],
    tree_pred: Dict[int, List[int]],
    n_leaves: int
) -> Tuple[float, int, int, int, int]:
    """
    GTD（Generalized Triplet Distance）。
      gtd = 1 - (shared / Rg)
    ここで
      - Rg = gold の解像 triplet 総数
      - shared = gold/pred の両方で同一トポロジーの triplet 数
    返り値: (gtd, Rg, Rp, shared, C(n,3))
    """
    if n_leaves < 3:
        return 0.0, 0, 0, 0, 0

    gold = _cluster_info(tree_gold, n_leaves)
    pred = _cluster_info(tree_pred, n_leaves)

    # gold/pred の解像triplet数
    Rg = 0
    for _, _, sz, child_bits, sum_sq in gold:
        pairs_exact = (sz * sz - sum_sq) // 2
        Rg += pairs_exact * (n_leaves - sz)

    Rp = 0
    for _, _, sz, child_bits, sum_sq in pred:
        pairs_exact = (sz * sz - sum_sq) // 2
        Rp += pairs_exact * (n_leaves - sz)

    if Rg == 0:
        return 0.0, 0, Rp, 0, comb(n_leaves, 3)

    # 共有triplet数
    shared = 0
    for _, Av, szA, A_children, _ in gold:
        for _, Bw, szB, B_children, _ in pred:
            T = _popcount(Av & Bw)
            if T < 2:
                continue
            # 行（gold子ごと）と列（pred子ごと）
            sum_rows_sq = 0
            for Ab in A_children:
                sum_rows_sq += (_popcount(Ab & Bw)) ** 2
            sum_cols_sq = 0
            for Bb in B_children:
                sum_cols_sq += (_popcount(Bb & Av)) ** 2
            sum_cells_sq = 0
            for Ab in A_children:
                Ab_and = Ab & Bw
                if Ab_and == 0:
                    continue
                for Bb in B_children:
                    m = _popcount(Ab_and & Bb)
                    if m:
                        sum_cells_sq += m * m

            pairs_exact_vw = (T * T - sum_rows_sq - sum_cols_sq + sum_cells_sq) // 2
            if pairs_exact_vw == 0:
                continue
            # 外側のcは A∪B の外：|A∪B| = szA + szB - T
            outside = n_leaves - (szA + szB - T)
            if outside > 0:
                shared += pairs_exact_vw * outside

    gtd = 1.0 - (shared / Rg)
    if gtd < 0.0:
        gtd = 0.0
    elif gtd > 1.0:
        gtd = 1.0
    return gtd, Rg, Rp, shared, comb(n_leaves, 3)





# --------------------------------------------------------------------------- #
# 2.55. Triplet distance (rooted; supports multifurcating/forest)
# --------------------------------------------------------------------------- #
def _resolve_triplet_topology_by_lca(
    parent: Dict[int, Optional[int]],
    depth: Dict[int, int],
    a: int, b: int, c: int
) -> int:
    """
    三つ組 {a,b,c} のトポロジーを LCA の深さで判定。
      戻り値: 0 = 未解像（star）, 1 = ab|c, 2 = ac|b, 3 = bc|a
    """
    la = _lca(a, b, parent, depth)
    lb = _lca(a, c, parent, depth)
    lc = _lca(b, c, parent, depth)
    # 別コンポーネントなどで LCA が得られない場合は未解像扱い
    if la is None or lb is None or lc is None:
        return 0

    dab = depth.get(la, -1)
    dac = depth.get(lb, -1)
    dbc = depth.get(lc, -1)

    # 最も深い（=より“近い”共通祖先）ペアが一意に決まるときだけ解像
    if dab > dac and dab > dbc:
        return 1  # ab|c
    if dac > dab and dac > dbc:
        return 2  # ac|b
    if dbc > dab and dbc > dac:
        return 3  # bc|a
    return 0     # タイ → 未解像


def triplet_distance(
    tree_gold: Dict[int, List[int]],
    tree_pred: Dict[int, List[int]],
    n_leaves: int,
    normalize: bool = True
) -> Tuple[float, int, int, int]:
    """
    標準 Triplet Distance（片方または両方が解像していて、トポロジーが異なる三つ組を数える）
    戻り値: (td, total_triplets, gold_resolved, pred_resolved)
    """
    if n_leaves < 3:
        return 0.0, 0, 0, 0

    total_triplets = comb(n_leaves, 3)

    depth_g, parent_g, _ = _compute_depth_map(tree_gold, n_leaves)
    depth_p, parent_p, _ = _compute_depth_map(tree_pred, n_leaves)

    mismatches = 0
    gold_resolved = 0
    pred_resolved = 0

    for a, b, c in combinations(range(n_leaves), 3):
        tg = _resolve_triplet_topology_by_lca(parent_g, depth_g, a, b, c)
        tp = _resolve_triplet_topology_by_lca(parent_p, depth_p, a, b, c)

        if tg != 0:
            gold_resolved += 1
        if tp != 0:
            pred_resolved += 1

        if (tg != 0 or tp != 0) and (tg != tp):
            mismatches += 1

    td_raw = mismatches
    td = (td_raw / total_triplets) if normalize else float(td_raw)
    return td, total_triplets, gold_resolved, pred_resolved


def generalized_triplet_distance(
    tree_gold: Dict[int, List[int]],
    tree_pred: Dict[int, List[int]],
    n_leaves: int
) -> Tuple[float, int, int, int, int]:
    """
    Generalized Triplet Distance（GTD）
      gold で解像している三つ組だけを分母に取り、両木で同一トポロジーの割合を引いた 1 を返す。
      戻り値: (gtd, resolved_in_gold, resolved_in_pred, shared_resolved, total_triplets)
    """
    if n_leaves < 3:
        return 0.0, 0, 0, 0, 0

    total_triplets = comb(n_leaves, 3)

    depth_g, parent_g, _ = _compute_depth_map(tree_gold, n_leaves)
    depth_p, parent_p, _ = _compute_depth_map(tree_pred, n_leaves)

    Rg = 0      # gold で解像
    Rp = 0      # pred で解像
    shared = 0  # 両方解像かつ一致

    for a, b, c in combinations(range(n_leaves), 3):
        tg = _resolve_triplet_topology_by_lca(parent_g, depth_g, a, b, c)
        tp = _resolve_triplet_topology_by_lca(parent_p, depth_p, a, b, c)

        if tg != 0:
            Rg += 1
            if tp != 0:
                Rp += 1
                if tg == tp:
                    shared += 1
        else:
            if tp != 0:
                Rp += 1

    if Rg == 0:
        # 規約により 0（NaN 回避）
        return 0.0, Rg, Rp, shared, total_triplets

    gtd = 1.0 - (shared / Rg)
    return gtd, Rg, Rp, shared, total_triplets


# --------------------------------------------------------------------------- #
# 2.6. Cophenetic correlation between two trees (topology-based)
# --------------------------------------------------------------------------- #
def _build_parent_map(adjacency: Dict[int, List[int]]) -> Dict[int, Optional[int]]:
    parent: Dict[int, Optional[int]] = {}
    for p, cs in adjacency.items():
        parent.setdefault(p, None)
        for c in cs:
            parent[c] = p
    return parent

def _find_roots(adjacency: Dict[int, List[int]], n_leaves: int) -> Set[int]:
    parents = set(adjacency.keys())
    children = {c for cs in adjacency.values() for c in cs}
    roots = parents - children
    # 孤立葉（どこにも登場しない葉）も root とみなす
    for i in range(n_leaves):
        if (i not in parents) and (i not in children):
            roots.add(i)
    return roots

def _compute_depth_map(adjacency: Dict[int, List[int]], n_leaves: int
                       ) -> Tuple[Dict[int, int], Dict[int, Optional[int]], Set[int]]:
    from collections import deque
    parent = _build_parent_map(adjacency)
    roots = _find_roots(adjacency, n_leaves)
    depth: Dict[int, int] = {}
    q = deque()
    for r in roots:
        q.append((r, 0))
    while q:
        u, d = q.popleft()
        if u in depth:
            continue
        depth[u] = d
        for v in adjacency.get(u, []):
            q.append((v, d + 1))
    # 念のため、未到達ノードがあれば0にしておく
    all_nodes = set(parent.keys()) | {c for cs in adjacency.values() for c in cs} | set(range(n_leaves))
    for u in all_nodes:
        depth.setdefault(u, 0)
        parent.setdefault(u, None)
    return depth, parent, roots

def _lca(u: int, v: int,
         parent: Dict[int, Optional[int]],
         depth: Dict[int, int]) -> Optional[int]:
    # 深い方を上に引き上げて揃える
    uu, vv = u, v
    du, dv = depth.get(uu, 0), depth.get(vv, 0)
    while du > dv and uu is not None:
        uu = parent.get(uu, None); du -= 1
    while dv > du and vv is not None:
        vv = parent.get(vv, None); dv -= 1
    # そろえたら同時に遡上
    while uu != vv:
        uu = parent.get(uu, None) if uu is not None else None
        vv = parent.get(vv, None) if vv is not None else None
        if uu is None and vv is None:
            return None
    return uu

def _cophenetic_vector_by_depth(adjacency: Dict[int, List[int]], n_leaves: int) -> np.ndarray:
    """
    各葉ペア (i<j) について LCA の“高さ”を並べたベクトルを返す。
    高さは depth を max_depth から反転した値（root が最大）を使用。
    異なる root 同士のペアは仮想 super-root（max_height+1）で扱う。
    """
    if n_leaves < 2:
        return np.array([], dtype=float)

    depth, parent, roots = _compute_depth_map(adjacency, n_leaves)
    max_depth = max(depth.values()) if depth else 0
    height = {u: (max_depth - d) for u, d in depth.items()}
    super_h = (max(height.values()) if height else 0) + 1.0

    vals = []
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            anc = _lca(i, j, parent, depth)
            h = height.get(anc, super_h) if anc is not None else super_h
            vals.append(float(h))
    return np.asarray(vals, dtype=float)

def cophenetic_correlation_between_trees(
    tree1_adj: Dict[int, List[int]],
    tree2_adj: Dict[int, List[int]],
    n_leaves: int
) -> Tuple[float, int, float]:
    """
    2つの木のトポロジー由来のコフェネティック距離ベクトル同士の Pearson 相関。
    戻り値: (相関係数 r, ペア数 m, p値)
    """
    v1 = _cophenetic_vector_by_depth(tree1_adj, n_leaves)
    v2 = _cophenetic_vector_by_depth(tree2_adj, n_leaves)
    assert v1.shape == v2.shape, "Cophenetic vectors must have the same length."
    m = int(v1.size)

    # データ点が1以下 or 定数ベクトルは未定義 -> NaN を返す
    if m < 2:
        return float("nan"), m, float("nan")
    if np.allclose(v1, v1.mean()) or np.allclose(v2, v2.mean()):
        return float("nan"), m, float("nan")

    r, p = pearsonr(v1, v2)
    return float(r), m, float(p)


# --------------------------------------------------------------------------- #
# 2.7. Ancestor Jaccard (CASet) between two rooted trees
# --------------------------------------------------------------------------- #
def _to_bits(indices: Set[int]) -> int:
    b = 0
    for i in indices:
        b |= (1 << i)
    return b

def _collect_cluster_bitsets(
    adjacency: Dict[int, List[int]],
    n_leaves: int,
    include_super_root: bool = False
) -> Set[int]:
    """
    各内部ノードが抱える葉集合（サイズ>=2）をビット集合(int)で収集。
    同一の葉集合を作る内部ノードが複数あっても set で一意化される。
    include_super_root=True の場合は全葉集合も1つだけ追加する。
    """
    clusters: Set[int] = set()
    if n_leaves < 2:
        return clusters

    leaf_sets = _compute_leaf_sets(adjacency, n_leaves)  # 既存のメモ化関数を利用
    for node, leaves in leaf_sets.items():
        if len(leaves) >= 2:
            clusters.add(_to_bits(leaves))
    if include_super_root:
        clusters.add((1 << n_leaves) - 1)
    return clusters

def caset_ancestor_jaccard(
    tree1_adj: Dict[int, List[int]],
    tree2_adj: Dict[int, List[int]],
    n_leaves: int,
    include_super_root: bool = True
) -> Tuple[float, float, int]:
    """
    CASet（Ancestor Jaccard）距離と類似度を計算。
      - 各葉ペア {i,j} について、両木の「{i,j} を含むクラスタ集合」を取り、
        その Jaccard 類似度の平均を sim、距離を dist=1-sim とする。
      - 空と空のJaccardは1とみなす（両者が等しいため）。
      - include_super_root=True なら全葉クラスタを両木に追加（森でも空集合を回避）。

    Returns
    -------
    (dist, sim, num_pairs)
    """
    m = comb(n_leaves, 2)
    if m == 0:
        return 0.0, 1.0, 0

    C1 = _collect_cluster_bitsets(tree1_adj, n_leaves, include_super_root)
    C2 = _collect_cluster_bitsets(tree2_adj, n_leaves, include_super_root)

    sum_sim = 0.0
    # 事前に list 化でループ高速化
    C1_list = list(C1); C2_list = list(C2)

    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            mask = (1 << i) | (1 << j)
            S1 = {c for c in C1_list if (c & mask) == mask}
            S2 = {c for c in C2_list if (c & mask) == mask}

            if not S1 and not S2:
                sim = 1.0
            else:
                inter = len(S1 & S2)
                union = len(S1) + len(S2) - inter
                sim = (inter / union) if union > 0 else 1.0
            sum_sim += sim

    avg_sim = sum_sim / m
    avg_dist = 1.0 * (1.0 - avg_sim)
    return avg_dist, avg_sim, m



# --------------------------------------------------------------------------- #
# 3. Gold tree binary conversion (two methods)
# --------------------------------------------------------------------------- #
def convert_multibranch_to_binary_left(
        adjacency: Dict[int, List[int]], 
        n_leaves: int
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Convert a multibranch tree to a left-leaning binary tree.
    
    For nodes with >2 children, create intermediate nodes to binarize
    using left-leaning approach (first child on left, rest on right).
    Example: parent -> [A, B, C, D] becomes:
             parent -> [A, internal1]
             internal1 -> [B, internal2]
             internal2 -> [C, D]
    
    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Original multibranch adjacency
    n_leaves : int
        Number of leaf nodes
    
    Returns
    -------
    binary_adj : Dict[int, List[int]]
        Binary tree adjacency
    birth_time : Dict[int, float]
        Birth times for nodes (synthetic)
    """
    binary_adj = {}
    next_internal_id = max(max(adjacency.keys(), default=n_leaves-1), 
                          max((c for cs in adjacency.values() for c in cs), default=n_leaves-1)) + 1
    
    def binarize_left_leaning(children: List[int]) -> Tuple[int, Dict[int, List[int]]]:
        """
        Convert a list of children to left-leaning binary structure.
        Returns root of the binary subtree and the adjacency dict.
        """
        nonlocal next_internal_id
        local_adj = {}
        
        if len(children) == 0:
            return None, {}
        elif len(children) == 1:
            return children[0], {}
        elif len(children) == 2:
            # Create internal node for two children
            new_node = next_internal_id
            next_internal_id += 1
            local_adj[new_node] = children
            return new_node, local_adj
        else:
            # Left-leaning: take first child as left, recursively process rest as right
            first = children[0]
            rest = children[1:]
            
            # Recursively binarize the rest
            right_root, right_adj = binarize_left_leaning(rest)
            
            # Create parent node
            new_node = next_internal_id
            next_internal_id += 1
            local_adj[new_node] = [first, right_root]
            local_adj.update(right_adj)
            
            return new_node, local_adj
    
    # Process each parent node
    for parent, children in adjacency.items():
        if len(children) == 0:
            continue
        elif len(children) == 1:
            binary_adj[parent] = children
        elif len(children) == 2:
            binary_adj[parent] = children
        else:
            # Need to binarize: use left-leaning approach
            root, sub_adj = binarize_left_leaning(children)
            if root is not None:
                # Connect original parent to the root of binarized subtree
                if root in sub_adj:
                    # root is an internal node, connect parent to its children
                    binary_adj[parent] = sub_adj[root]
                    del sub_adj[root]
                else:
                    # root is a single child
                    binary_adj[parent] = [root]
                binary_adj.update(sub_adj)
    
    # Create synthetic birth times based on tree depth
    birth_time = {}
    
    def assign_birth_times(node: int, depth: float = 0.0, visited: Set[int] = None):
        if visited is None:
            visited = set()
        if node in visited:
            return
        visited.add(node)
        birth_time[node] = depth
        for child in binary_adj.get(node, []):
            assign_birth_times(child, depth + 0.1, visited)
    
    # Find roots
    all_children = {c for cs in binary_adj.values() for c in cs}
    roots = set(binary_adj.keys()) - all_children
    
    # Also check for leaf nodes that might be roots
    for i in range(n_leaves):
        if i not in all_children and i not in binary_adj:
            roots.add(i)
    
    for root in roots:
        assign_birth_times(root)
    
    # Ensure all leaf nodes have birth times
    for i in range(n_leaves):
        if i not in birth_time:
            birth_time[i] = 1.0  # Default depth for unconnected leaves
    
    return binary_adj, birth_time


def convert_multibranch_to_binary_balanced(
        adjacency: Dict[int, List[int]], 
        n_leaves: int
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Convert a multibranch tree to a balanced binary tree.
    
    For nodes with >2 children, create intermediate nodes to binarize
    using balanced splitting (divide children list in half recursively).
    Example: parent -> [A, B, C, D] becomes:
             parent -> [internal1, internal2]
             internal1 -> [A, B]
             internal2 -> [C, D]
    
    Parameters
    ----------
    adjacency : Dict[int, List[int]]
        Original multibranch adjacency
    n_leaves : int
        Number of leaf nodes
    
    Returns
    -------
    binary_adj : Dict[int, List[int]]
        Binary tree adjacency
    birth_time : Dict[int, float]
        Birth times for nodes (synthetic)
    """
    binary_adj = {}
    next_internal_id = max(max(adjacency.keys(), default=n_leaves-1), 
                          max((c for cs in adjacency.values() for c in cs), default=n_leaves-1)) + 1
    
    def binarize_children(children: List[int]) -> Tuple[int, Dict[int, List[int]]]:
        """
        Recursively binarize a list of children nodes.
        Returns the root of the binary subtree and updates to adjacency.
        """
        nonlocal next_internal_id
        local_adj = {}
        
        if len(children) == 0:
            return None, {}
        elif len(children) == 1:
            return children[0], {}
        elif len(children) == 2:
            # Create internal node for two children
            new_node = next_internal_id
            next_internal_id += 1
            local_adj[new_node] = children
            return new_node, local_adj
        else:
            # Split children list in half for balanced tree
            mid = len(children) // 2
            left_children = children[:mid]
            right_children = children[mid:]
            
            # Recursively binarize each half
            left_root, left_adj = binarize_children(left_children)
            right_root, right_adj = binarize_children(right_children)
            
            # Create parent node for the two subtrees
            new_node = next_internal_id
            next_internal_id += 1
            local_adj[new_node] = [left_root, right_root]
            
            # Merge adjacencies
            local_adj.update(left_adj)
            local_adj.update(right_adj)
            
            return new_node, local_adj
    
    # Process each parent node
    for parent, children in adjacency.items():
        if len(children) == 0:
            continue
        elif len(children) == 1:
            binary_adj[parent] = children
        elif len(children) == 2:
            binary_adj[parent] = children
        else:
            # Need to binarize: use balanced approach
            root, sub_adj = binarize_children(children)
            if root is not None:
                # Connect original parent to the root of binarized subtree
                if root in sub_adj:
                    # root is an internal node, connect parent to its children
                    binary_adj[parent] = sub_adj[root]
                    del sub_adj[root]
                else:
                    # root is a single child
                    binary_adj[parent] = [root]
                binary_adj.update(sub_adj)
    
    # Create synthetic birth times based on tree depth
    birth_time = {}
    
    def assign_birth_times(node: int, depth: float = 0.0, visited: Set[int] = None):
        if visited is None:
            visited = set()
        if node in visited:
            return
        visited.add(node)
        birth_time[node] = depth
        for child in binary_adj.get(node, []):
            assign_birth_times(child, depth + 0.1, visited)
    
    # Find roots
    all_children = {c for cs in binary_adj.values() for c in cs}
    roots = set(binary_adj.keys()) - all_children
    
    # Also check for leaf nodes that might be roots
    for i in range(n_leaves):
        if i not in all_children and i not in binary_adj:
            roots.add(i)
    
    for root in roots:
        assign_birth_times(root)
    
    # Ensure all leaf nodes have birth times
    for i in range(n_leaves):
        if i not in birth_time:
            birth_time[i] = 1.0  # Default depth for unconnected leaves
    
    return binary_adj, birth_time

# --------------------------------------------------------------------------- #
# 4. Predicted tree builder (modified to support gold_binary variants)
# --------------------------------------------------------------------------- #
def build_predicted_tree(
        entity_names: List[str],
        model_type: str = "gpt2",
        method: str = "last_token",
        layer: Union[int, str] = 0,
        device: str = "cuda",
        template_name: str = "entity_only",
        gold_adj: Dict[int, List[int]] = None,
        n_leaves: int = None,
        verbose: bool = False,
        random_dim: int = 768,
        random_std: float = 1.0,
        random_seed: int = 42
) -> Union[
    Tuple[Dict[int, List[int]], Dict[int, float]],
    Dict[int, Tuple[Dict[int, List[int]], Dict[int, float]]]
]:
    """
    Build predicted tree using hierarchical clustering persistence 
    or gold tree binary conversion.
    
    Parameters
    ----------
    entity_names : List[str]
        List of entity names
    model_type : str
        Model type (gpt2, llama, fasttext, random_emb, gold_binary_left, gold_binary_balanced)
    method : str
        Embedding method
    layer : int
        Layer number for transformer models
    device : str
        Device for computation
    template_name : str
        Template name for text generation
    gold_adj : Dict[int, List[int]]
        Gold tree adjacency (required for gold_binary variants)
    n_leaves : int
        Number of leaf nodes (required for gold_binary variants)
    
    Returns
    -------
    adjacency : Dict[int, List[int]]
        Predicted tree adjacency
    birth_time : Dict[int, float]
        Node birth times
    """
    if model_type == "gold_binary_left":
        if gold_adj is None or n_leaves is None:
            raise ValueError("gold_adj and n_leaves required for gold_binary_left model")
        return convert_multibranch_to_binary_left(gold_adj, n_leaves)
    elif model_type == "gold_binary_balanced":
        if gold_adj is None or n_leaves is None:
            raise ValueError("gold_adj and n_leaves required for gold_binary_balanced model")
        return convert_multibranch_to_binary_balanced(gold_adj, n_leaves)
    
    # Original implementation for other models
    template_str = template.get_template(template_name)
    texts = [template.apply_template(template_str, n) for n in entity_names]


    # Handle random embeddings through EmbeddingModel
    if model_type == "random_emb":
        print(f"{random_dim=}, {random_std=}, {random_seed=}")
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
        embs = embedder.encode(texts, list(entity_names))

        # PyTorch Tensor の場合は numpy に
        if hasattr(embs, "detach"):
            embs = embs.detach().cpu().numpy()
        # (1, N, D) のような 3 次元なら 2 次元に
        if getattr(embs, "ndim", None) == 3:
            embs = embs[0]

        print("Starting hierarchical clustering persistence calculation …")
        hierarchy = HierarchyNode(embs)
        print("Calculating persistence …")
        hierarchy.calculate_persistence()
        return hierarchy.h_nodes_adj, hierarchy.birth_time
        
    else:
        cfg = EmbeddingConfig(model_type=model_type,
                            method=method,
                            layer=layer,
                            device=device,
                            verbose=verbose)
        embs = EmbeddingModel(cfg).encode(texts, entity_names)
        # If layer=="all", we want all layers at once (L, N, D)
        if isinstance(layer, str) and layer == "all":
            assert embs.ndim == 3, "Expected (L, N, D) when layer=='all'."
            # Build a tree for each layer and return a dict of results.
            results: Dict[int, Tuple[Dict[int, List[int]], Dict[int, float]]] = {}
            for li in range(embs.shape[0]):
                # Build hierarchy from the li-th layer embeddings
                hierarchy = HierarchyNode(embs[li])
                hierarchy.calculate_persistence()
                results[li] = (hierarchy.h_nodes_adj, hierarchy.birth_time)
            return results
        else:
            # Fallback: single-layer path (ensure 2D embeddings)
            if embs.ndim == 3:
                # Some encoders might still return (1, N, D). Take the first.
                embs = embs[0]
            hierarchy = HierarchyNode(embs)
            hierarchy.calculate_persistence()
            return hierarchy.h_nodes_adj, hierarchy.birth_time

# --------------------------------------------------------------------------- #
# 5. Visualization helpers（元コードから変更なし）
# --------------------------------------------------------------------------- #
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
    node_colors = {
        i: PROFESSION_COLORS.get(profession_map.get(n, "Person"), "#CCCCCC")
        for i, n in enumerate(entity_names)
    }
    labels = (gold_node_labels if is_gold_tree and gold_node_labels
              else {i: n for i, n in enumerate(entity_names)})

    if is_gold_tree:
        enc = MultiBranchTreeEncoding(
            adjacency, birth_time, n_leaves,
            n_nodes=max(adjacency.keys()) + 1 if adjacency else n_leaves,
            highlights=None, labels=labels, node_colors=node_colors,
            title=title, height_px=1000, width_pct=100, font_size=16,
            group_spacing_multiplier=group_spacing_multiplier,
            sibling_spacing_multiplier=sibling_spacing_multiplier)
    else:
        bin_adj = {p: tuple(c) for p, c in adjacency.items() if len(c) == 2}
        enc = TreeEncoding(
            bin_adj, birth_time, n_leaves,
            n_nodes=max(adjacency.keys()) + 1 if adjacency else n_leaves,
            highlights=None, labels=labels, node_colors=node_colors,
            title=title, height_px=1000, width_pct=100, font_size=16)

    enc.draw(str(output_path))

# --------------------------------------------------------------------------- #
# 6. Main entry
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate tree distance between gold and predicted trees")
    ap.add_argument("--input", required=True, help="Input JSONL (gold tree)")
    ap.add_argument("--output_dir", required=True, help="Output dir")
    ap.add_argument("--model", default="gpt2",
                    choices=["gpt2", "meta-llama/Meta-Llama-3-8B",
                             "fasttext", "random_emb", 
                             "gold_binary_left", "gold_binary_balanced"])
    ap.add_argument("--method", default="last_token",
                    choices=["average", "last_token"])
    ap.add_argument("--layer", default="0",
                    help="Layer index (int) or 'all' to use all transformer layers.")    
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--template", default="entity_only")
    ap.add_argument("--export_visualizations", action="store_true")
    ap.add_argument("--group_spacing_multiplier", type=float, default=10.0)
    ap.add_argument("--sibling_spacing_multiplier", type=float, default=0.8)
    ap.add_argument("--verbose", action="store_true",
                    help="Enable verbose output for debugging")
    ap.add_argument("--random_dim", type=int, default=768,
                    help="Dimension for random embeddings (default: 768)")
    ap.add_argument("--random_std", type=float, default=1.0,
                    help="Standard deviation for random embeddings (default: 1.0)")
    ap.add_argument("--random_seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    args = ap.parse_args()

    df = pd.read_json(args.input, lines=True)
    print(f"Loaded {len(df)} records")

    print("Building gold tree …")
    _, entity_names, prof_map = build_gold_tree(df)
    gold_adj, gold_labels = reconstruct_entity_tree_adjacency(df, entity_names)
    n_leaves = len(entity_names)

    print("Building predicted tree …")
    # Normalize layer argument: int or "all"
    if isinstance(args.layer, str) and args.layer.lower() == "all":
        layer_arg: Union[int, str] = "all"
    else:
        try:
            layer_arg = int(args.layer)
        except Exception:
            raise ValueError("--layer must be an integer or 'all'")
    
    if args.model in ["gold_binary_left", "gold_binary_balanced"]:
        binary_method = "left-leaning" if args.model == "gold_binary_left" else "balanced"
        print(f"  Using gold tree binary conversion ({binary_method}) as predicted tree")
        pred_adj, pred_birth = build_predicted_tree(
            entity_names, args.model, args.method,
            args.layer, args.device, args.template,
            gold_adj=gold_adj, n_leaves=n_leaves)
        
        # Debug: Print some statistics
        print(f"\nDebug Info:")
        print(f"  Gold adjacency size: {len(gold_adj)}")
        print(f"  Predicted adjacency size: {len(pred_adj)}")
        
        # Check a sample of the structure
        gold_sample = list(gold_adj.items())[:3]
        pred_sample = list(pred_adj.items())[:3]
        print(f"  Gold sample: {gold_sample}")
        print(f"  Pred sample: {pred_sample}")
        
        # Count children distribution
        gold_children_counts = {}
        pred_children_counts = {}
        for p, cs in gold_adj.items():
            count = len(cs)
            gold_children_counts[count] = gold_children_counts.get(count, 0) + 1
        for p, cs in pred_adj.items():
            count = len(cs)
            pred_children_counts[count] = pred_children_counts.get(count, 0) + 1
        print(f"  Gold children distribution: {gold_children_counts}")
        print(f"  Pred children distribution: {pred_children_counts}")

    else:
        pred = build_predicted_tree(
            entity_names, args.model, args.method,
            layer_arg, args.device, args.template, 
            verbose=args.verbose, random_dim=args.random_dim,
            random_std=args.random_std, random_seed=args.random_seed)

    # Handle evaluation & saving for either a single layer or all layers.
    def eval_and_save_one(layer_idx: int,
                          pred_adj: Dict[int, List[int]],
                          pred_birth: Dict[int, float]) -> None:
        """Evaluate one layer, print results, and save to disk."""

        # Jaccard-Robinson-Foulds distance
        with timed("Jaccard-Robinson-Foulds distance (k=1)"):
            jrf1 = jaccard_robinson_foulds_distance(gold_adj, pred_adj, n_leaves, k=1)
        with timed("Jaccard-Robinson-Foulds distance (k=2)"):
            jrf2 = jaccard_robinson_foulds_distance(gold_adj, pred_adj, n_leaves, k=2)

        # Generalized quartet distance
        with timed("Generalized quartet distance"):
            gqd, g_res_gold, g_res_pred, g_shared, g_total = generalized_quartet_distance(
                gold_adj, pred_adj, n_leaves
            )
        # Generalized triplet distance
        with timed("Generalized triplet distance"):
            gtd, t_res_gold, t_res_pred, t_shared, t_total2 = generalized_triplet_distance(
                gold_adj, pred_adj, n_leaves
            )

        # Cophenetic correlation
        with timed("Evaluating cophenetic correlation"):
            coph_r, coph_pairs, coph_p = cophenetic_correlation_between_trees(
                gold_adj, pred_adj, n_leaves
            )

        # Debug splits summary (optional)
        splits1 = _collect_splits(gold_adj, n_leaves)
        splits2 = _collect_splits(pred_adj, n_leaves)
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS  (Layer {layer_idx})")
        print("=" * 60)
        print(f"Dataset                : {Path(args.input).name}")
        print(f"Model / Layer          : {args.model} / {layer_idx}")
        print(f"Template               : {args.template}")
        print(f"Entities               : {n_leaves}")
        print("-----------------------------------------------------------")
        print(f"JRF Distance (k=1)     : {jrf1:.4f}")
        print(f"JRF Distance (k=2)     : {jrf2:.4f}")
        print("-----------------------------------------------------------")
        print(f"GTD (gold reference)   : {gtd:.6f}")
        print("-----------------------------------------------------------")
        print(f"GQD (gold reference)   : {gqd:.6f}")
        print("-----------------------------------------------------------")
        print(f"Cophenetic correlation  : {coph_r:.6f}  (pairs = {coph_pairs}, p = {coph_p:.3g})")
        print("-----------------------------------------------------------")
        print(f"Resolved triplets (gold/pred/shared): {t_res_gold} / {t_res_pred} / {int(t_shared)}")
        print(f"Splits (gold/pred)     : {len(splits1)} / {len(splits2)}")

        # Save per-layer results
        results = dict(dataset=str(args.input), model=args.model, layer=layer_idx,
                       template=args.template, n_entities=n_leaves,
                       jrf_k1=jrf1, jrf_k2=jrf2,
                       gtd=float(gtd),
                       triplets_shared_resolved=int(t_shared),
                       gqd=float(gqd),
                       cophenetic_corr=float(coph_r),
                       cophenetic_pairs=int(coph_pairs),
                       cophenetic_p=float(coph_p),
                       gold_internal_nodes=len(gold_adj),
                       pred_internal_nodes=len(pred_adj))
        with open(out_dir / f"evaluation_results_L{layer_idx}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_dir / ('evaluation_results_L'+str(layer_idx)+'.json')}")

        # Optional visualization per layer
        if args.export_visualizations:
            export_tree_visualization(
                pred_adj, pred_birth, entity_names, prof_map,
                out_dir / f"predicted_tree_L{layer_idx}.html",
                f"Predicted Tree – {args.model} L{layer_idx}")
            print(f"Visualization saved to {out_dir / f'predicted_tree_L{layer_idx}.html'}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ["gold_binary_left", "gold_binary_balanced"]:
        print(f"\nGold tree structure:")
        print(f"  Original internal nodes : {len(gold_adj)}")
        print(f"  Binary internal nodes   : {len(pred_adj)}")
        binary_method = "left-leaning" if args.model == "gold_binary_left" else "balanced"
        print(f"  Binary method           : {binary_method}")

        # For gold_binary_* we already computed pred_adj/pred_birth above:
        eval_and_save_one(layer_idx=0, pred_adj=pred_adj, pred_birth=pred_birth)
    else:
        if isinstance(layer_arg, str) and layer_arg == "all":
            # pred is a dict: layer_idx -> (adj, birth)
            for li, (p_adj, p_birth) in pred.items():
                eval_and_save_one(layer_idx=li, pred_adj=p_adj, pred_birth=p_birth)
        else:
            # Single layer as before
            pred_adj, pred_birth = pred
            eval_and_save_one(layer_idx=int(layer_arg), pred_adj=pred_adj, pred_birth=pred_birth)

    if args.export_visualizations:
        print("\nExporting visualizations …")
        # Fake birth times for gold tree (depth-based)
        gold_birth: Dict[int, float] = {}
        def set_birth(n: int, d: float = 0.0):
            gold_birth[n] = d
            for ch in gold_adj.get(n, []):
                set_birth(ch, d + 0.1)
        roots = set(gold_adj) - {c for cs in gold_adj.values() for c in cs}
        for r in roots:
            set_birth(r)

        export_tree_visualization(
            gold_adj, gold_birth, entity_names, prof_map,
            out_dir / "gold_tree.html",
            f"Gold Tree – {Path(args.input).name}",
            is_gold_tree=True, gold_node_labels=gold_labels,
            group_spacing_multiplier=args.group_spacing_multiplier,
            sibling_spacing_multiplier=args.sibling_spacing_multiplier)

        if args.model in ["gold_binary_left", "gold_binary_balanced"]:
            # For gold_binary variants, visualize as binary tree
            binary_method = "Left-leaning" if args.model == "gold_binary_left" else "Balanced"
            export_tree_visualization(
                pred_adj, pred_birth, entity_names, prof_map,
                out_dir / "predicted_tree.html",
                f"Gold Tree ({binary_method} Binary Conversion)",
                is_gold_tree=False)  # Use binary tree visualization
        else:
            # When layer=='all', per-layer visualizations are already saved above.
            if not (isinstance(layer_arg, str) and layer_arg == "all"):
                export_tree_visualization(
                    pred_adj, pred_birth, entity_names, prof_map,
                    out_dir / "predicted_tree.html",
                    f"Predicted Tree – {args.model} L{layer_arg}")

        print("Visualizations exported.")

if __name__ == "__main__":
    main()