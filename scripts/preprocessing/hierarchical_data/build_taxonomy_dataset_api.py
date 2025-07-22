#!/usr/bin/env python3
# coding: utf-8
"""
PopQA の QID から YAGO4.5 のタクソノミー木を生成して JSONL 出力。

制約
  A) エンティティ直上（leaf）階層に兄弟カテゴリ >= min_leaf_categories
  B) その 1 つ上（parent）階層にも兄弟カテゴリ >= min_parent_categories
  C) 各 leaf カテゴリにエンティティ >= min_entities_per_leaf
  D) ツリー深さ >= min_tree_depth, ノード数 >= min_nodes_per_tree

幅の上限
  ・parent カテゴリ数     <= max_parent_categories
  ・leaf   カテゴリ数     <= max_leaf_categories
  ・leaf 内エンティティ数 <= max_entities_per_leaf

pop 値による優先度付けは行わずランダム抽出。
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from datasets import load_dataset
from SPARQLWrapper import SPARQLWrapper, JSON, POST, URLENCODED
from tqdm import tqdm


class TaxonomyDatasetCreator:
    ENDPOINT = "https://yago-knowledge.org/sparql/query"

    # ----------------------------------------------------
    def __init__(self,
                 # 必須条件
                 min_tree_depth: int = 5,
                 min_nodes_per_tree: int = 20,
                 min_parent_categories: int = 2,
                 min_leaf_categories: int = 2,
                 min_entities_per_leaf: int = 4,
                 # 幅の上限
                 max_parent_categories: int = 4,
                 max_leaf_categories: int = 6,
                 max_entities_per_leaf: int = 10,
                 # その他
                 target_trees: int = 300,
                 cache_path: Optional[str] = None,
                 random_seed: int = 42):

        # --- 条件と上限 ---------------------------------
        self.min_tree_depth = min_tree_depth
        self.min_nodes_per_tree = min_nodes_per_tree
        self.min_parent_categories = min_parent_categories
        self.min_leaf_categories = min_leaf_categories
        self.min_entities_per_leaf = min_entities_per_leaf

        self.max_parent_categories = max_parent_categories
        self.max_leaf_categories = max_leaf_categories
        self.max_entities_per_leaf = max_entities_per_leaf

        self.target_trees = target_trees
        self.cache_path = cache_path
        random.seed(random_seed)

        # --- データ構造 ---------------------------------
        self.popqa_entities: Dict[str, Dict] = {}     # QID -> {"pop":…, "wiki_title":…}
        self.entity_types = defaultdict(set)          # QID -> {type}
        self.type_entities = defaultdict(set)         # type -> {QID}
        self.type_hierarchy = defaultdict(set)        # parent -> {child}
        self.trees: Dict[int, Dict[str, Set[str]]] = {}

        # パターン
        self.re_yago = re.compile(r'^http://yago-knowledge\.org/resource/(.+)')
        self.re_schema = re.compile(r'^http://schema\.org/(.+)')

    # --------------- PopQA ---------------------------------
    def load_popqa_entities(self, dataset_name: str):
        print("Loading PopQA entities …")
        ds = load_dataset(dataset_name, split="train")
        for row in ds:
            self.popqa_entities[row["qid"]] = {
                "pop": row.get("pop", 0),
                "wiki_title": row.get("wiki_title", row.get("entity", "")),
            }
        print(f"  → {len(self.popqa_entities):,} entities")

    # --------------- SPARQL helper -------------------------
    def _run_sparql(self, query: str) -> List[dict]:
        sparql = SPARQLWrapper(self.ENDPOINT)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        sparql.setMethod(POST)
        sparql.setRequestMethod(URLENCODED)
        return sparql.query().convert()["results"]["bindings"]

    # --------------- cache I/O -----------------------------
    def _save_cache(self):
        if not self.cache_path:
            return
        data = {qid: list(ts) for qid, ts in self.entity_types.items()}
        opener = gzip.open if self.cache_path.endswith(".gz") else open
        with opener(self.cache_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  → cache saved to {os.path.abspath(self.cache_path)}")

    def _load_cache(self) -> bool:
        if not self.cache_path or not Path(self.cache_path).exists():
            return False
        opener = gzip.open if self.cache_path.endswith(".gz") else open
        with opener(self.cache_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for qid, ts in data.items():
            self.entity_types[qid].update(ts)
            for t in ts:
                self.type_entities[t].add(qid)
        print(f"  → entity-type map loaded from cache ({len(self.entity_types):,} entities)")
        return True

    # --------------- rdf:type ------------------------------
    def fetch_entity_types(self, batch_size: int = 200):
        if self._load_cache():
            return
        print("Fetching rdf:type via owl:sameAs …")
        qids = list(self.popqa_entities.keys())
        for i in tqdm(range(0, len(qids), batch_size)):
            sub = qids[i:i + batch_size]
            values = " ".join(f"wd:{q}" for q in sub)
            query = f"""
            PREFIX owl:  <http://www.w3.org/2002/07/owl#>
            PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX wd:   <http://www.wikidata.org/entity/>
            SELECT ?wd ?type WHERE {{
              VALUES ?wd {{ {values} }}
              ?yago owl:sameAs ?wd ; rdf:type ?type .
              FILTER(STRSTARTS(STR(?type),"http://yago-knowledge.org/resource/")
                     || STRSTARTS(STR(?type),"http://schema.org/"))
            }}"""
            for b in self._run_sparql(query):
                qid = b["wd"]["value"].split("/")[-1]
                typ = self._compact(b["type"]["value"])
                if typ:
                    self.entity_types[qid].add(typ)
                    self.type_entities[typ].add(qid)
        print(f"  → got types for {len(self.entity_types):,} entities")
        self._save_cache()

    # --------------- subclass hierarchy --------------------
    def fetch_type_hierarchy(self):
        print("Downloading rdfs:subClassOf hierarchy …")
        query = textwrap.dedent("""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?parent ?child WHERE {
          ?child rdfs:subClassOf ?parent .
          FILTER(STRSTARTS(STR(?parent),"http://yago-knowledge.org/resource/")
                 && STRSTARTS(STR(?child),"http://yago-knowledge.org/resource/"))
        }""")
        for b in self._run_sparql(query):
            p = self._compact(b["parent"]["value"])
            c = self._compact(b["child"]["value"])
            if p and c:
                self.type_hierarchy[p].add(c)
        print(f"  → {sum(len(v) for v in self.type_hierarchy.values()):,} subclass links")

    # --------------- URI → 文字列 ---------------------------
    def _compact(self, uri: str) -> Optional[str]:
        m = self.re_yago.match(uri)
        if m:
            return m.group(1)
        m = self.re_schema.match(uri)
        if m:
            return "schema_" + m.group(1)
        return None

    # --------------- root までの経路 -------------------------
    def _path_to_root(self, cat: str) -> List[str]:
        """schema_Thing を root として cat までの経路（root を含む）"""
        if cat == "schema_Thing":
            return ["schema_Thing"]
        path = [cat]
        while True:
            parents = [p for p, children in self.type_hierarchy.items() if path[0] in children]
            if not parents:
                break
            path.insert(0, parents[0])
            if len(path) > 25:  # 安全装置
                break
        return path

    # --------------- tree construction ----------------------
    def build_trees(self):
        print("Building trees …")
        # (1) grand-parent 候補 (= entity から 2 つ上)
        grand_parent_candidates = []
        for gp, parent_set in self.type_hierarchy.items():
            good_parents = []
            for parent in parent_set:
                # parent の子 = leaf
                leaf_candidates = [c for c in self.type_hierarchy.get(parent, [])
                                   if len(self.type_entities.get(c, set()) &
                                          self.popqa_entities.keys()) >= self.min_entities_per_leaf]
                if len(leaf_candidates) >= self.min_leaf_categories:
                    good_parents.append((parent, leaf_candidates))
            if len(good_parents) >= self.min_parent_categories:
                grand_parent_candidates.append((gp, good_parents))

        if not grand_parent_candidates:
            print("No hierarchy satisfies sibling constraints.")
            sys.exit(1)

        random.shuffle(grand_parent_candidates)
        tid = 1
        for gp, good_parents in grand_parent_candidates:
            if tid > self.target_trees:
                break

            # root → … → gp の経路
            path_root_to_gp = self._path_to_root(gp)
            if path_root_to_gp and path_root_to_gp[0] == "schema_Thing":
                path_root_to_gp = path_root_to_gp[1:]

            # 深さチェック (gp までの長さ + parent + leaf + entity = +3)
            if len(path_root_to_gp) + 3 < self.min_tree_depth:
                continue

            tree = defaultdict(set)

            # root → … → gp
            for i in range(len(path_root_to_gp) - 1):
                tree[path_root_to_gp[i]].add(path_root_to_gp[i + 1])

            # gp → parent (幅制限)
            parent_cats = random.sample(
                good_parents,
                k=min(len(good_parents), self.max_parent_categories))

            for parent_cat, leaf_cats in parent_cats:
                tree[gp].add(parent_cat)

                # parent → leaf (幅制限)
                leaf_sel = random.sample(
                    leaf_cats,
                    k=min(len(leaf_cats), self.max_leaf_categories))
                for leaf_cat in leaf_sel:
                    tree[parent_cat].add(leaf_cat)

                    # leaf → entities (幅制限)
                    ents = list(self.type_entities[leaf_cat] & self.popqa_entities.keys())
                    random.shuffle(ents)
                    ents = ents[: self.max_entities_per_leaf]
                    for q in ents:
                        tree[leaf_cat].add(q)

            # ノード数チェック
            nodes = set(tree.keys()) | {c for v in tree.values() for c in v}
            if len(nodes) < self.min_nodes_per_tree:
                continue

            self.trees[tid] = tree
            tid += 1

        print(f"  → built {len(self.trees)} trees")

    # --------------- ラベル整形 ------------------------------
    _re_qid = re.compile(r'\bQ\d+\b', re.IGNORECASE)
    _re_ucode = re.compile(r'[Uu]([0-9A-Fa-f]{4})')

    def _clean_label(self, raw: str) -> str:
        # 1) アンダースコア除去 & schema_ 前置詞除去
        s = raw.replace('_', ' ').replace('schema ', '')

        # 2) U002E/U0028 … を Unicode 文字に変換
        s = self._re_ucode.sub(lambda m: chr(int(m.group(1), 16)), s)

        # 3) 記号前後の余計な空白を除去
        #    ・ピリオド/カンマ/コロン/セミコロン/括弧
        s = re.sub(r'\s*([.,;:])\s*', r'\1 ', s)   # “U . S .” → “U.S.”
        s = re.sub(r'\(\s+', '(',  s)              # “( Business” → “(Business”
        s = re.sub(r'\s+\)', ')',  s)              # “Business )” → “Business)”

        # 4) 連続空白を 1 つに
        s = re.sub(r'\s+', ' ', s).strip()

        # 5) QID が残っていれば除去
        s = self._re_qid.sub('', s).strip()

        # 6) 1 文字目だけ大文字化（“u.s. state” → “U.S. State”）
        if s:
            s = s[0].upper() + s[1:]

        return s

    def _label(self, node: str) -> str:
        if node in self.popqa_entities:
            return self.popqa_entities[node]["wiki_title"]
        return self._clean_label(node)

    # --------------- JSONL 出力 ------------------------------
    def to_jsonl(self) -> List[str]:
        lines = []
        for tid, tree in self.trees.items():
            nodes = set(tree.keys()) | {c for v in tree.values() for c in v}
            for n in nodes:
                label = self._label(n)
                if not label:   # 追記: wiki_titleが""ならスキップ
                    continue
                edges = []
                # 下向き P527
                if n in tree:
                    edges += [{"property": "P527",
                               "target_qid": c,
                               "target_label": self._label(c)} for c in tree[n]]
                # 上向き P361
                for p, ch in tree.items():
                    if n in ch:
                        edges.append({"property": "P361",
                                      "target_qid": p,
                                      "target_label": self._label(p)})
                data = {
                    "tree_id": tid,
                    "wiki_title": label,
                    "qid": n,
                    "num_edges": len(edges),
                    "source_props": list({e["property"] for e in edges}),
                }
                if n in self.popqa_entities:
                    data["pop"] = self.popqa_entities[n]["pop"]
                if edges:
                    data["edges"] = edges
                lines.append(json.dumps(data, ensure_ascii=False))
        return lines

    # --------------- pipeline -------------------------------
    def run(self, outfile: str, popqa_ds: str):
        self.load_popqa_entities(popqa_ds)
        self.fetch_entity_types()
        if not self.entity_types:
            print("No entity-type data — abort.")
            sys.exit(1)
        self.fetch_type_hierarchy()
        self.build_trees()

        with open(outfile, "w", encoding="utf-8") as f:
            f.write("\n".join(self.to_jsonl()))
        print(f"Saved {outfile} ({len(self.trees)} trees)")


# -------------------- CLI -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="taxonomy.jsonl")

    # 必須条件
    ap.add_argument("--min-parent-categories", type=int, default=2)
    ap.add_argument("--min-leaf-categories", type=int, default=2)
    ap.add_argument("--min-entities-per-leaf", type=int, default=4)

    # 幅の上限
    ap.add_argument("--max-parent-categories", type=int, default=3)
    ap.add_argument("--max-leaf-categories", type=int, default=4)
    ap.add_argument("--max-entities-per-leaf", type=int, default=5)

    # 深さ・ノード数・本数
    ap.add_argument("--min-tree-depth", type=int, default=5)
    ap.add_argument("--min-nodes-per-tree", type=int, default=20)
    ap.add_argument("--target-trees", type=int, default=300)

    # その他
    ap.add_argument("--entity-type-cache",
                    help="JSON(.gz) for QID→types cache (read & write)")
    ap.add_argument("--popqa-dataset",
                    default="masaki-sakata/popqa-unique-entities")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    creator = TaxonomyDatasetCreator(
        min_tree_depth=args.min_tree_depth,
        min_nodes_per_tree=args.min_nodes_per_tree,
        min_parent_categories=args.min_parent_categories,
        min_leaf_categories=args.min_leaf_categories,
        min_entities_per_leaf=args.min_entities_per_leaf,
        max_parent_categories=args.max_parent_categories,
        max_leaf_categories=args.max_leaf_categories,
        max_entities_per_leaf=args.max_entities_per_leaf,
        target_trees=args.target_trees,
        cache_path=args.entity_type_cache,
        random_seed=args.seed,
    )
    creator.run(args.output, args.popqa_dataset)


if __name__ == "__main__":
    main()