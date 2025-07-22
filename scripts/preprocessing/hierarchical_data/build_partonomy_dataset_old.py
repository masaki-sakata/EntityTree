#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_partonomy_dataset.py
入力 jsonl (qid 必須) から
  – 固有名詞判定
  – partonomy 木をたどり，深さ >=3 の木だけ残す
  – 各木に tree_id を振る
  – 直接の part-of / has-part エッジも保持
して出力 jsonl を作る。
"""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

###############################################################################
# 1. CLI
###############################################################################
def get_args():
    ap = argparse.ArgumentParser(
        description="Build partonomy dataset (depth ≥ N, QID already present)"
    )
    ap.add_argument("--src", required=True, help="path to input jsonl")
    ap.add_argument("--dst", required=True, help="path to output jsonl")
    ap.add_argument("--sleep", type=float, default=0.1,
                    help="seconds to sleep between SPARQL queries")
    ap.add_argument("--user_agent",
                    default="PartonomyDataset/0.3 (you@example.com)",
                    help="User-Agent string for WDQS / REST API")
    ap.add_argument("--depth_min", type=int, default=3,
                    help="keep trees whose depth is at least this value")
    return ap.parse_args()

###############################################################################
# 2. constants & init
###############################################################################
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

NAMED_ENTITY_ROOTS = [
    "Q5", "Q43229", "Q783794", "Q618123", "Q6256",
    "Q811979", "Q17537576", "Q838948", "Q1190554",
    "Q2424752", "Q167270"
]

def init_sparql(user_agent: str):
    s = SPARQLWrapper(SPARQL_ENDPOINT, agent=user_agent)
    s.setReturnFormat(JSON)
    return s

def run_sparql(sp: SPARQLWrapper, query: str, sleep_sec: float):
    sp.setQuery(query)
    time.sleep(sleep_sec)
    return sp.query().convert()

###############################################################################
# 3. named-entity 判定
###############################################################################
def make_is_named_entity(sparql_obj: SPARQLWrapper, roots: list[str], sleep_sec: float):
    roots_values = " ".join(f"wd:{r}" for r in roots)

    @lru_cache(maxsize=100_000)
    def _inner(qid: str) -> bool:
        q = f"""
        ASK {{
          wd:{qid} wdt:P31/wdt:P279* ?cls .
          VALUES ?cls {{ {roots_values} }}
        }}
        """
        try:
            return run_sparql(sparql_obj, q, sleep_sec)["boolean"]
        except Exception:
            return False
    return _inner

###############################################################################
# 3-a. partonomy helpers
###############################################################################
def make_get_parts(sparql_obj: SPARQLWrapper, sleep_sec: float, is_ne):
    @lru_cache(maxsize=100_000)
    def _inner(qid: str) -> list[str]:
        q = f"""
        SELECT ?c WHERE {{
          wd:{qid} wdt:P527 ?c .
        }}
        """
        try:
            res = run_sparql(sparql_obj, q, sleep_sec)
            cs = [b["c"]["value"].split("/")[-1]
                  for b in res["results"]["bindings"]]
            return [c for c in cs if is_ne(c)]
        except Exception:
            return []
    return _inner

def make_tree_depth(get_parts):
    @lru_cache(maxsize=100_000)
    def _inner(qid: str, visited=frozenset()) -> int:
        if qid in visited:
            return 0
        children = get_parts(qid)
        if not children:
            return 1
        return 1 + max(_inner(c, visited | {qid}) for c in children)
    return _inner

###############################################################################
# 4. SPARQL helpers (エッジ取得・description)
###############################################################################
def get_partonomy_edges(qid: str, sparql_obj, sleep_sec: float, is_ne) -> list[dict]:
    query = f"""
    SELECT ?prop ?target ?targetLabel WHERE {{
      VALUES ?item {{ wd:{qid} }}
      {{
        ?item wdt:P527 ?target .
        BIND("P527" AS ?prop)
      }}
      UNION
      {{
        ?item wdt:P361 ?target .
        BIND("P361" AS ?prop)
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    try:
        res = run_sparql(sparql_obj, query, sleep_sec)
    except Exception:
        return []

    edges = []
    for b in res["results"]["bindings"]:
        tgt_qid = b["target"]["value"].split("/")[-1]
        if is_ne(tgt_qid):
            edges.append({
                "property": b["prop"]["value"],
                "target_qid": tgt_qid,
                "target_label": b["targetLabel"]["value"]
            })
    return edges

def get_description(qid: str, sparql_obj, sleep_sec: float) -> str | None:
    q = f"""
    SELECT ?d WHERE {{
      wd:{qid} schema:description ?d .
      FILTER(LANG(?d)="en")
    }} LIMIT 1
    """
    try:
        res = run_sparql(sparql_obj, q, sleep_sec)
        b = res["results"]["bindings"]
        return b[0]["d"]["value"] if b else None
    except Exception:
        return None

###############################################################################
# 5. main
###############################################################################
def main():
    args = get_args()
    src_path = Path(args.src)
    dst_path = Path(args.dst)

    df = pd.read_json(src_path, lines=True)
    if "qid" not in df.columns:
        raise ValueError("input must contain 'qid' column")
    print(f"Loaded {len(df):,} rows")

    sparql = init_sparql(args.user_agent)
    is_ne = make_is_named_entity(sparql, NAMED_ENTITY_ROOTS, args.sleep)
    get_parts  = make_get_parts(sparql, args.sleep, is_ne)
    tree_depth = make_tree_depth(get_parts)

    # 1) root NE filter
    mask = [is_ne(q) for q in tqdm(df["qid"].tolist(), desc="NE filter")]
    df = df[mask]
    print(f"{len(df):,} rows kept after NE filter")

    # 2) depth filter + edge取得 + tree_id 付与
    records  = []
    next_tid = 1
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching tree"):
        qid = row["qid"]
        depth = tree_depth(qid)
        if depth >= args.depth_min:
            edges = get_partonomy_edges(qid, sparql, args.sleep, is_ne)
            rec = row.to_dict()
            rec["wiki_title"] = rec.pop("entity")
            rec.update({
                "tree_id": next_tid,
                "tree_depth": depth,
                "description": get_description(qid, sparql, args.sleep),
                "edges": edges,
                "num_edges": len(edges),
                "source_props": sorted({e["property"] for e in edges})
            })
            records.append(rec)
            next_tid += 1

    print(f"{len(records):,} rows satisfy depth ≥ {args.depth_min}")

    with dst_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records):,} records to {dst_path}")

if __name__ == "__main__":
    main()