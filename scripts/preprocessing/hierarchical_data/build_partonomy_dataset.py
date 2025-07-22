#!/usr/bin/env python3
"""
collect_trees.py  ─ WDQS から P361/P527 ツリーを収集
架空世界(Q1778821, Q11755880)・天体(Q6999) を除外
"""

import json, time, random, argparse, sys
from collections import deque
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

ENDPOINT = "https://query.wikidata.org/sparql"

EXCLUDED_CLASSES = {
    "Q6999",      # astronomical object
    "Q1778821",   # fictional universe
    "Q11755880"   # fictional location
}

# ----------------------------------------------------------------------
def run_query(query: str, user_agent: str,
              retry: int = 5, base_sleep: float = 2.0):
    """
    WDQS へクエリを投げ JSON で結果を返す。
    429, 504, connection error 等が起こったら指数バックオフでリトライ。
    """
    sleep = base_sleep
    for n in range(retry):
        try:
            sparql = SPARQLWrapper(ENDPOINT)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.addCustomHttpHeader("User-Agent", user_agent)
            sparql.setTimeout(60_000)          # 60 秒
            return sparql.query().convert()["results"]["bindings"]
        except Exception as e:
            if n == retry - 1:
                raise
            time.sleep(sleep + random.random())  # jitter
            sleep *= 2.0
    return []  # never reached


def ask_subclass(item_qid: str, class_qid: str, user_agent: str) -> bool:
    """
    item が class_qid の (直接/間接) instance/subclass か ASK で判定
    """
    res = run_query(f"""
        ASK {{
          wd:{item_qid} (wdt:P31|wdt:P279|wdt:P31/wdt:P279*) wd:{class_qid}
        }}
    """, user_agent)
    return res[0]["boolean"] if res else False


def get_label_desc(qid: str, user_agent: str):
    r = run_query(f"""
        SELECT ?l ?d WHERE {{
          wd:{qid} rdfs:label ?l FILTER(LANG(?l) IN ('ja','en')) .
          OPTIONAL {{ wd:{qid} schema:description ?d FILTER(LANG(?d) IN ('ja','en')) }}
        }} LIMIT 1
    """, user_agent)
    if r:
        l = r[0]['l']['value']
        d = r[0].get('d', {}).get('value', '')
    else:
        l, d = qid, ''
    return l, d

# ----------------------------------------------------------------------
def build_subtree(root_qid: str, user_agent: str,
                  max_depth: int, visited: set):
    edges = []
    stack = deque([(root_qid, 0)])
    while stack:
        parent, depth = stack.pop()
        if depth >= max_depth:
            continue
        childs = run_query(f"""
            SELECT ?c ?cLabel WHERE {{
              wd:{parent} wdt:P527 ?c .
              ?c wdt:P361 wd:{parent}.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ja,en" }}
            }}
        """, user_agent)
        for b in childs:
            cqid = b['c']['value'].split('/')[-1]
            if cqid in visited:
                continue
            edges.append({
                "property": "P527",
                "target_qid": cqid,
                "target_label": b['cLabel']['value']
            })
            visited.add(cqid)
            stack.append((cqid, depth + 1))
    return edges

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', default='trees.jsonl')
    ap.add_argument('-n', '--trees', type=int, default=50)
    ap.add_argument('--min-edges', type=int, default=10)
    ap.add_argument('--max-depth', type=int, default=3)
    ap.add_argument('--user-agent', required=True,
                    help="e.g. 'TreeCollector/1.0 (mailto:you@example.com)'")
    args = ap.parse_args()

    ua = args.user_agent

    # -------------------------------------------------- 1) root 候補
    root_query = f"""
    SELECT ?item (COUNT(?part) AS ?cnt) WHERE {{
      ?item wdt:P527 ?part .
      ?item wdt:P361 ?parent .
    }} GROUP BY ?item HAVING(?cnt > 5) ORDER BY DESC(?cnt) LIMIT 500
    """
    roots = [r['item']['value'].split('/')[-1]
             for r in run_query(root_query, ua)]

    trees, visited_global = [], set()

    for root in tqdm(roots, desc="collect"):
        if len(trees) >= args.trees:
            break
        if root in visited_global:
            continue

        # ---------- 除外タイプ判定 ----------
        if any(ask_subclass(root, ex, ua) for ex in EXCLUDED_CLASSES):
            continue

        edges = build_subtree(root, ua, args.max_depth, visited=set([root]))
        if len(edges) < args.min_edges:
            continue

        # 親(P361) 1 個だけ取得
        parent = run_query(f"""
            SELECT ?p ?pLabel WHERE {{
              wd:{root} wdt:P361 ?p .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ja,en" }}
            }} LIMIT 1
        """, ua)
        if parent:
            pqid = parent[0]['p']['value'].split('/')[-1]
            edges.insert(0, {
                "property": "P361",
                "target_qid": pqid,
                "target_label": parent[0]['pLabel']['value']
            })

        title, desc = get_label_desc(root, ua)
        trees.append({
            "tree_id": len(trees) + 1,
            "wiki_title": title,
            "qid": root,
            "description": desc,
            "edges": edges,
            "num_edges": len(edges),
            "source_props": ["P361", "P527"]
        })
        visited_global.add(root)

    # -------------------------------------------------- 出力
    with open(args.output, 'w', encoding='utf-8') as fw:
        for t in trees:
            fw.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"saved {len(trees)} trees → {args.output}")

# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()