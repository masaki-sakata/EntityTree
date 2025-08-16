#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, sys, time, re
from collections import defaultdict
from functools import lru_cache

import requests
from tqdm import tqdm

import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SESSION = requests.Session()
RETRY = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1.2,  # 1.2, 2.4, 3.6, ...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET", "POST"]),
    raise_on_status=False,
)
ADAPTER = HTTPAdapter(max_retries=RETRY)
SESSION.mount("https://", ADAPTER)
SESSION.mount("http://", ADAPTER)


WD_SPARQL = "https://query.wikidata.org/sparql"
YAGO_SPARQL = "https://yago-knowledge.org/sparql"

HEADERS = {
    "User-Agent": "occupation-subclass-annotator/1.0 (mailto:you@example.com)"
}

# ---- 上位職業（あなたのJSONに合わせた QID） ----
TOP_OCC_QIDS = {
    "Politician": "Q82955",
    "Actor": "Q33999",
    "Athlete": "Q2066131",
    "Musician": "Q639669",
    "Scientist": "Q901",
    "Business Person": "Q43845",
    # 入力 JSON にある表記ぶれを吸収
    "Business person": "Q43845",
    "Business_person": "Q43845",
}

# ---- 下位概念の候補（ラベル）。QID は起動時に SPARQL で解決する ----
LOWER_CANDIDATES = {
    "Politician": {
        # politician は P39（position）から推定
        "positions": ["Legislator", "Governor", "Mayor"],
        "occupations": []  # 明示の下位職業は使わない
    },
    "Actor": {
        "positions": [],
        "occupations": ["Comedian", "Film actor", "Stage actor", "Television actor", "Voice actor"]
    },
    "Athlete": {
        "positions": [],
        "occupations": [
            "Association football player", "Basketball player", "Tennis player", "Golfer",
            "Boxer", "Sprinter", "Baseball player", "Ice hockey player"
        ]
    },
    "Musician": {
        "positions": [],
        "occupations": [
            "Singer", "Composer", "Songwriter", "Conductor", "Instrumentalist",
            "Pianist", "Guitarist", "Drummer", "DJ", "Record producer"
        ]
    },
    "Scientist": {
        "positions": [],
        "occupations": [
            "Physicist", "Chemist", "Mathematician", "Biologist", "Computer scientist",
            "Astronomer", "Economist", "Psychologist", "Neuroscientist", "Engineer"
        ]
    },
    "Business Person": {
        "positions": [],
        "occupations": ["Entrepreneur", "Business executive", "Investor", "Marketer", "Financier"]
    },
    "Business person": {
        "positions": [],
        "occupations": ["Entrepreneur", "Business executive", "Investor", "Marketer", "Financier"]
    },
    "Business_person": {
        "positions": [],
        "occupations": ["Entrepreneur", "Business executive", "Investor", "Marketer", "Financier"]
    },
}

# Politician のポジション判定で使う「アンカー」QID
ANCHOR_QIDS = {
    "Legislator": "Q4175034",
    "Governor": "Q132050",
    "Mayor": "Q30185",
}

# SPARQL ヘルパ
def run_sparql(query, endpoint=WD_SPARQL, retry=5, base_sleep=0.8):
    last_err = None
    for i in range(retry):
        try:
            r = SESSION.post(
                endpoint,
                data={"query": query},
                headers={**HEADERS, "Accept": "application/sparql-results+json"},
                timeout=(10, 90),  # (connect, read)
            )
            # レート制限
            if r.status_code == 429:
                wait = r.headers.get("Retry-After")
                wait_s = int(wait) if wait and wait.isdigit() else (base_sleep * (i + 1))
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            last_err = e
            # エクスポネンシャル + ジッター
            sleep_s = base_sleep * (2 ** i) + random.random() * 0.3
            time.sleep(sleep_s)
            continue
    # 最後に失敗内容を上にあげる
    raise last_err if last_err else RuntimeError("Unknown SPARQL error")


# ラベル → QID 解決（英語/日本語の厳密一致。見つからなければゆるめ一致）
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

@lru_cache(maxsize=4096)
def resolve_qid_by_label(label):
    label_clean = label.replace("_", " ").strip()

    def _search(language):
        params = {
            "action": "wbsearchentities",
            "search": label_clean,
            "language": language,  # まず en、ダメなら ja
            "format": "json",
            "type": "item",
            "limit": 1,
        }
        r = SESSION.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=(10, 30))
        if r.ok:
            js = r.json()
            if js.get("search"):
                return js["search"][0]["id"]  # e.g., "Q4175034"
        return None

    # 1) 軽い検索（英→日）
    qid = _search("en") or _search("ja")
    if qid:
        return qid

    # 2) それでもダメなら「厳密一致のみ」の軽量 SPARQL
    q_exact = f"""
    SELECT ?item WHERE {{
      VALUES ?lbl {{ "{label_clean}"@en "{label_clean}"@ja }}
      ?item rdfs:label ?lbl .
    }} LIMIT 5
    """
    try:
        data = run_sparql(q_exact)
        if data["results"]["bindings"]:
            return data["results"]["bindings"][0]["item"]["value"].split("/")[-1]
    except Exception:
        pass

    # 3) 最後のフォールバック：曖昧検索（重い）だが回数を最小化
    q_fuzzy = f"""
    SELECT ?item WHERE {{
      ?item rdfs:label ?l .
      FILTER (lang(?l) in ('en','ja') && CONTAINS(LCASE(?l), LCASE("{label_clean}")))
    }} LIMIT 3
    """
    try:
        data2 = run_sparql(q_fuzzy)
        if data2["results"]["bindings"]:
            return data2["results"]["bindings"][0]["item"]["value"].split("/")[-1]
    except Exception:
        pass

    return None



# まとめて QID にする
def labels_to_qids(labels):
    out = {}
    for lab in labels:
        if lab in ANCHOR_QIDS:
            out[lab] = ANCHOR_QIDS[lab]
        else:
            try:
                qid = resolve_qid_by_label(lab)
                if qid:
                    out[lab] = qid
            except Exception:
                # ラベル解決に失敗しても他は続ける
                continue
    return out


# P106 ベースで「上位→下位」職業の一致を調べる
def infer_lower_by_P106(person_qid, parent_qid, lower_qids):
    """
    person の P106（職業）から、候補の下位職業に P279* で到達するものを返す。
    lower_qids: {label: qid}
    """
    if not lower_qids:
        return set()
    candidates = " ".join(f"wd:{qid}" for qid in lower_qids.values())
    q = f"""
    SELECT DISTINCT ?lower WHERE {{
      VALUES ?person {{ wd:{person_qid} }}
      VALUES ?parent {{ wd:{parent_qid} }}
      VALUES ?lower {{ {candidates} }}

      ?person wdt:P106 ?occ .
      ?occ wdt:P279* ?lower .
      ?lower wdt:P279* ?parent .
    }}
    """
    data = run_sparql(q)
    out = set()
    for b in data["results"]["bindings"]:
        lower_q = b["lower"]["value"].split("/")[-1]
        out.add(lower_q)
    return out

# P39 ベースで Politician を Legislator/Governor/Mayor に振り分ける
def infer_politician_positions(person_qid, anchors_qids=ANCHOR_QIDS):
    # anchors は Legislator / Governor / Mayor
    values = " ".join(f"wd:{qid}" for qid in anchors_qids.values())
    q = f"""
    SELECT DISTINCT ?anchor WHERE {{
      VALUES ?person {{ wd:{person_qid} }}
      VALUES ?anchor {{ {values} }}
      ?person wdt:P39 ?pos .
      ?pos wdt:P279* ?anchor .
    }}
    """
    data = run_sparql(q)
    out = set()
    for b in data["results"]["bindings"]:
        out.add(b["anchor"]["value"].split("/")[-1])
    return out

# ラベルを取得
@lru_cache(maxsize=8192)
def get_label(qid):
    q = f"""
    SELECT ?l WHERE {{
      wd:{qid} rdfs:label ?l .
      FILTER (lang(?l) IN ('en','ja'))
    }} LIMIT 1
    """
    data = run_sparql(q)
    if data["results"]["bindings"]:
        return data["results"]["bindings"][0]["l"]["value"]
    return qid

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_index(objs):
    by_title = {}
    per_parent_people = defaultdict(list)  # parent_title -> [person_qid]
    top_nodes = {}

    for obj in objs:
        by_title[obj["wiki_title"]] = obj
        # 上位ノードの people を拾う
        if not obj.get("is_entity") and obj["wiki_title"] in TOP_OCC_QIDS:
            top_nodes[obj["wiki_title"]] = obj
            ppl = []
            for e in obj.get("edges", []):
                # 入力のエッジは P106 固定
                if e.get("property") == "P106" and e.get("target_qid", "").startswith("Q"):
                    ppl.append(e["target_qid"])
            per_parent_people[obj["wiki_title"]].extend(ppl)
    return by_title, per_parent_people, top_nodes

def make_lower_node(tree_id, parent_title, parent_qid, lower_qid, people_qids):
    lower_label = get_label(lower_qid)
    edges = []

    # 下位概念 → 各人物（P106 で付ける：アノテーション用）
    for pq in sorted(set(people_qids)):
        edges.append({
            "property": "P106",
            "target_qid": pq,
            "target_label": get_label(pq)
        })

    # 下位概念 → 親（P279 で付ける：階層の明示）
    edges.append({
        "property": "P279",
        "target_qid": parent_qid,
        "target_label": get_label(parent_qid)
    })

    # 下位概念 → Person も（サンプルに合わせて）
    edges.append({
        "property": "P106",
        "target_qid": "Q215627",
        "target_label": "Person"
    })

    node = {
        "tree_id": tree_id,
        "wiki_title": lower_label,
        "qid": lower_qid,
        "num_edges": len(edges),
        "source_props": ["P106", "P279"],
        "edges": edges,
        "is_entity": False
    }
    return node

def annotate(in_path, out_path, prefer_yago=False, sleep_between=0.0):
    # エンドポイントを切り替えたい場合（デフォルトは Wikidata）
    global WD_SPARQL
    if prefer_yago:
        # 可能な部分は YAGO でも動くが、P106/P39 は Wikidata が確実
        # 実運用では Wikidata を推奨
        WD_SPARQL = YAGO_SPARQL

    # まず入力を読み込み
    objs = list(load_jsonl(in_path))
    by_title, per_parent_people, top_nodes = build_index(objs)

    new_nodes = []
    # 1) ラベル→QID 解決（候補）
    resolved_lower = {
        parent: {
            "positions": labels_to_qids(LOWER_CANDIDATES[parent]["positions"]),
            "occupations": labels_to_qids(LOWER_CANDIDATES[parent]["occupations"]),
        }
        for parent in LOWER_CANDIDATES.keys()
        if parent in per_parent_people
    }

    # 2) 親ごとに人物を精査
    for parent_title, people in per_parent_people.items():
        parent_qid = TOP_OCC_QIDS[parent_title]
        cand_pos = resolved_lower.get(parent_title, {}).get("positions", {})
        cand_occ = resolved_lower.get(parent_title, {}).get("occupations", {})

        # 下位概念 → {人物の集合}
        buckets = defaultdict(set)

        for pq in tqdm(people, desc=f"[{parent_title}] annotating", unit="person"):
            # 政治家：P39（position）から Legislator/Governor/Mayor にマップ
            if parent_title == "Politician":
                matched = infer_politician_positions(pq, ANCHOR_QIDS)
                for lower_q in matched:
                    buckets[lower_q].add(pq)
            # それ以外：P106（occupation）から下位職業を推定
            else:
                matched = infer_lower_by_P106(pq, parent_qid, cand_occ)
                for lower_q in matched:
                    buckets[lower_q].add(pq)

            if sleep_between:
                time.sleep(sleep_between)

        # 3) 下位ノードを作成
        for lower_qid, ppl in sorted(buckets.items(), key=lambda x: (-len(x[1]), x[0])):
            node = make_lower_node(
                tree_id=top_nodes[parent_title]["tree_id"],
                parent_title=parent_title,
                parent_qid=parent_qid,
                lower_qid=lower_qid,
                people_qids=ppl
            )
            new_nodes.append(node)

    # 4) 出力：元データ + 新ノード
    save_jsonl(out_path, objs + new_nodes)

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Add sub-occupation annotations under top-level occupations.")
    ap.add_argument("--in", dest="in_path", required=True, help="input jsonl path")
    ap.add_argument("--out", dest="out_path", required=True, help="output jsonl path")
    ap.add_argument("--prefer-yago", action="store_true", help="try using YAGO 4.5 SPARQL endpoint")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between person queries")
    args = ap.parse_args()
    annotate(args.in_path, args.out_path, prefer_yago=args.prefer_yago, sleep_between=args.sleep)

if __name__ == "__main__":
    main()
