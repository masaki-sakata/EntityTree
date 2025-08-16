#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotate JSONL with sub-occupations via YAGO 4.5 SPARQL — label matching version.

ポイント
- 事前の「ラベル→IRI」解決をやめ、SPARQL内でクラスのラベル文字列を直接マッチ。
- ラベル取得は rdfs:label | skos:prefLabel | schema:name | http版 schema:name を許容。
- 言語タグは en or 無しを許容。大小無視、_/- を空白化してマッチ。
- 人物は owl:sameAs で Wikidata QID → YAGO IRI に解決。
- rdf:type と schema:hasOccupation（http/https両対応）を UNION で探索。
- ルートごとに VALUES で許容サブ（＋同義語）を投入し、祖先クラスのラベルと一致したら採用。
- 実行後、(root, sub) カウントを標準出力。
 uv run python3 annotate_lower_concepts_yago.py --input /home/masaki/hierarchical-repr/EntityTree/input/tree_yago_1000people.jsonl --output /home/masaki/hierarchical-repr/EntityTree/input/tree_yago_1000people_annotated.jsonl


"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set

from SPARQLWrapper import SPARQLWrapper, JSON
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm

# ===== Controlled hierarchy (canonical labels) =====
HIERARCHY = {
    "Politician": ["Legislator", "Governor", "Mayor"],
    "Actor": ["Comedian", "Film actor", "Stage actor", "Television actor", "Voice actor"],
    "Athlete": [
        "Association football player", "Basketball player", "Tennis player", "Golfer",
        "Boxer", "Sprinter", "Baseball player", "Ice hockey player"
    ],
    "Musician": [
        "Singer", "Composer", "Songwriter", "Conductor", "Instrumentalist",
        "Pianist", "Guitarist", "Drummer", "DJ", "Record producer"
    ],
    "Scientist": [
        "Physicist", "Chemist", "Mathematician", "Biologist", "Computer scientist",
        "Astronomer", "Economist", "Psychologist", "Neuroscientist", "Engineer"
    ],
    "Business person": ["Entrepreneur", "Business executive", "Investor", "Marketer", "Financier"],
}

# ===== Light synonym map to absorb YAGO label variants =====
# 例：DJ ↔ Disc jockey, Stage/Theatre actor など
SUB_SYNONYMS: Dict[str, List[str]] = {
    "Association football player": ["association football player", "footballer", "soccer player"],
    "Film actor": ["film actor", "movie actor", "film actress", "movie actress"],
    "Stage actor": ["stage actor", "theatre actor", "theater actor"],
    "Television actor": ["television actor", "tv actor", "television actress", "tv actress"],
    "Voice actor": ["voice actor", "voice actress", "voice-over actor", "voice over actor"],
    "DJ": ["dj", "disc jockey"],
    "Record producer": ["record producer", "music producer"],
    "Instrumentalist": ["instrumentalist", "musical instrumentalist"],
    "Business executive": ["business executive", "corporate executive", "company executive"],
    "Entrepreneur": ["entrepreneur", "businessperson", "business person"],  # “entrepreneur”優先
    "Governor": ["governor", "state governor", "provincial governor"],
    "Legislator": ["legislator", "lawmaker"],
    "Marketer": ["marketer", "marketing professional", "marketing specialist"],
    "Boxer": ["boxer", "pugilist"],
}

# ===== Normalize & alias for root labels from input edges =====
def norm_text(s: str) -> str:
    return " ".join(s.replace("_", " ").replace("-", " ").split()).strip().lower()

ROOT_ALIASES = {
    "business person": "Business person",
    "business_person": "Business person",
}

# ===== Build per-root needle list (canonical -> canonical) =====
def build_needles() -> Dict[str, List[Tuple[str, str]]]:
    """Return {root: [(needle_norm, canonical_label), ...]}"""
    out = {}
    for root, subs in HIERARCHY.items():
        pairs = []
        for can in subs:
            variants = SUB_SYNONYMS.get(can, []) + [can]
            for v in variants:
                pairs.append((norm_text(v), can))
        # 重複排除（needle_norm基準）
        seen = {}
        for nrm, can in pairs:
            if nrm not in seen:
                seen[nrm] = can
        out[root] = sorted(seen.items())  # [(needle_norm, canonical)]
    return out

# ===== SPARQL client with retry =====
class YagoClient:
    def __init__(self, endpoint: str, timeout: int = 60):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        # ★ 追加：長いVALUESでも安定させる
        from SPARQLWrapper import POST
        self.sparql.setMethod(POST)
        # （任意）結果JSONを明示
        self.sparql.addCustomHttpHeader("Accept", "application/sparql-results+json")


    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def query(self, q: str) -> Dict:
        self.sparql.setQuery(q)
        return self.sparql.query().convert()

# ===== Batch query by ROOT with label matching (no IRI resolution needed) =====
BATCH_MATCH_BY_LABEL = """
PREFIX wikidata: <http://www.wikidata.org/entity/>
PREFIX owl:     <http://www.w3.org/2002/07/owl#>
PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema:  <https://schema.org/>
PREFIX schemax: <http://schema.org/>
PREFIX skos:    <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?wd ?canon WHERE {
  VALUES ?wd { %WDS% }
  VALUES (?needle ?canon) {
    %NEEDLES%
  }

  ?y owl:sameAs ?wd .

  {
    ?y a ?t .
    ?t rdfs:subClassOf* ?sub .
  }
  UNION
  {
    ?y schema:hasOccupation|schemax:hasOccupation ?occ .
    ?occ rdfs:subClassOf* ?sub .
  }

  ?sub (rdfs:label|skos:prefLabel|schema:name|schemax:name) ?lab .
  FILTER (lang(?lab) = 'en' || lang(?lab) = '')
  BIND(lcase(REPLACE(REPLACE(STR(?lab), '_',' '), '-',' ')) AS ?norm)
  FILTER (?norm = ?needle)
}
"""

def chunks(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def batch_match_by_label(yago: YagoClient, wd_qids: List[str], needles: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return list of (wdQid, canonical_sub_label)"""
    if not wd_qids or not needles:
        return []
    qids_vals = " ".join(f"wikidata:{qid}" for qid in wd_qids)
    # VALUES ( ?needle ?canon ) { ("legislator" "Legislator") ... }
    needles_vals = "\n    ".join(f"({json.dumps(n)} {json.dumps(c)})" for n, c in needles)

    q = (BATCH_MATCH_BY_LABEL
         .replace("%WDS%", qids_vals)
         .replace("%NEEDLES%", needles_vals))

    res = yago.query(q)
    out = []
    for b in res.get("results", {}).get("bindings", []):
        wd = b["wd"]["value"].rpartition("/")[-1]
        canon = b["canon"]["value"]
        out.append((wd, canon))
    return out

# ===== Main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input JSONL path")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--endpoint", default="https://yago-knowledge.org/sparql/query")
    ap.add_argument("--chunk", type=int, default=120, help="WD-QID batch size")
    args = ap.parse_args()

    yago = YagoClient(args.endpoint)
    needles_by_root = build_needles()

    # 読み込み＆rootごとに人物QIDを集約
    rows = []
    persons_by_root: Dict[str, Set[str]] = defaultdict(set)
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)
            if not obj.get("is_entity", False):
                continue
            qid = obj.get("qid")
            for e in obj.get("edges", []):
                if e.get("property") != "P106":
                    continue
                root_raw = e.get("target_label", "")
                norm = norm_text(root_raw)
                root = ROOT_ALIASES.get(norm) or next((k for k in HIERARCHY if norm_text(k)==norm), None)
                if root:
                    persons_by_root[root].add(qid)

    # ルートごとにバッチ照合（文字列マッチ）
    match_map: Dict[Tuple[str, str], str] = {}  # (root, QID) -> canonical sub label
    for root, qids_set in tqdm(persons_by_root.items(), desc="Querying by root"):
        qids = sorted(qids_set)
        needles = needles_by_root[root]  # [(needle_norm, canonical)]
        for chunk in chunks(qids, args.chunk):
            for qid, canon in batch_match_by_label(yago, chunk, needles):
                match_map[(root, qid)] = canon

    # 出力＋カウント
    counts = defaultdict(int)
    roots_seen = defaultdict(int)
    none_by_root = defaultdict(list)  # root -> List[(qid, person_label)]
    counts_after = defaultdict(int)  # (root, sub) -> n （None除去後の最終集計）
    saved_person_ids = set()         # 保存した人物QIDを一意に集計


    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in rows:
            if not obj.get("is_entity", False):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            qid = obj.get("qid")
            person_label = obj.get("wiki_title", qid)

            # 既存ロジックで ann を作る（rootごとに sub_label を入れている前提）
            ann = []
            for e in obj.get("edges", []):
                if e.get("property") != "P106":
                    continue
                root_raw = e.get("target_label", "")
                norm = norm_text(root_raw)
                root = ROOT_ALIASES.get(norm) or next((k for k in HIERARCHY if norm_text(k)==norm), None)
                if not root:
                    continue

                # 既存のマッチ結果を利用
                sub_label = match_map.get((root, qid))
                ann.append({
                    "root_category_label": root,
                    "sub_category_label": sub_label,  # None の可能性あり
                })

            # ★ ここで None を除去してから保存
            ann_filtered = [a for a in ann if a.get("sub_category_label")]

            obj["annotated_occupations"] = ann_filtered
            obj["person_qid"] = qid
            obj["person_label"] = person_label

            saved_person_ids.add(qid) 

            # ★ 除去後データで最終集計
            for a in ann_filtered:
                key = (a["root_category_label"], a["sub_category_label"])
                counts_after[key] += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


    # 集計出力
    print("\n=== Counts per (root, sub) ===")
    by_root = defaultdict(list)
    for (root, sub), c in counts.items():
        by_root[root].append((sub, c))
    for root in sorted(by_root):
        total = sum(c for _, c in by_root[root])
        print(f"[{root}] total={total}")
        for sub, c in sorted(by_root[root], key=lambda x: (-x[1], x[0])):
            print(f"  - {sub}: {c}")
    # print("\n=== Persons with None sub-category (per root) ===")
    # for root in sorted(none_by_root):
    #     items = sorted(none_by_root[root], key=lambda x: x[1].lower())  # nameでソート
    #     print(f"[{root}] total={len(items)}")
    #     for qid, name in items:
    #         print(f"  - {name} ({qid})")


    print("\n=== Seen roots in input ===")
    for root, c in sorted(roots_seen.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {root}: {c}")
    
    print("\n=== Counts per (root, sub) AFTER filtering None ===")
    by_root = defaultdict(list)
    for (root, sub), c in counts_after.items():
        by_root[root].append((sub, c))

    for root in sorted(by_root):
        total = sum(c for _, c in by_root[root])
        print(f"[{root}] total={total}")
        for sub, c in sorted(by_root[root], key=lambda x: (-x[1], x[0])):
            print(f"  - {sub}: {c}")
    print(f"\n=== Total persons saved ===")
    print(len(saved_person_ids))
    


if __name__ == "__main__":
    main()
