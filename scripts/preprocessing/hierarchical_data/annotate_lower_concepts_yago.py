#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotate JSONL with sub-occupations via YAGO 4.5 SPARQL — canonical tree & single-sub assignment.

- すべての行の `tree_id` を 1 に固定。
- 下位概念ノードは上位概念ノードと同じスキーマ。
- 下位概念ノードの qid は None、wiki_title はアンダースコア表記のサブ概念名。
- 人物は sub が 1 つも付かない場合は出力しない。
- 人物には sub_category エッジを「ひとつだけ」追加（複数付与しない）。
- トップは Person 固定。ルートは Politician / Actor / Athlete / Musician / Scientist / Business_person。
- ツリー形状は Person → Root → Sub → Person だけになるように**サニタイズ**（NEW）。

使い方（例）
uv run python3 annotate_lower_concepts_yago.py \
  --input /home/masaki/hierarchical-repr/EntityTree/input/taxonomy_person_test50.jsonl \
  --output /home/masaki/hierarchical-repr/EntityTree/input/taxonomy_person_test50_annotated.jsonl
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional

from SPARQLWrapper import SPARQLWrapper, JSON
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm

# =========================================================
# 固定ツリー定義
# =========================================================

PERSON_ROOT = "Person"  # トップ

ROOTS_IN_ORDER: List[str] = [
    "Politician",
    "Actor",
    "Athlete",
    "Musician",
    "Scientist",
    "Business_person",
]

HIERARCHY: Dict[str, List[str]] = {
    "Politician": ["Legislator", "Governor", "Mayor"],
    "Actor": ["Comedian", "Film_actor", "Stage_actor", "Television_actor", "Voice_actor"],
    "Athlete": [
        "Association_football_player", "Basketball_player", "Tennis_player", "Golfer",
        "Boxer", "Sprinter", "Baseball_player", "Ice_hockey_player"
    ],
    "Musician": [
        "Singer", "Composer", "Songwriter", "Conductor", "Instrumentalist", "DJ", "Record_producer"
    ],
    "Scientist": [
        "Physicist", "Chemist", "Mathematician", "Biologist", "Computer_scientist",
        "Astronomer", "Economist", "Psychologist", "Neuroscientist", "Engineer"
    ],
    "Business_person": ["Entrepreneur", "Business_executive", "Investor", "Marketer", "Financier"],
}

ROOT_PRIORITY: Dict[str, int] = {r: i for i, r in enumerate(ROOTS_IN_ORDER)}
SUB_PRIORITY: Dict[Tuple[str, str], int] = {(root, sub): i for root, subs in HIERARCHY.items() for i, sub in enumerate(subs)}

# =========================================================
# 正規化・同義語
# =========================================================

def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.replace("_", " ").replace("-", " ").split()).strip().lower()

TOP_ROOT_SYNONYMS = {norm_text("Person"), norm_text("Worker")}  # Worker を吸収

ROOT_SYNONYMS: Dict[str, List[str]] = {
    "Politician": ["politician", "statesman", "stateswoman", "political figure"],
    "Actor": ["actor", "actress", "performer"],
    "Athlete": ["athlete", "sportsperson", "sportsman", "sportswoman"],
    "Musician": ["musician"],
    "Scientist": ["scientist"],
    "Business_person": ["business person", "businessperson", "business man", "business woman", "businesswoman", "businessman"],
}

SUB_SYNONYMS: Dict[str, List[str]] = {
    "Association_football_player": ["association football player", "footballer", "soccer player"],
    "Basketball_player": ["basketball player"],
    "Tennis_player": ["tennis player"],
    "Golfer": ["golfer"],
    "Boxer": ["boxer", "pugilist"],
    "Sprinter": ["sprinter"],
    "Baseball_player": ["baseball player"],
    "Ice_hockey_player": ["ice hockey player"],

    "Comedian": ["comedian", "stand-up comedian", "standup comedian"],
    "Film_actor": ["film actor", "movie actor", "film actress", "movie actress"],
    "Stage_actor": ["stage actor", "theatre actor", "theater actor"],
    "Television_actor": ["television actor", "tv actor", "television actress", "tv actress"],
    "Voice_actor": ["voice actor", "voice actress", "voice-over actor", "voice over actor"],

    "Singer": ["singer", "vocalist"],
    "Composer": ["composer"],
    "Songwriter": ["songwriter", "lyricist"],
    "Conductor": ["conductor"],
    "Instrumentalist": ["instrumentalist", "musical instrumentalist"],
    "DJ": ["dj", "disc jockey"],
    "Record_producer": ["record producer", "music producer"],

    "Physicist": ["physicist"],
    "Chemist": ["chemist"],
    "Mathematician": ["mathematician"],
    "Biologist": ["biologist"],
    "Computer_scientist": ["computer scientist"],
    "Astronomer": ["astronomer"],
    "Economist": ["economist"],
    "Psychologist": ["psychologist"],
    "Neuroscientist": ["neuroscientist"],
    "Engineer": ["engineer"],

    "Entrepreneur": ["entrepreneur", "business founder", "founder"],
    "Business_executive": ["business executive", "corporate executive", "company executive"],
    "Investor": ["investor", "venture capitalist", "vc"],
    "Marketer": ["marketer", "marketing professional", "marketing specialist"],
    "Financier": ["financier"],
}

# 逆引きインデックス
SUB_SYNONYM_INDEX: Dict[str, Tuple[str, str]] = {}
for root, subs in HIERARCHY.items():
    for sub in subs:
        for syn in SUB_SYNONYMS.get(sub, []) + [sub]:
            SUB_SYNONYM_INDEX[norm_text(syn)] = (root, sub)

ROOT_SYNONYM_INDEX: Dict[str, str] = {}
for root, syns in ROOT_SYNONYMS.items():
    for syn in syns + [root]:
        ROOT_SYNONYM_INDEX[norm_text(syn)] = root

def map_label_to_canonical_root(label: str) -> Optional[str]:
    n = norm_text(label)
    if n in ROOT_SYNONYM_INDEX:
        return ROOT_SYNONYM_INDEX[n]
    if n in SUB_SYNONYM_INDEX:
        return SUB_SYNONYM_INDEX[n][0]
    return None

def is_root_synonym(label: str) -> bool:
    """そのラベルが本スキーマの『ルート』同義語に該当するか（サブ同義語は含めない）"""
    return norm_text(label) in ROOT_SYNONYM_INDEX

def is_sub_synonym(label: str) -> bool:
    """そのラベルが本スキーマの『サブ』同義語に該当するか"""
    return norm_text(label) in SUB_SYNONYM_INDEX

# =========================================================
# YAGO バッチ照合
# =========================================================

def build_needles() -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for root, subs in HIERARCHY.items():
        pairs = []
        for can in subs:
            variants = SUB_SYNONYMS.get(can, []) + [can]
            for v in variants:
                pairs.append((norm_text(v), can))
        seen: Dict[str, str] = {}
        for nrm, can in pairs:
            if nrm not in seen:
                seen[nrm] = can
        out[root] = sorted(seen.items())
    return out

class YagoClient:
    def __init__(self, endpoint: str, timeout: int = 60):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(timeout)
        from SPARQLWrapper import POST
        self.sparql.setMethod(POST)
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
    if not wd_qids or not needles:
        return []
    qids_vals = " ".join(f"wikidata:{qid}" for qid in wd_qids)
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

# =========================================================
# “もっともらしいひとつのサブ”選定
# =========================================================

def choose_best_sub(
    qid: str,
    candidate_pairs: Set[Tuple[str, str]],
    p106_labels_raw: List[str],
) -> Optional[Tuple[str, str]]:
    if not candidate_pairs:
        return None

    p106_norms: List[str] = [norm_text(x) for x in p106_labels_raw if isinstance(x, str)]
    p106_tokens: Set[str] = set()
    for s in p106_norms:
        p106_tokens |= set(s.split())

    def sub_variants(sub: str) -> List[str]:
        return [norm_text(v) for v in (SUB_SYNONYMS.get(sub, []) + [sub])]

    def score(root: str, sub: str) -> Tuple:
        substr_hit = 0
        for v in sub_variants(sub):
            if any(v and (v in s or s in v) for s in p106_norms):
                substr_hit = 1
                break
        sub_token_sets = [set(v.split()) for v in sub_variants(sub)]
        token_overlap = max((len(p106_tokens & ts) for ts in sub_token_sets), default=0)
        root_pri = ROOT_PRIORITY.get(root, 1_000)
        sub_pri = SUB_PRIORITY.get((root, sub), 1_000)
        return (-substr_hit, -token_overlap, root_pri, sub_pri, sub)

    return sorted(candidate_pairs, key=lambda rs: score(rs[0], rs[1]))[0]

# =========================================================
# エッジのサニタイズ（NEW）
# =========================================================

def sanitize_root_node_edges_to_person_only(node: Dict[str, Any], person_qid: Optional[str]) -> None:
    """
    ルートノードのエッジを「P279(root→Person) のみ」に制限。
    既存の P106（root→person）等はすべて削除して、重複のない 1 本に統一。
    """
    edges_in = node.get("edges", [])
    if not isinstance(edges_in, list):
        edges_in = []
    new_edges: List[Dict[str, Any]] = []
    had_p279_to_person = False

    for e in edges_in:
        prop = e.get("property")
        tgt_norm = norm_text(e.get("target_label", ""))

        # 既存の root→(Worker|Person) は置換・統一
        if prop == "P279" and tgt_norm in TOP_ROOT_SYNONYMS:
            if not had_p279_to_person:
                new_edges.append({
                    "property": "P279",
                    "target_qid": person_qid,
                    "target_label": PERSON_ROOT,
                })
                had_p279_to_person = True
            continue

        # その他のエッジはルートから削除（P106 等は全て排除）
        # 何もしない（drop）

    if not had_p279_to_person:
        new_edges.append({
            "property": "P279",
            "target_qid": person_qid,
            "target_label": PERSON_ROOT,
        })

    node["edges"] = new_edges
    node["num_edges"] = len(new_edges)
    node["source_props"] = sorted({e["property"] for e in new_edges})

def prune_person_p106_for_our_taxonomy(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    人物の既存 P106 のうち、**本ツリーのルート/サブ（シノニム含む）**を指すエッジを削除。
    それ以外の P106（例: lawyer, writer 等）は残す。
    """
    pruned: List[Dict[str, Any]] = []
    for e in edges:
        if e.get("property") == "P106":
            label = e.get("target_label", "")
            # ルート同義語 or サブ同義語を指す P106 は落とす
            if is_root_synonym(label) or is_sub_synonym(label):
                continue
        pruned.append(e)
    return pruned

def build_root_edges_topdown(root_label: str, subs: Set[str]) -> List[Dict[str, Any]]:
    """Root ノードに Sub への P279（親→子）を付与"""
    edges = []
    for sub in sorted(subs):
        edges.append({
            "property": "P279",
            "target_qid": None,           # サブは qid=None の設計
            "target_label": sub,
        })
    return edges



# =========================================================
# メイン
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input JSONL path")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--endpoint", default="https://yago-knowledge.org/sparql/query")
    ap.add_argument("--chunk", type=int, default=120, help="WD-QID batch size")
    args = ap.parse_args()

    yago = YagoClient(args.endpoint)
    needles_by_root = build_needles()

    rows: List[Dict[str, Any]] = []
    persons_by_root: Dict[str, Set[str]] = defaultdict(set)
    person_label_by_qid: Dict[str, str] = {}
    p106_labels_by_qid: Dict[str, List[str]] = defaultdict(list)
    input_non_entity_nodes: List[Dict[str, Any]] = []

    root_meta: Dict[str, Dict[str, Any]] = {r: {"qid": None, "node": None} for r in ROOTS_IN_ORDER}

    existing_person_node: Optional[Dict[str, Any]] = None
    existing_worker_node: Optional[Dict[str, Any]] = None

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)

            if not obj.get("is_entity", False):
                wtitle = obj.get("wiki_title", "")
                if norm_text(wtitle) in TOP_ROOT_SYNONYMS:
                    if norm_text(wtitle) == norm_text("person"):
                        existing_person_node = obj
                    else:
                        existing_worker_node = obj
                    continue

                can_root = map_label_to_canonical_root(wtitle)
                if can_root in ROOTS_IN_ORDER:
                    root_meta[can_root]["qid"] = obj.get("qid")
                    root_meta[can_root]["node"] = obj
                else:
                    input_non_entity_nodes.append(obj)
                continue

            # entity (person)
            qid = obj.get("qid")
            person_label_by_qid[qid] = obj.get("wiki_title", qid)
            for e in obj.get("edges", []):
                if e.get("property") != "P106":
                    continue
                raw = e.get("target_label", "")
                if isinstance(raw, str) and raw:
                    p106_labels_by_qid[qid].append(raw)
                can_root = map_label_to_canonical_root(raw)
                if can_root:
                    persons_by_root[can_root].add(qid)

    # YAGO 照合
    match_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for root, qids_set in tqdm(persons_by_root.items(), desc="Querying by root"):
        qids = sorted(qids_set)
        needles = needles_by_root.get(root, [])
        for chunk in chunks(qids, args.chunk):
            for qid, canon_sub in batch_match_by_label(yago, chunk, needles):
                match_map[(root, qid)].add(canon_sub)

    # “ひとつだけ”選定
    chosen_sub_by_person: Dict[str, Tuple[str, str]] = {}
    for qid in person_label_by_qid.keys():
        candidate_pairs: Set[Tuple[str, str]] = set()
        for root in ROOTS_IN_ORDER:
            subs = match_map.get((root, qid), set())
            for sub in subs:
                candidate_pairs.add((root, sub))
        if not candidate_pairs:
            continue
        best = choose_best_sub(qid, candidate_pairs, p106_labels_by_qid.get(qid, []))
        if best:
            chosen_sub_by_person[qid] = best

    # 人物ノードを構築（P106 の整理 + sub_category 1 本付与）
    output_persons: List[Dict[str, Any]] = []
    sub_to_persons: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    for obj in rows:
        if not obj.get("is_entity", False):
            continue
        qid = obj.get("qid")
        person_label = obj.get("wiki_title", qid)

        if qid not in chosen_sub_by_person:
            continue

        root, sub = chosen_sub_by_person[qid]

        # 既存 P106 を整理（本ツリーの root/sub を指すものは削除）
        existing_edges = list(obj.get("edges", []))
        existing_edges = prune_person_p106_for_our_taxonomy(existing_edges)  # NEW

        # sub_category を 1 本だけ追加
        existing_edges.append({
            "property": "sub_category",
            "target_label": sub,
            "root_category_label": root,
        })

        sub_to_persons[(root, sub)].append((qid, person_label))

        obj.pop("annotated_occupations", None)
        obj["person_qid"] = qid
        obj["person_label"] = person_label
        obj["edges"] = existing_edges
        obj["tree_id"] = 1
        obj["num_edges"] = len(obj["edges"])
        obj["source_props"] = sorted({e["property"] for e in obj["edges"]})

        output_persons.append(obj)

    # 下位概念ノード（sub -> root, sub -> persons）
    output_sub_nodes: List[Dict[str, Any]] = []
    root_to_subs: Dict[str, Set[str]] = defaultdict(set)  # NEW: Root -> {Sub,...}

    for (root_label, sub_label), ppl in sorted(sub_to_persons.items()):
        root_to_subs[root_label].add(sub_label)  # NEW: 後で root->sub を張るために集計

        # サブノードは人物への P106 のみを持つ
        edges = []
        for qid, name in sorted(ppl, key=lambda x: x[1].lower()):
            edges.append({
                "property": "P106",
                "target_qid": qid,
                "target_label": name,
            })

        node = {
            "tree_id": 1,
            "wiki_title": sub_label,
            "qid": None,
            "num_edges": len(edges),
            "source_props": sorted({e["property"] for e in edges}),
            "edges": edges,
            "is_entity": False,
        }
        output_sub_nodes.append(node)

    # トップ（Person）ノードを用意
    if existing_person_node is not None:
        person_node = existing_person_node
        person_node["wiki_title"] = PERSON_ROOT
    elif existing_worker_node is not None:
        person_node = existing_worker_node
        person_node["wiki_title"] = PERSON_ROOT
    else:
        person_node = {
            "tree_id": 1,
            "wiki_title": PERSON_ROOT,
            "qid": None,
            "num_edges": 0,
            "source_props": [],
            "edges": [],
            "is_entity": False,
        }

    person_qid = person_node.get("qid")
    # --- ルートノードの取り込み（正規化のみ。エッジは後で root->sub に再構成）  
    normalized_root_nodes: Dict[str, Dict[str, Any]] = {}
    passthrough_others: List[Dict[str, Any]] = []

    for node in input_non_entity_nodes:
        wtitle = node.get("wiki_title", "")
        if norm_text(wtitle) in TOP_ROOT_SYNONYMS:
            continue  # 念のため（Person/Worker はここに来ない）

        can_root = map_label_to_canonical_root(wtitle)
        if can_root in ROOTS_IN_ORDER:
            node["wiki_title"] = can_root
            node["tree_id"] = 1
            # エッジは一旦空に（この後で root->sub を構成する）
            node["edges"] = []
            node["num_edges"] = 0
            node["source_props"] = []
            normalized_root_nodes[can_root] = node
            root_meta[can_root]["qid"] = node.get("qid")
            root_meta[can_root]["node"] = node
        else:
            node["tree_id"] = 1
            passthrough_others.append(node)

    # ルート不足分を補完生成（この時点ではまだエッジは付けない）
    used_roots = sorted(root_to_subs.keys(), key=lambda r: ROOT_PRIORITY.get(r, 999))
    for root in used_roots:
        if root not in normalized_root_nodes:
            normalized_root_nodes[root] = {
                "tree_id": 1,
                "wiki_title": root,
                "qid": root_meta.get(root, {}).get("qid"),
                "edges": [],
                "num_edges": 0,
                "source_props": [],
                "is_entity": False,
            }

    # ここで root のエッジを「P279(root -> sub)」に再構成（親→子）
    for root, subs in root_to_subs.items():
        node = normalized_root_nodes[root]
        edges = [{
            "property": "P279",
            "target_qid": None,          # sub の qid は None の仕様
            "target_label": sub,
        } for sub in sorted(subs)]
        node["edges"] = edges
        node["num_edges"] = len(edges)
        node["source_props"] = (["P279"] if edges else [])

    # Person ノードの整合
    person_node["wiki_title"] = PERSON_ROOT
    person_edges = [{
        "property": "P279",
        "target_qid": root_meta.get(root, {}).get("qid"),
        "target_label": root,
    } for root in sorted(root_to_subs.keys(), key=lambda r: ROOT_PRIORITY.get(r, 999))]
    person_node["edges"] = person_edges
    person_node["num_edges"] = len(person_edges)
    person_node["source_props"] = (["P279"] if person_edges else [])
    person_node["tree_id"] = 1

    # 書き出し順：Person → ルート群 → 下位概念群 → 人物 → その他
    with open(args.output, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(person_node, ensure_ascii=False) + "\n")
        # ルート群（実際に Sub を持つものだけ）
        for root in sorted(root_to_subs.keys(), key=lambda r: ROOT_PRIORITY.get(r, 999)):
            node = normalized_root_nodes[root]
            node["tree_id"] = 1
            fout.write(json.dumps(node, ensure_ascii=False) + "\n")
        for o in output_sub_nodes:
            o["tree_id"] = 1
            fout.write(json.dumps(o, ensure_ascii=False) + "\n")
        for o in output_persons:
            o["tree_id"] = 1
            fout.write(json.dumps(o, ensure_ascii=False) + "\n")
        for o in passthrough_others:
            o["tree_id"] = 1
            fout.write(json.dumps(o, ensure_ascii=False) + "\n")

    # 集計
    print("\n=== Counts per (root, sub) (chosen once per person) ===")
    by_root = defaultdict(list)
    for (root, sub), ppl in sub_to_persons.items():
        by_root[root].append((sub, len(ppl)))
    for root in ROOTS_IN_ORDER:
        if root not in by_root:
            continue
        total = sum(c for _, c in by_root[root])
        print(f"[{root}] total={total}")
        for sub, c in sorted(by_root[root], key=lambda x: (-x[1], x[0])):
            print(f"  - {sub}: {c}")
    print(f"\n=== Top root ===")
    print(PERSON_ROOT)
    print(f"\n=== Total persons saved ===")
    print(len(output_persons))

if __name__ == "__main__":
    main()
