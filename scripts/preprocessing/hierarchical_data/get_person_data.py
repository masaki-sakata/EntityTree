# -*- coding: utf-8 -*-
"""
YAGO 4.5 から各カテゴリ k 名（計 k * 6 名）をサンプルし、
「最初に与えた木」と「各カテゴリの人物ノード」を同じ JSONL に保存する。

- YAGO: 候補抽出（rdf:type が該当職能クラス）
- Wikidata: QID 付与（人物ラベル＋生年で極力同定、職業がカテゴリの下位に属することも確認）
- JSONL: あなたの例と同じスキーマで 1 行 = 1 ノード

依存: pip install SPARQLWrapper pandas numpy
"""

import json
import re
import random
from collections import Counter, defaultdict
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, CSV, POSTDIRECTLY
import csv
from io import StringIO
import time

# ====== 基本設定 ======
TREE_ID = 1
TARGET_PER_CATEGORY = 2000
PER_CATEGORY_LIMIT  = 6000 * (TARGET_PER_CATEGORY / 50)
MAX_PER_COUNTRY     = 8 * (TARGET_PER_CATEGORY / 50)

OUT_PATH = f"/home/masaki/hierarchical-repr/EntityTree/input/tree_yago_{TARGET_PER_CATEGORY}people.jsonl"

# YAGO / Wikidata SPARQL エンドポイント
YAGO_ENDPOINT = "https://yago-knowledge.org/sparql/query"
WD_ENDPOINT   = "https://query.wikidata.org/sparql"
UA = "yago-wikidata-tree-sampler/0.3 (+https://yago-knowledge.org)"

random.seed(42); np.random.seed(42)

# ルートと 6 カテゴリ（Wikidata の QIDを固定）
ROOT_NODE = {"label": "Person", "qid": "Q215627"}
CATEGORIES = [
    {"label": "Politician",      "qid": "Q82955"},
    {"label": "Actor",           "qid": "Q33999"},
    {"label": "Athlete",         "qid": "Q2066131"},
    {"label": "Musician",        "qid": "Q639669"},
    {"label": "Scientist",       "qid": "Q901"},
    {"label": "Business Person", "qid": "Q43845"},
]

# 上位カテゴリ → YAGO 側の検索ラベル（これを YAGOクラスに解決して rdf:type で人物取得）
ROLE_LABELS = {
    "Politician": [
        "politician", "legislator", "governor", "mayor",
        "member of parliament", "senator"
    ],
    "Actor": [
        "actor", "film actor", "stage actor", "television actor", "voice actor",
        "comedian"
    ],
    "Athlete": [
        "athlete", "association football player", "basketball player",
        "tennis player", "golfer", "boxer", "sprinter",
        "baseball player", "ice hockey player"
    ],
    "Musician": [
        "musician", "singer", "composer", "songwriter", "conductor",
        "pianist", "guitarist", "drummer", "disc jockey", "dj",
        "record producer", "instrumentalist"
    ],
    "Scientist": [
        "scientist", "physicist", "chemist", "mathematician", "biologist",
        "computer scientist", "astronomer", "economist",
        "psychologist", "neuroscientist", "engineer"
    ],
    "Business Person": [
        "businessperson", "entrepreneur", "business executive",
        "investor", "marketer", "financier"
    ],
}

PREFIXES = """
PREFIX schema: <http://schema.org/>
PREFIX yago:   <http://yago-knowledge.org/resource/>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
"""



def _csv_to_bindings(csv_text: str):
    # CSV -> SPARQL JSON風 [{"var":{"value":...}}, ...] に合わせる
    f = StringIO(csv_text)
    reader = csv.DictReader(f)
    out = []
    for row in reader:
        b = {}
        for k, v in row.items():
            if k is None:
                continue
            key = k.lstrip("?")
            if v is None or v == "":
                continue
            b[key] = {"value": v}
        out.append(b)
    return out

def sparql_literal(text: str, lang: str = "en") -> str:
    """SPARQL/Turtle の文字列リテラル用に最低限のエスケープをして返す。"""
    if text is None:
        text = ""
    s = str(text)
    s = s.replace("\\", "\\\\").replace('"', '\\"')  # バックスラッシュ→\\, ダブルクォート→\"
    # 改行などが混じると面倒なので消しておく（必要なら \\n に変換でもOK）
    s = s.replace("\r", " ").replace("\n", " ")
    return f'"{s}"@{lang}' if lang else f'"{s}"'


def run_sparql(endpoint, query, timeout=60, retries=2):
    last_err = None
    for attempt in range(retries):
        # まず JSON（POST）
        s = SPARQLWrapper(endpoint, agent=UA)
        s.setMethod(POSTDIRECTLY)
        s.setTimeout(timeout)
        s.setReturnFormat(JSON)
        s.setQuery(query)
        try:
            return s.query().convert()["results"]["bindings"]
        except Exception as e_json:
            last_err = e_json
            # フォールバック：CSV（POST）
            try:
                s = SPARQLWrapper(endpoint, agent=UA)
                s.setMethod(POSTDIRECTLY)
                s.setTimeout(timeout)
                s.setReturnFormat(CSV)
                s.setQuery(query)
                res = s.query().convert()
                if isinstance(res, bytes):
                    res = res.decode("utf-8", errors="replace")
                return _csv_to_bindings(res)
            except Exception as e_csv:
                last_err = e_csv
                # ちょっと待って再試行
                time.sleep(1.0)
    # ここまで来たら投げるが、デバッグ用にクエリ頭を表示
    head = "\n".join(query.splitlines()[:20])
    print("[SPARQL ERROR]", type(last_err).__name__, str(last_err)[:200])
    print("---- query head ----\n" + head + "\n--------------------")
    raise last_err



def get_val(b, key):
    return b[key]["value"] if key in b and "value" in b[key] else None

# ---------- YAGO: ラベル→クラス解決 ----------
def build_class_resolver_query(en_labels):
    lits = " ".join('"%s"' % l.lower() for l in sorted(set(en_labels)))
    return PREFIXES + f"""
SELECT DISTINCT ?cls ?clsLabel WHERE {{
  ?cls rdf:type rdfs:Class .
  ?cls (rdfs:label|schema:alternateName) ?lab .
  FILTER(LANGMATCHES(LANG(?lab),'en')) .
  FILTER(LCASE(STR(?lab)) IN ({lits})) .
  OPTIONAL {{ ?cls rdfs:label ?clsLabel . FILTER(LANGMATCHES(LANG(?clsLabel),'en')) }}
}}
"""

# 置き換え：ラベル→クラス解決クエリ（STRICT と RELAXED の2段構え）

def build_class_resolver_query_strict(en_labels):
    """
    英語ラベル/別名に対して大小文字無視の等号一致でクラスURIを解決
    （VALUESで列挙し、等号で比較。INは使わない）
    """
    needles = sorted({l.strip().lower() for l in en_labels if l.strip()})
    values = " ".join('"%s"' % n.replace('"', '\\"') for n in needles)
    return PREFIXES + f"""
SELECT DISTINCT ?cls ?clsLabel WHERE {{
  VALUES ?needle {{ {values} }}
  ?cls rdf:type rdfs:Class .
  ?cls (rdfs:label|schema:alternateName) ?lab .
  FILTER(LANGMATCHES(LANG(?lab),'en')) .
  FILTER(LCASE(STR(?lab)) = LCASE(STR(?needle))) .
  OPTIONAL {{ ?cls rdfs:label ?clsLabel . FILTER(LANGMATCHES(LANG(?clsLabel),'en')) }}
}}
"""

def build_class_resolver_query_relaxed(en_labels):
    """
    STRICTで0件のときのフォールバック：部分一致（CONTAINS）
    """
    needles = sorted({l.strip().lower() for l in en_labels if l.strip()})
    values = " ".join('"%s"' % n.replace('"', '\\"') for n in needles)
    return PREFIXES + f"""
SELECT DISTINCT ?cls ?clsLabel WHERE {{
  VALUES ?needle {{ {values} }}
  ?cls rdf:type rdfs:Class .
  ?cls (rdfs:label|schema:alternateName) ?lab .
  FILTER(LANGMATCHES(LANG(?lab),'en')) .
  FILTER(CONTAINS(LCASE(STR(?lab)), LCASE(STR(?needle)))) .
  OPTIONAL {{ ?cls rdfs:label ?clsLabel . FILTER(LANGMATCHES(LANG(?clsLabel),'en')) }}
}}
"""

def resolve_yago_classes(en_labels):
    # まず厳密一致
    q1 = build_class_resolver_query_strict(en_labels)
    res1 = run_sparql(YAGO_ENDPOINT, q1)
    classes = [get_val(b,"cls") for b in res1 if get_val(b,"cls")]
    if classes:
        return classes
    # ダメなら部分一致で救済
    q2 = build_class_resolver_query_relaxed(en_labels)
    res2 = run_sparql(YAGO_ENDPOINT, q2)
    classes = [get_val(b,"cls") for b in res2 if get_val(b,"cls")]
    return classes

# ---------- YAGO: クラス→人物候補 ----------
def build_people_query(class_uris, limit):
    values = " ".join(f"<{u}>" for u in class_uris)
    return PREFIXES + f"""
SELECT DISTINCT ?person ?personLabel ?cls ?clsLabel ?birthDate ?birthPlaceLabel ?countryLabel WHERE {{
  VALUES ?cls {{ {values} }}
  ?person rdf:type ?cls .

  OPTIONAL {{ ?cls rdfs:label ?clsLabel . FILTER(LANGMATCHES(LANG(?clsLabel),'en')) }}

  OPTIONAL {{ ?person rdfs:label ?pl_en . FILTER(LANGMATCHES(LANG(?pl_en),'en')) }}
  OPTIONAL {{ ?person rdfs:label ?pl_any . FILTER(LANG(?pl_any) != '') }}
  BIND(COALESCE(?pl_en, ?pl_any) AS ?personLabel)

  OPTIONAL {{ ?person schema:birthDate ?birthDate . }}
  OPTIONAL {{
    ?person schema:birthPlace ?bp .
    ?bp rdfs:label ?birthPlaceLabel .
    FILTER(LANGMATCHES(LANG(?birthPlaceLabel),'en'))
  }}
  OPTIONAL {{
    ?person schema:nationality ?country .
    ?country rdfs:label ?countryLabel .
    FILTER(LANGMATCHES(LANG(?countryLabel),'en'))
  }}

  # レスポンス縮小: いずれかのメタ情報が必須
  FILTER(BOUND(?personLabel) && (BOUND(?birthDate) || BOUND(?birthPlaceLabel) || BOUND(?countryLabel)))
}}
LIMIT {int(limit)}
"""


def parse_year(v):
    if not v: return None
    m = re.match(r"^-?\d{1,4}", str(v))
    return int(m.group(0)) if m else None

def decade_of(y): return int(y//10*10) if y is not None else None

def dedupe_by_uri(rows):
    seen=set(); out=[]
    for r in rows:
        if r["person"] not in seen:
            out.append(r); seen.add(r["person"])
    return out

def balanced_pick(rows, k=50, max_per_country=8):
    # 前処理
    for r in rows:
        r["birth_year"] = parse_year(r.get("birthDate"))
        r["decade"] = decade_of(r["birth_year"])
        r["countryLabel"] = r.get("countryLabel") or "Unknown"
        r["clsLabel"] = r.get("clsLabel") or r.get("cls","").split("/")[-1]
    rows = [r for r in rows if r["birth_year"] is not None and r["countryLabel"]]
    rows = dedupe_by_uri(rows)
    if not rows: return []

    random.shuffle(rows)
    dec_cnt = Counter(); cty_cnt = Counter(); sub_cnt = Counter()
    selected=[]
    while len(selected) < k and rows:
        scored=[]
        for r in rows:
            c=r["countryLabel"]; d=r["decade"]; s=r["clsLabel"]
            if cty_cnt[c] >= max_per_country: continue
            score = (1/(1+dec_cnt[d])) * (1/(1+cty_cnt[c])) * (1/(1+sub_cnt[s])) + random.random()*1e-6
            scored.append((score, r))
        if not scored:
            max_per_country += 2
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        selected.append(best)
        cty_cnt[best["countryLabel"]] += 1
        dec_cnt[best["decade"]] += 1
        sub_cnt[best["clsLabel"]] += 1
        rows = [r for r in rows if r["person"] != best["person"]]
    return selected

def yago_candidates_for_category(cat_label):
    labels = ROLE_LABELS[cat_label]
    classes = resolve_yago_classes(labels + [cat_label])
    if not classes:
        classes = resolve_yago_classes([cat_label])
    # 同じURIが重複しがちなので除重
    classes = sorted(set(classes))

    if not classes:
        return []

    # クラスごとに小さく取得（例: 1クラスあたり 800 件まで）してマージ
    PER_CLASS_LIMIT = max(300, min(800, PER_CATEGORY_LIMIT // max(1, len(classes)//2 or 1)))
    all_rows = []
    for cls_uri in tqdm(classes, desc=f"{cat_label}: fetching per class", leave=False):
        q = build_people_query([cls_uri], limit=PER_CLASS_LIMIT)
        res = run_sparql(YAGO_ENDPOINT, q)
        for b in res:
            all_rows.append({
                "person": get_val(b,"person"),
                "personLabel": get_val(b,"personLabel"),
                "cls": get_val(b,"cls") or cls_uri,
                "clsLabel": get_val(b,"clsLabel"),
                "birthDate": get_val(b,"birthDate"),
                "birthPlaceLabel": get_val(b,"birthPlaceLabel"),
                "countryLabel": get_val(b,"countryLabel"),
            })

    # ここからは既存と同じ：バランス取り→50名
    return balanced_pick(all_rows, k=TARGET_PER_CATEGORY, max_per_country=MAX_PER_COUNTRY)

# ---------- Wikidata: QID 付与 ----------
def wd_qid_from_sameas(yago_uri):
    """YAGO側に Wikidata への sameAs/exactMatch が付いていればQIDを拾う（ある場合のみ）。"""
    q = f"""
PREFIX owl:   <http://www.w3.org/2002/07/owl#>
PREFIX schema:<http://schema.org/>
PREFIX skos:  <http://www.w3.org/2004/02/skos/core#>
SELECT ?qid WHERE {{
  VALUES ?s {{ <{yago_uri}> }}
  OPTIONAL {{ ?s owl:sameAs ?x . FILTER(STRSTARTS(STR(?x),"http://www.wikidata.org/entity/")) }}
  OPTIONAL {{ ?s schema:sameAs ?y . FILTER(STRSTARTS(STR(?y),"http://www.wikidata.org/entity/")) }}
  OPTIONAL {{ ?s skos:exactMatch ?z . FILTER(STRSTARTS(STR(?z),"http://www.wikidata.org/entity/")) }}
  BIND(COALESCE(?x,?y,?z) AS ?wd)
  BIND(REPLACE(STR(?wd), ".*/", "") AS ?qid)
}}
LIMIT 1
"""
    res = run_sparql(YAGO_ENDPOINT, q)
    return get_val(res[0], "qid") if res else None

def wd_find_qid_by_label_and_occ(name, birth_year, category_qid):
    """
    ラベル（英語）＋人間＋職業がカテゴリの下位（P106/P279*）でQIDを探す。
    ・VALUES は { ... } で囲む
    ・birth_year は BOUND(?b) を確認してから YEAR(?b) を使う
    ・失敗しても (None, None) を返してスキップ
    """
    lit = sparql_literal(name, "en")

    def _query(use_year: bool) -> str:
        fy = f"FILTER(BOUND(?b) && YEAR(?b) = {int(birth_year)})" if (use_year and birth_year) else ""
        return f"""
SELECT ?item ?itemLabel WHERE {{
  VALUES ?needle {{ {lit} }}
  ?item rdfs:label ?needle .
  ?item wdt:P31 wd:Q5 .
  # 職業がカテゴリの下位
  ?item wdt:P106 ?occ .
  ?occ wdt:P279* wd:{category_qid} .
  OPTIONAL {{ ?item wdt:P569 ?b . }}
  {fy}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 1
"""

    try:
        res = run_sparql(WD_ENDPOINT, _query(True))
        if res:
            qid = get_val(res[0], "item").rsplit("/", 1)[-1]
            lbl = get_val(res[0], "itemLabel")
            return qid, lbl
        # 年指定でダメなら年なし
        res = run_sparql(WD_ENDPOINT, _query(False))
        if res:
            qid = get_val(res[0], "item").rsplit("/", 1)[-1]
            lbl = get_val(res[0], "itemLabel")
            return qid, lbl
        # ゆるいフォールバック：case-insensitive 等号（=）で一致
        q_fallback = f"""
SELECT ?item ?itemLabel WHERE {{
  VALUES ?needle {{ {lit} }}
  ?item rdfs:label ?lab .
  FILTER(LANG(?lab)='en' && LCASE(STR(?lab)) = LCASE(STR(?needle)))
  ?item wdt:P31 wd:Q5 .
  ?item wdt:P106 ?occ .
  ?occ wdt:P279* wd:{category_qid} .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 1
"""
        res = run_sparql(WD_ENDPOINT, q_fallback)
        if res:
            qid = get_val(res[0], "item").rsplit("/", 1)[-1]
            lbl = get_val(res[0], "itemLabel")
            return qid, lbl

    except Exception as e:
        # クエリ不正や一時的エラーは、この人物をスキップして続行
        # tqdm を使っているなら tqdm.write で騒がしすぎないよう1行ログ
        try:
            tqdm.write(f"[WD skip] label={name!r} yr={birth_year} cat=wd:{category_qid} err={type(e).__name__}")
        except Exception:
            print(f"[WD skip] label={name!r} yr={birth_year} cat=wd:{category_qid} err={type(e).__name__}")

    return None, None


def attach_wikidata_qids(samples_by_cat, show_progress=True):
    total_to_try = sum(len(v) for v in samples_by_cat.values())
    pbar = tqdm(total=total_to_try, desc="Resolving Wikidata QIDs") if show_progress else None

    used_qids = set()
    out_by_cat = {}

    for cat in samples_by_cat:
        cat_qid = next(c["qid"] for c in CATEGORIES if c["label"] == cat)
        picked = []
        for r in samples_by_cat[cat]:
            qid = None; wdlabel = None

            # 1) sameAs でQID直取り（例外は飲み込んで続行）
            try:
                qid = wd_qid_from_sameas(r["person"])
            except Exception as e:
                try:
                    tqdm.write(f"[sameAs skip] {r.get('personLabel')!r} err={type(e).__name__}")
                except Exception:
                    pass

            # 2) ダメならラベル＋生年＋職業でWikidata検索（←ここが今回の修正点）
            if not qid:
                qid, wdlabel = wd_find_qid_by_label_and_occ(
                    r.get("personLabel"), parse_year(r.get("birthDate")), cat_qid
                )

            if pbar: pbar.update(1)

            if not qid or qid in used_qids:
                continue

            label = wdlabel or r.get("personLabel")
            if not label:
                continue

            used_qids.add(qid)
            rr = dict(r)
            rr["qid"] = qid
            rr["wdLabel"] = label
            picked.append(rr)

            if len(picked) >= TARGET_PER_CATEGORY:
                break

        out_by_cat[cat] = picked

    if pbar: pbar.close()
    return out_by_cat


# ---------- JSONL 書き出し ----------
def write_jsonl(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def make_root_line():
    edges = []
    for c in CATEGORIES:
        edges.append({"property": "P106", "target_qid": c["qid"], "target_label": c["label"]})
    line = {
        "tree_id": TREE_ID,
        "wiki_title": ROOT_NODE["label"],
        "qid": ROOT_NODE["qid"],
        "num_edges": len(edges),
        "source_props": ["P106"],
        "edges": edges,
        "is_entity": False
    }
    return line

def make_category_line(cat_label, people_edges):
    # Personへの逆リンクも 1 本（あなたの例に合わせる）
    edges = list(people_edges) + [{"property": "P106", "target_qid": ROOT_NODE["qid"], "target_label": ROOT_NODE["label"]}]
    cat_qid = next(c["qid"] for c in CATEGORIES if c["label"] == cat_label)
    line = {
        "tree_id": TREE_ID,
        "wiki_title": cat_label,
        "qid": cat_qid,
        "num_edges": len(edges),
        "source_props": ["P106"],
        "edges": edges,
        "is_entity": False
    }
    return line

def make_person_line(person_label, person_qid, cat_label):
    cat_qid = next(c["qid"] for c in CATEGORIES if c["label"] == cat_label)
    edges = [{"property": "P106", "target_qid": cat_qid, "target_label": cat_label}]
    line = {
        "tree_id": TREE_ID,
        "wiki_title": person_label,
        "qid": person_qid,
        "num_edges": len(edges),
        "source_props": ["P106"],
        "edges": edges,
        "is_entity": True
    }
    return line

# ---------- メイン ----------
def main():
    # 1) 各カテゴリで YAGO から候補
    samples_by_cat = {}
    cats = [c["label"] for c in CATEGORIES]

    for cat in tqdm(cats, desc="YAGO sampling (by category)"):
        rows = yago_candidates_for_category(cat)
        tqdm.write(f"{cat}: candidates picked (pre-WD) = {len(rows)}")
        samples_by_cat[cat] = rows

    # 2) Wikidata QID を付与（グローバル重複なしで50人目標）
    resolved = attach_wikidata_qids(samples_by_cat, show_progress=True)

    # 3) JSONL を構築しながら進捗表示
    total_lines = 1  # root
    for cat in cats:
        total_lines += 1                            # category node
        total_lines += len(resolved.get(cat, []))   # person nodes

    lines = []
    with tqdm(total=total_lines, desc="Writing JSONL") as wbar:
        # ルート
        lines.append(make_root_line()); wbar.update(1)

        # カテゴリ & 人物
        total_people = 0
        for cat in cats:
            ppl = resolved.get(cat, [])
            edges = [{"property": "P106", "target_qid": p["qid"], "target_label": p["wdLabel"]} for p in ppl]
            lines.append(make_category_line(cat, edges)); wbar.update(1)

            for p in ppl:
                lines.append(make_person_line(p["wdLabel"], p["qid"], cat)); wbar.update(1)

            total_people += len(ppl)
            tqdm.write(f"{cat}: {len(ppl)} people (after WD)")

    write_jsonl(lines, OUT_PATH)
    print(f"Total people written: {sum(len(resolved.get(c, [])) for c in cats)}")
    print(f"Saved JSONL: {OUT_PATH}")


if __name__ == "__main__":
    main()
