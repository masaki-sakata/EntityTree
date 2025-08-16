#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YAGO 4.5 で任意クラスの下位概念（rdfs:subClassOf の *下流*）を列挙して表示するスクリプト
- デフォルトは英語ラベル "businessperson" の下位概念
- 直接の下位 (rdfs:subClassOf) / 推移的な全下位 (rdfs:subClassOf+) を切替可
- ラベル検索 (--label/--lang) か URI 直指定 (--uri)

使い方例:
  # businessperson の直接の下位概念
  uv run python3 yago_subclasses_print.py --label businessperson --lang en --mode direct
  uv run python3 yago_subclasses_print.py --label Worker --lang en --mode direct

  # businessperson の全ての下位概念（推移的）
  uv run python3 yago_subclasses_print.py --label businessperson --lang en --mode all
  uv run python3 yago_subclasses_print.py --label Worker --lang en --mode all

  # URI 直指定で全下位
  uv run python3 yago_subclasses_print.py --uri http://yago-knowledge.org/resource/Businessperson --mode all
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request

ENDPOINT = "https://yago-knowledge.org/sparql/query"

PREFIXES = """\
PREFIX yago: <http://yago-knowledge.org/resource/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

def _escape_sparql_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def sparql_select(query: str, timeout: int = 60) -> dict:
    params = {"query": query}
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def find_class_uris_by_label(label: str, lang: str = "en", limit: int = 50):
    s = _escape_sparql_string(label)
    q = PREFIXES + f"""
SELECT ?cls ?label WHERE {{
  ?cls a rdfs:Class ;
       rdfs:label ?label .
  FILTER(LANG(?label) = "{lang}" && LCASE(STR(?label)) = LCASE("{s}"))
}}
LIMIT {limit}
"""
    res = sparql_select(q)
    out = []
    for b in res.get("results", {}).get("bindings", []):
        cls = b.get("cls", {}).get("value")
        lab = b.get("label", {}).get("value")
        if cls:
            out.append((cls, lab))
    return out

def get_direct_subclasses(cls_uri: str, lang: str = "en"):
    q = PREFIXES + f"""
SELECT ?sub ?subLabel WHERE {{
  ?sub a rdfs:Class ;
       rdfs:subClassOf <{cls_uri}> .
  OPTIONAL {{ ?sub rdfs:label ?subLabel FILTER(LANG(?subLabel) = "{lang}") }}
}}
ORDER BY ?subLabel
"""
    return _bindings_to_pairs(sparql_select(q), "sub", "subLabel")

def get_all_subclasses(cls_uri: str, lang: str = "en"):
    q = PREFIXES + f"""
SELECT DISTINCT ?sub ?subLabel WHERE {{
  ?sub a rdfs:Class ;
       rdfs:subClassOf+ <{cls_uri}> .
  OPTIONAL {{ ?sub rdfs:label ?subLabel FILTER(LANG(?subLabel) = "{lang}") }}
}}
ORDER BY ?subLabel
"""
    return _bindings_to_pairs(sparql_select(q), "sub", "subLabel")

def _bindings_to_pairs(res: dict, iri_key: str, label_key: str):
    pairs = []
    for b in res.get("results", {}).get("bindings", []):
        iri = b.get(iri_key, {}).get("value")
        lab = b.get(label_key, {}).get("value")
        pairs.append((iri, lab))
    return pairs

def print_results(title: str, rows):
    print(f"\n=== {title} ({len(rows)} 件) ===")
    for iri, label in rows:
        if label:
            print(f"- {label}  <{iri}>")
        else:
            print(f"- <{iri}>")

def main():
    ap = argparse.ArgumentParser(description="YAGO 4.5: 下位概念（subclasses）列挙（画面出力）")
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--label", default="businessperson", help="対象クラスのラベル（例: businessperson / 実業家）")
    target.add_argument("--uri", help="対象クラスの URI（指定時はラベル検索をスキップ）")
    ap.add_argument("--lang", default="en", help="ラベル言語（en / ja など）")
    ap.add_argument("--mode", choices=["direct", "all"], default="all",
                    help="direct=直接の下位, all=推移的な全下位")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP タイムアウト秒")
    args = ap.parse_args()

    if args.uri:
        cls_uri = args.uri
        cls_label = None
    else:
        candidates = find_class_uris_by_label(args.label, args.lang)
        if not candidates:
            print(f'ラベル "{args.label}" (lang={args.lang}) に一致するクラスが見つかりませんでした。', file=sys.stderr)
            sys.exit(1)
        if len(candidates) > 1:
            print("複数候補が見つかりました。先頭を使用します。候補一覧:")
            for i, (uri, lab) in enumerate(candidates, 1):
                print(f"  [{i}] {lab}  <{uri}>")
        cls_uri, cls_label = candidates[0]

    print(f'\n対象クラス: {cls_label or "(no label)"} <{cls_uri}>')

    if args.mode == "direct":
        rows = get_direct_subclasses(cls_uri, lang=args.lang)
        print_results("直接の下位概念 (rdfs:subClassOf)", rows)
    else:
        rows = get_all_subclasses(cls_uri, lang=args.lang)
        print_results("推移的な全下位概念 (rdfs:subClassOf+)", rows)

if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTPError: {e.code} {e.reason}\n")
        try:
            body = e.read().decode("utf-8", errors="ignore")
            if body:
                sys.stderr.write(body[:1000] + ("\n...省略...\n" if len(body) > 1000 else ""))
        except Exception:
            pass
        sys.exit(2)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(3)
