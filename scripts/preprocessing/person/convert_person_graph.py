#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
person.graph → persons.jsonl 変換スクリプト
  - subject predicate object のストリーミング処理
  - Category は YAGO URI(types) に and 元文字列(raw)に
  - その他 predicate は properties に
  - label / predicateLabels も保持
実行:
  $ python3 convert_person_graph.py
"""

import json
import re
from urllib.parse import quote

INPUT_FILE  = "/work03/masaki/data/Person_dataset/processed_dataset/person.graph"      # 入力ファイル名
OUTPUT_FILE = "/work03/masaki/data/Person_dataset/processed_dataset/persons_prcs.jsonl"     # 出力ファイル名

DBPEDIA_RES_PREFIX = "http://dbpedia.org/resource/"
YAGO_RES_PREFIX    = "http://yago-knowledge.org/resource/"

def dbpedia_uri(name: str) -> str:
    return DBPEDIA_RES_PREFIX + quote(name)

def yago_uri(cat: str) -> str:
    # 'Category:Foo' → YAGO URI
    return YAGO_RES_PREFIX + quote(cat.split("Category:",1)[1])

def human_label_from_id(name: str) -> str:
    return name.replace("_", " ")

def predicate_to_label(pred: str) -> str:
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', pred)
    s = s.replace("_", " ").lower().capitalize()
    return s

with open(INPUT_FILE, encoding="utf-8") as fr, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fw:

    current = None
    ent = None

    for line in fr:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        subj, pred, obj = parts

        # 新規 subject なら直前のエントリを書き出し
        if subj != current:
            if ent is not None:
                ent["types"] = sorted(ent["types"])
                ent["categoryObjects"] = sorted(ent["categoryObjects"])
                for k,v in list(ent["properties"].items()):
                    lst = sorted(v)
                    ent["properties"][k] = lst[0] if len(lst)==1 else lst
                fw.write(json.dumps(ent, ensure_ascii=False) + "\n")
            current = subj
            ent = {
                "id":               dbpedia_uri(subj),
                "label":            human_label_from_id(subj),
                "types":            [],
                "categoryObjects":  [],
                "properties":       {},
                "predicateLabels":  {}
            }

        # カテゴリは types に YAGO URI／raw にオリジナル文字列
        if pred == "subject" and obj.startswith("Category:"):
            ent["types"].append(yago_uri(obj))
            ent["categoryObjects"].append(obj)
        else:
            # 型付きリテラルを検出: "値"^^<...#型>
            m = re.match(r'^"(.*)"\^\^<[^>]+#([^>]+)>$', obj)
            if m:
                literal, dtype = m.groups()
                # gYear は整数に
                if dtype == "gYear" and pred.endswith("Year"):
                    val = int(literal)
                # date 型は文字列のまま (または ISO フォーマットに変換)
                else:
                    val = literal
            else:
                # 通常のリソース URI や文字列はそのまま
                val = obj

            ent["properties"].setdefault(pred, set()).add(val)
            if pred not in ent["predicateLabels"]:
                ent["predicateLabels"][pred] = predicate_to_label(pred)

    # 最後のエントリを書き出し
    if ent is not None:
        ent["types"] = sorted(ent["types"])
        ent["categoryObjects"] = sorted(ent["categoryObjects"])
        for k,v in list(ent["properties"].items()):
            lst = sorted(v)
            ent["properties"][k] = lst[0] if len(lst)==1 else lst
        fw.write(json.dumps(ent, ensure_ascii=False) + "\n")

print(f"✅ 生成完了: {OUTPUT_FILE}")
