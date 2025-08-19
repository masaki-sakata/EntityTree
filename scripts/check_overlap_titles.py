# check_overlap_titles.py

"""
uv run python3 check_overlap_titles.py /home/masaki/hierarchical-repr/EntityTree/input/300people/test50.jsonl /home/masaki/hierarchical-repr/EntityTree/input/300people/train250.jsonl --ignore-case
# CSVに保存したい場合
uv run python3 check_overlap_titles.py データA.jsonl データB.jsonl --out overlap_titles.csv
"""
# check_overlap_titles_entities.py
import argparse
import pandas as pd

def load_jsonl(path: str) -> pd.DataFrame:
    """jsonlを1行=1レコードとして読み込む。"""
    df = pd.read_json(path, lines=True)
    for col in ("wiki_title", "is_entity"):
        if col not in df.columns:
            raise ValueError(f"{path} に '{col}' 列が見つかりません。")
    return df

def filter_entities(df: pd.DataFrame) -> pd.DataFrame:
    """is_entity が True の行だけ残す。"""
    # JSONでは true/false は bool のはずだが、保険で型を合わせておく
    return df[df["is_entity"] == True].copy()

def main():
    parser = argparse.ArgumentParser(description="2つのjsonl間で(is_entity=Trueの)wiki_title重複をチェック")
    parser.add_argument("data_a", help="データA.jsonlへのパス")
    parser.add_argument("data_b", help="データB.jsonlへのパス")
    parser.add_argument("--out", help="重複wiki_titleを書き出すCSVパス（任意）")
    parser.add_argument("--ignore-case", action="store_true", help="大文字小文字を無視して比較する")
    args = parser.parse_args()

    df_a = filter_entities(load_jsonl(args.data_a))
    df_b = filter_entities(load_jsonl(args.data_b))

    # 前後空白を除去し、必要なら小文字化してから比較
    col = pd.Series.astype
    s_a = df_a["wiki_title"].dropna().astype(str).str.strip()
    s_b = df_b["wiki_title"].dropna().astype(str).str.strip()
    if args.ignore_case:
        s_a = s_a.str.lower()
        s_b = s_b.str.lower()

    titles_a = set(s_a)
    titles_b = set(s_b)
    overlap = sorted(titles_a & titles_b)

    print(f"[A] is_entity=True の行数: {len(df_a)} / ユニークwiki_title数: {len(titles_a)}")
    print(f"[B] is_entity=True の行数: {len(df_b)} / ユニークwiki_title数: {len(titles_b)}")
    print(f"重複（共通）wiki_title数: {len(overlap)}")

    if overlap:
        print("\n重複しているwiki_title一覧:")
        for t in overlap:
            print(t)
    else:
        print("\n重複はありません。")

    if args.out:
        pd.Series(overlap, name="wiki_title").to_csv(args.out, index=False)
        print(f"\nCSVに書き出しました: {args.out}")

if __name__ == "__main__":
    main()
