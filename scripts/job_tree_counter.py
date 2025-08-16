# job_tree_counter.py
import re
import json
import pandas as pd
from collections import defaultdict

PATH_TOP = {
    "politician": "Worker/Politician",
    "actor": "Worker/Actor",
    "athlete": "Worker/Athlete",
    "musician": "Worker/Musician",
    "scientist": "Worker/Scientist",
    "business_person": "Worker/Business_person",
}

# 下位カテゴリの正規名（キー）はツリーの葉・中間ノード名に合わせる
LEAF_PATHS = {
    # Politician
    "legislator": "Worker/Politician/Legislator",
    "governor": "Worker/Politician/Governor",
    "mayor": "Worker/Politician/Mayor",

    # Actor
    "comedian": "Worker/Actor/Comedian",
    "film_actor": "Worker/Actor/Film_actor",
    "stage_actor": "Worker/Actor/Stage_actor",
    "television_actor": "Worker/Actor/Television_actor",
    "voice_actor": "Worker/Actor/Voice_actor",

    # Athlete
    "association_football_player": "Worker/Athlete/Association_football_player",
    "basketball_player": "Worker/Athlete/Basketball_player",
    "tennis_player": "Worker/Athlete/Tennis_player",
    "golfer": "Worker/Athlete/Golfer",
    "boxer": "Worker/Athlete/Boxer",
    "sprinter": "Worker/Athlete/Sprinter",
    "baseball_player": "Worker/Athlete/Baseball_player",
    "ice_hockey_player": "Worker/Athlete/Ice_hockey_player",

    # Musician
    "singer": "Worker/Musician/Singer",
    "composer": "Worker/Musician/Composer",
    "songwriter": "Worker/Musician/Songwriter",
    "conductor": "Worker/Musician/Conductor",
    "instrumentalist": "Worker/Musician/Instrumentalist",
    "pianist": "Worker/Musician/Instrumentalist/Pianist",
    "guitarist": "Worker/Musician/Instrumentalist/Guitarist",
    "drummer": "Worker/Musician/Instrumentalist/Drummer",
    "dj": "Worker/Musician/DJ",
    "record_producer": "Worker/Musician/Record_producer",

    # Scientist
    "physicist": "Worker/Scientist/Physicist",
    "chemist": "Worker/Scientist/Chemist",
    "mathematician": "Worker/Scientist/Mathematician",
    "biologist": "Worker/Scientist/Biologist",
    "computer_scientist": "Worker/Scientist/Computer_scientist",
    "astronomer": "Worker/Scientist/Astronomer",
    "economist": "Worker/Scientist/Economist",
    "psychologist": "Worker/Scientist/Psychologist",
    "neuroscientist": "Worker/Scientist/Neuroscientist",
    "engineer": "Worker/Scientist/Engineer",

    # Business person
    "entrepreneur": "Worker/Business_person/Entrepreneur",
    "business_executive": "Worker/Business_person/Business_executive",
    "investor": "Worker/Business_person/Investor",
    "marketer": "Worker/Business_person/Marketer",
    "financier": "Worker/Business_person/Financier",
}

# 同義語・表記ゆれ → 正規名（キーは正規化後の文字列）
ALIASES = {
    # top-level
    "business person": "business_person",
    "businessperson": "business_person",
    "business-person": "business_person",

    # actor branch
    "movie actor": "film_actor",
    "film actor": "film_actor",
    "tv actor": "television_actor",
    "television actor": "television_actor",
    "voice actor": "voice_actor",
    "stage actor": "stage_actor",

    # athlete branch
    "soccer player": "association_football_player",
    "footballer": "association_football_player",
    "ice hockey player": "ice_hockey_player",
    "base-ball player": "baseball_player",

    # musician branch
    "disc jockey": "dj",
    "record producer": "record_producer",

    # scientist branch
    "computer scientist": "computer_scientist",
}

# 追加で拾うためのパターン（正規化後の文字列に対して）
PATTERNS = [
    (re.compile(r"\b(movie|film)\s*actor\b"), "film_actor"),
    (re.compile(r"\btv\s*actor\b"), "television_actor"),
    (re.compile(r"\btelevision\s*actor\b"), "television_actor"),
    (re.compile(r"\bvoice\s*actor\b"), "voice_actor"),
    (re.compile(r"\bsoccer\s*player\b|\bfootballer\b"), "association_football_player"),
    (re.compile(r"\bice\s*hockey\s*player\b"), "ice_hockey_player"),
    (re.compile(r"\brecord\s*producer\b"), "record_producer"),
]

# 子→親ロールアップ用：葉から最上位（top）までの親チェーン
PARENTS = {}
for key, path in LEAF_PATHS.items():
    parts = path.split("/")
    # Worker/Top[/sub...]
    chain = []
    for i in range(2, len(parts) + 1):
        chain.append("/".join(parts[:i]))
    PARENTS[key] = chain  # 例: ["Worker/Actor", "Worker/Actor/Film_actor"]

# top-levelノード自体（職業そのものが上位ラベルのとき）
TOP_CANON = set(PATH_TOP.keys())

def canonize(label: str) -> str:
    """
    空白/ハイフン→アンダースコア、小文字化、重複アンダースコア整理。
    """
    s = label.strip().lower()
    s = re.sub(r"[–—−\-]+", "_", s)          # ダッシュ類→_
    s = re.sub(r"[^\w\s/]+", "", s)          # 記号除去
    s = re.sub(r"\s+", "_", s)               # 空白→_
    s = re.sub(r"_+", "_", s)                # 連続_圧縮
    return s

def normalize_to_key(label: str) -> str | None:
    """
    ラベルを正規化して、下位カテゴリ or トップカテゴリの正規キーに変換。
    該当なしなら None。
    """
    c = canonize(label)

    # まずは直接一致（葉）
    if c in LEAF_PATHS:
        return c

    # 同義語置換
    if c in ALIASES:
        ali = ALIASES[c]
        if ali in LEAF_PATHS or ali in TOP_CANON:
            return ali

    # パターンで推定
    for pat, key in PATTERNS:
        if pat.search(c):
            return key

    # トップカテゴリ一致（politician, actor など）
    if c in TOP_CANON:
        return c

    # よくあるスペース/アンダースコアゆれ（例：“Business Person”）
    c2 = c.replace(" ", "_")
    if c2 in LEAF_PATHS or c2 in TOP_CANON:
        return c2

    return None

def classify_to_paths(label: str) -> list[str]:
    """
    ラベルをツリー内のパス（親ロールアップ含む）に変換。
    例：'film actor' → ['Worker/Actor', 'Worker/Actor/Film_actor']
    例：'Actor' → ['Worker/Actor']
    """
    key = normalize_to_key(label)
    if key is None:
        return []

    # トップカテゴリならそのパスのみ
    if key in TOP_CANON:
        return [PATH_TOP[key]]

    # 葉や中間なら親チェーン（トップ含む）を返す
    return PARENTS.get(key, [])

def main():
    # ===== 1) JSONL 読み込み =====
    df = pd.read_json("/home/masaki/hierarchical-repr/EntityTree/input/tree_yago_300people_annotated.jsonl", lines=True)

    # ===== 2) edges を展開し、職業ラベル抽出 =====
    df_ex = df.explode("edges", ignore_index=True)
    df_ex["profession_raw"] = df_ex["edges"].apply(
        lambda x: x.get("target_label") if isinstance(x, dict) else None
    )

    # ===== 3) ツリーパス（ロールアップ含む）へ変換 =====
    df_ex["paths"] = df_ex["profession_raw"].apply(
        lambda s: classify_to_paths(s) if isinstance(s, str) else []
    )

    # 空リストを除外して explode
    df_ex = df_ex[df_ex["paths"].map(bool)].explode("paths", ignore_index=True)

    # ===== 4) 人数カウント（wiki_title単位で重複排除） =====
    # ※ 同一人物が同一カテゴリに複数ラベルでヒットしても1人として数える
    counts = (
        df_ex.groupby("paths")["wiki_title"]
        .nunique()
        .reset_index()
        .rename(columns={"paths": "category_path", "wiki_title": "num_people"})
        .sort_values(["category_path"])
        .reset_index(drop=True)
    )

    # 見やすく上位→下位の順に並べ替え（任意）
    def sort_key(p):
        parts = p.split("/")
        return (len(parts), p)

    counts = counts.sort_values(
        by="category_path",
        key=lambda col: col.map(sort_key)
    ).reset_index(drop=True)



    # 出力
    print(counts.to_string(index=False))

if __name__ == "__main__":
    main()
