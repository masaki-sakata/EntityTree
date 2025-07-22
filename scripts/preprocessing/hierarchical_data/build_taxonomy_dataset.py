import json
import argparse
import gzip
from datasets import load_dataset
from collections import defaultdict, Counter
import re
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import sys
from tqdm import tqdm

# rdflibは一切使用しない（手動パーサーのみ）

class TaxonomyDatasetCreator:
    def __init__(self, min_entities_per_category: int = 50, max_category_depth: int = 6, min_tree_depth: int = 3):
        self.popqa_entities = {}
        self.entity_types = defaultdict(set)  # qid -> set of types
        self.type_entities = defaultdict(set)  # type -> set of qids
        self.type_hierarchy = defaultdict(set)  # parent_type -> set of child_types
        self.selected_categories = set()
        self.taxonomy_tree = defaultdict(set)
        self.entity_info = {}
        self.yago2qid = {}
        
        # パラメータ
        self.min_entities_per_category = min_entities_per_category
        self.max_category_depth = max_category_depth
        self.min_tree_depth = min_tree_depth
        
        # YAGOエンティティとタイプのパターン
        self.yago_entity_pattern = re.compile(r'<http://yago-knowledge\.org/resource/([^>]+)>')
        self.wikidata_entity_pattern = re.compile(r'<http://www\.wikidata\.org/entity/(Q\d+)>')
        self.schema_org_pattern = re.compile(r'<http://schema\.org/([^>]+)>')
        self.rdf_type_pattern = re.compile(r'<http://www\.w3\.org/1999/02/22-rdf-syntax-ns#type>')
        self.rdfs_subclass_pattern = re.compile(r'<http://www\.w3\.org/2000/01/rdf-schema#subClassOf>')
        
    def load_popqa_entities(self, dataset_name: str = "masaki-sakata/popqa-unique-entities"):
        """PopQAデータセットから固有名詞エンティティを読み込み"""
        print(f"Loading PopQA entities from {dataset_name}...")
        try:
            dataset = load_dataset(dataset_name)
            for item in dataset['train']:
                qid = item['qid']
                pop = item.get('pop', 0)
                self.popqa_entities[qid] = {
                    'pop': pop,
                    'label': item.get('label', ''),
                    'description': item.get('description', '')
                }
            print(f"Loaded {len(self.popqa_entities)} PopQA entities")
        except Exception as e:
            print(f"Error loading PopQA dataset: {e}")
            sys.exit(1)
    
    def parse_yago_facts(self, facts_file: str):
        """
        yago-facts.ttl を rdflib ではなく自前パーサで処理する
        ── rdflib は巨大ファイルで遅い & BCE 日付で落ちるため
        """
        print(f"Parsing YAGO facts file (manual fast mode): {facts_file}")
        self.parse_yago_facts_manual(facts_file)

    # def parse_yago_facts(self, facts_file: str):
    #     """yago-facts.ttlからエンティティのタイプ情報を抽出（Turtle形式対応）"""
    #     print(f"Parsing YAGO facts file: {facts_file}")
        
    #     try:
    #         import rdflib
    #         # ---- BCE 日付で落ちないように変換を無効化 -----------------
    #         from rdflib.namespace import XSD
    #         from rdflib.term import _toPythonMapping
    #         _toPythonMapping[XSD.dateTime] = lambda v: v
    #         # ------------------------------------------------------------
    #     except ImportError:
    #         print("Error: rdflib is required for parsing Turtle files. Install with: pip install rdflib")
    #         sys.exit(1)
        
    #     print("Loading RDF graph (this may take several minutes for large files)...")
    #     g = rdflib.Graph()
        
    #     try:
    #         g.parse(facts_file, format='turtle')
    #         print(f"Successfully loaded RDF graph with {len(g)} triples")
    #     except Exception as e:
    #         print(f"Error loading RDF graph: {e}")
    #         print("Trying alternative parsing methods...")
    #         self.parse_yago_facts_manual(facts_file)
    #         return
        
    #     found_type_relations = 0
    #     processed_triples = 0
        
    #     # rdf:type関係を検索
    #     RDF_TYPE = rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        
    #     for subject, predicate, obj in g:
    #         processed_triples += 1
    #         if processed_triples % 1000000 == 0:
    #             print(f"Processed {processed_triples} triples, found {found_type_relations} type relations")
            
    #         if predicate == RDF_TYPE:
    #             # Wikidataエンティティかチェック
    #             subject_str = str(subject)
    #             entity_match = self.wikidata_entity_pattern.match(f"<{subject_str}>")
                
    #             if entity_match:
    #                 qid = entity_match.group(1)
    #                 if qid in self.popqa_entities:
    #                     # タイプを抽出
    #                     type_value = self.extract_type_from_object(f"<{str(obj)}>")
    #                     if type_value:
    #                         self.entity_types[qid].add(type_value)
    #                         self.type_entities[type_value].add(qid)
    #                         found_type_relations += 1
        
    #     print(f"Finished parsing facts. Found {found_type_relations} type relations for {len(self.entity_types)} entities")
    
    def parse_yago_facts_manual(self, facts_file: str):
        """手動でTurtle形式を解析（rdflibが使えない場合のフォールバック）"""
        print("Using manual Turtle parsing...")
        
        open_func = gzip.open if facts_file.endswith('.gz') else open
        processed_lines = 0
        found_type_relations = 0
        
        # プレフィックス定義を保存
        prefixes = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'owl': 'http://www.w3.org/2002/07/owl#',
            'schema': 'http://schema.org/',
            'yago': 'http://yago-knowledge.org/resource/',
            'wd': 'http://www.wikidata.org/entity/'
        }
        
        try:
            with open_func(facts_file, 'rt', encoding='utf-8') as f:
                current_subject = None
                buffer = ""
                
                for line in tqdm(f):
                    processed_lines += 1
                    if processed_lines % 10000000 == 0:
                        print(f"Processed {processed_lines} lines, found {found_type_relations} type relations")
                    
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # プレフィックス定義を解析
                    if line.startswith('@prefix'):
                        parts = line.split()
                        if len(parts) >= 3:
                            prefix_name = parts[1].rstrip(':')
                            prefix_uri = parts[2].strip('<> .')
                            prefixes[prefix_name] = prefix_uri
                        continue
                    
                    # @base定義をスキップ
                    if line.startswith('@base'):
                        continue
                    
                    # 複数行のトリプルを処理
                    buffer += " " + line
                    
                    # トリプルが完了したかチェック（. で終わる）
                    if buffer.strip().endswith('.'):
                        found_type_relations = self.process_turtle_statement(buffer.strip(), prefixes, found_type_relations)
                        buffer = ""
                    elif buffer.strip().endswith(';'):
                        # セミコロンで続く場合は主語を保持
                        continue
                    elif buffer.strip().endswith(','):
                        # カンマで続く場合は主語と述語を保持
                        continue
        
        except Exception as e:
            print(f"Error parsing YAGO facts manually: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Manual parsing finished. Found {found_type_relations} type relations")
    

    
    # def process_turtle_statement(self, statement: str, prefixes: Dict[str, str], current_count: int) -> int:
    #     """Turtleステートメントを処理してタイプ関係を抽出"""
    #     statement = statement.rstrip(' .')
    #     found_relations = current_count
        
    #     try:
    #         # 簡単なケース: subject predicate object .
    #         if ' a ' in statement or 'rdf:type' in statement:
    #             parts = statement.split()
    #             if len(parts) >= 3:
    #                 subject = self.expand_prefixed_uri(parts[0], prefixes)
    #                 predicate = parts[1]
    #                 obj_parts = parts[2:]
                    
    #                 # オブジェクトが複数の単語にまたがる場合を処理
    #                 obj = ' '.join(obj_parts).rstrip(' .,;')
    #                 obj = self.expand_prefixed_uri(obj, prefixes)
                    
    #                 # rdf:typeまたは'a'の場合
    #                 if predicate in ['a', 'rdf:type']:
    #                     # Wikidataエンティティかチェック
    #                     if 'wikidata.org/entity/' in subject:
    #                         qid_match = re.search(r'/entity/(Q\d+)', subject)
    #                         if qid_match:
    #                             qid = qid_match.group(1)
    #                             if qid in self.popqa_entities:
    #                                 type_value = self.extract_type_from_object_uri(obj)
    #                                 if type_value:
    #                                     self.entity_types[qid].add(type_value)
    #                                     self.type_entities[type_value].add(qid)
    #                                     found_relations += 1
    #     except Exception as e:
    #         # 個別の行でエラーが発生してもスキップして続行
    #         pass
        
    #     return found_relations
    
    def process_turtle_statement(self, statement, prefixes, current_count):
        statement = statement.rstrip(' .')
        found = current_count

        try:
            parts = statement.split(maxsplit=2)
            if len(parts) < 3:
                return found

            subj_raw, pred_raw, obj_raw = parts
            subj = self.expand_prefixed_uri(subj_raw.strip('<>'), prefixes)
            pred = self.expand_prefixed_uri(pred_raw.strip('<>'), prefixes)
            obj  = self.expand_prefixed_uri(obj_raw.strip(' .;<>'), prefixes)

            # rdf:type のみ対象
            if pred not in (
                "a",
                "rdf:type",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            ):
                return found

            # --- 主語を QID に変換 ---------------------------------
            qid = None
            if "wikidata.org/entity/" in subj:
                m = re.search(r"/entity/(Q\d+)", subj)
                if m:
                    qid = m.group(1)
            elif "yago-knowledge.org/resource/" in subj:
                yago_ent = subj.replace("http://yago-knowledge.org/resource/", "")
                qid = self.yago2qid.get(yago_ent)       # ← ここがポイント
            if qid is None or qid not in self.popqa_entities:
                return found
            # -------------------------------------------------------

            # オブジェクト（タイプ）を取得
            type_val = self.extract_type_from_object_uri(obj)
            if not type_val:
                return found

            self.entity_types[qid].add(type_val)
            self.type_entities[type_val].add(qid)
            found += 1

        except Exception:
            pass

        return found


    def extract_type_from_object_uri(self, obj_uri: str) -> Optional[str]:
        """完全なURIからタイプ名を抽出"""
        # YAGO resource
        if obj_uri.startswith('http://yago-knowledge.org/resource/'):
            return obj_uri.replace('http://yago-knowledge.org/resource/', '')
        
        # Schema.org type
        if obj_uri.startswith('http://schema.org/'):
            return f"schema_{obj_uri.replace('http://schema.org/', '')}"
        
        return None
    
    def expand_prefixed_uri(self, uri: str, prefixes: Dict[str, str]) -> str:
        """プレフィックス付きURIを展開"""
        # 既に完全URI（角括弧付き）
        if uri.startswith('<') and uri.endswith('>'):
            return uri[1:-1]
        
        # プレフィックス付きURI
        if ':' in uri:
            prefix, local = uri.split(':', 1)
            if prefix in prefixes:
                return prefixes[prefix] + local
        
        # そのままの場合
        return uri
    
    def expand_prefixed_uri(self, uri: str, prefixes: Dict[str, str]) -> str:
        """プレフィックス付きURIを展開"""
        if uri.startswith('<') and uri.endswith('>'):
            return uri[1:-1]  # 既に完全URI
        
        if ':' in uri:
            prefix, local = uri.split(':', 1)
            if prefix in prefixes:
                return prefixes[prefix] + local
        
        return uri
    

    def parse_yago_beyond_wikipedia(self, beyond_file: str):
        """
        yago-beyond-wikipedia.ttl を rdflib ではなく自前パーサで処理する
        ── rdflib は巨大ファイルで遅い & BCE 日付で落ちるため
        """
        print(f"Parsing YAGO beyond Wikipedia file: {beyond_file}")
        self.parse_yago_facts_manual(beyond_file)

    # def parse_yago_beyond_wikipedia(self, beyond_file: str):
    #     """yago-beyond-wikipedia.ttlからエンティティのタイプ情報を抽出（Turtle形式対応）"""
    #     print(f"Parsing YAGO beyond Wikipedia file: {beyond_file}")
        
    #     try:
    #         import rdflib
    #         # ---- BCE 日付で落ちないように変換を無効化 -----------------
    #         from rdflib.namespace import XSD
    #         from rdflib.term import _toPythonMapping
    #         _toPythonMapping[XSD.dateTime] = lambda v: v
    #         # ------------------------------------------------------------
    #     except ImportError:
    #         print("Error: rdflib is required for parsing Turtle files. Install with: pip install rdflib")
    #         sys.exit(1)
        
    #     print("Loading beyond Wikipedia RDF graph (this may take several minutes)...")
    #     g = rdflib.Graph()
        
    #     try:
    #         g.parse(beyond_file, format='turtle')
    #         print(f"Successfully loaded beyond Wikipedia RDF graph with {len(g)} triples")
    #     except Exception as e:
    #         print(f"Error loading beyond Wikipedia RDF graph: {e}")
    #         print("Trying alternative parsing methods...")
    #         self.parse_yago_facts_manual(beyond_file)
    #         return
        
    #     found_relations = 0
    #     processed_triples = 0
        
    #     # rdf:type関係を検索
    #     RDF_TYPE = rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        
    #     for subject, predicate, obj in g:
    #         processed_triples += 1
    #         if processed_triples % 500000 == 0:
    #             print(f"Processed {processed_triples} triples, found {found_relations} relations")
            
    #         if predicate == RDF_TYPE:
    #             # Wikidataエンティティかチェック
    #             subject_str = str(subject)
    #             entity_match = self.wikidata_entity_pattern.match(f"<{subject_str}>")
                
    #             if entity_match:
    #                 qid = entity_match.group(1)
    #                 if qid in self.popqa_entities:
    #                     type_value = self.extract_type_from_object(f"<{str(obj)}>")
    #                     if type_value:
    #                         self.entity_types[qid].add(type_value)
    #                         self.type_entities[type_value].add(qid)
    #                         found_relations += 1
        
    #     print(f"Finished parsing beyond Wikipedia. Found {found_relations} additional relations")
    
    def parse_yago_taxonomy(self, taxonomy_file: str):
        """yago-taxonomy.ttlから分類階層情報を抽出（Turtle形式対応）"""
        print(f"Parsing YAGO taxonomy: {taxonomy_file}")
        
        try:
            import rdflib
            # ---- BCE 日付で落ちないように変換を無効化 -----------------
            from rdflib.namespace import XSD
            from rdflib.term import _toPythonMapping
            _toPythonMapping[XSD.dateTime] = lambda v: v
            # ------------------------------------------------------------
        except ImportError:
            print("Error: rdflib is required for parsing Turtle files. Install with: pip install rdflib")
            sys.exit(1)
        
        print("Loading taxonomy RDF graph...")
        g = rdflib.Graph()
        
        try:
            g.parse(taxonomy_file, format='turtle')
            print(f"Successfully loaded taxonomy RDF graph with {len(g)} triples")
        except Exception as e:
            print(f"Error loading taxonomy RDF graph: {e}")
            print("Trying alternative parsing methods...")
            self.parse_yago_taxonomy_manual(taxonomy_file)
            return
        
        found_hierarchies = 0
        processed_triples = 0
        
        # rdfs:subClassOf関係を検索
        RDFS_SUBCLASS_OF = rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
        
        for subject, predicate, obj in g:
            processed_triples += 1
            if processed_triples % 100000 == 0:
                print(f"Processed {processed_triples} triples, found {found_hierarchies} hierarchy relations")
            
            if predicate == RDFS_SUBCLASS_OF:
                child = self.extract_type_from_object(f"<{str(subject)}>")
                parent = self.extract_type_from_object(f"<{str(obj)}>")
                
                if child and parent:
                    self.type_hierarchy[parent].add(child)
                    found_hierarchies += 1
        
        print(f"Finished parsing taxonomy. Found {found_hierarchies} hierarchy relations")
    
    def parse_yago_taxonomy_manual(self, taxonomy_file: str):
        """手動でTurtle形式のtaxonomyを解析（rdflibフォールバック）"""
        print("Using manual taxonomy parsing...")
        
        open_func = gzip.open if taxonomy_file.endswith('.gz') else open
        processed_lines = 0
        found_hierarchies = 0
        prefixes = {}
        
        try:
            with open_func(taxonomy_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    processed_lines += 1
                    if processed_lines % 100000 == 0:
                        print(f"Processed {processed_lines} lines, found {found_hierarchies} hierarchy relations")
                    
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # プレフィックス定義を解析
                    if line.startswith('@prefix'):
                        parts = line.split()
                        if len(parts) >= 3:
                            prefix_name = parts[1].rstrip(':')
                            prefix_uri = parts[2].strip('<> .')
                            prefixes[prefix_name] = prefix_uri
                        continue
                    
                    # rdfs:subClassOf関係を探す
                    if 'rdfs:subClassOf' in line:
                        if line.endswith(' .'):
                            line = line[:-2]
                        
                        parts = line.split()
                        if len(parts) >= 3:
                            subject = self.expand_prefixed_uri(parts[0], prefixes)
                            predicate = parts[1]
                            obj = self.expand_prefixed_uri(parts[2], prefixes)
                            
                            if predicate == 'rdfs:subClassOf':
                                child = self.extract_type_from_object(f"<{subject}>")
                                parent = self.extract_type_from_object(f"<{obj}>")
                                
                                if child and parent:
                                    self.type_hierarchy[parent].add(child)
                                    found_hierarchies += 1
        
        except Exception as e:
            print(f"Error parsing taxonomy manually: {e}")
        
    def extract_type_from_object(self, obj: str) -> Optional[str]:
        """オブジェクトからタイプ名を抽出（Turtle形式対応）"""
        # 角括弧を除去
        if obj.startswith('<') and obj.endswith('>'):
            obj = obj[1:-1]
        
        # YAGO resource
        if obj.startswith('http://yago-knowledge.org/resource/'):
            return obj.replace('http://yago-knowledge.org/resource/', '')
        
        # Schema.org type
        if obj.startswith('http://schema.org/'):
            return f"schema_{obj.replace('http://schema.org/', '')}"
        
        # 正規表現での古い方法もフォールバック
        yago_match = self.yago_entity_pattern.match(f"<{obj}>")
        if yago_match:
            return yago_match.group(1)
        
        schema_match = self.schema_org_pattern.match(f"<{obj}>")
        if schema_match:
            return f"schema_{schema_match.group(1)}"
        
        return None


    def parse_wikidata_mappings(self, meta_file: str) -> Dict[str, str]:
        print(f"Extracting Wikidata mappings from: {meta_file}")
        open_func = gzip.open if meta_file.endswith('.gz') else open

        y2q = {}
        processed = found = 0
        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                processed += 1
                if processed % 1_000_000 == 0:
                    print(f"\rProcessed {processed:,} lines, "
                        f"found {found:,} mappings", end='')

                if 'hasWikidataId' not in line:
                    continue

                subj_match = self.yago_entity_pattern.match(line.split()[0])
                qid_match  = re.search(r'Q\d+', line)
                if subj_match and qid_match:
                    y2q[subj_match.group(1)] = qid_match.group(0)
                    found += 1
        print(f"\nFound {found:,} Wikidata mappings\n")
        return y2q

    
    def calculate_type_statistics(self) -> Dict[str, Dict]:
        """タイプの統計情報を計算"""
        print("Calculating type statistics...")
        
        type_stats = {}
        
        for yago_type, entities in self.type_entities.items():
            if len(entities) >= self.min_entities_per_category:
                # エンティティ数
                entity_count = len(entities)
                
                # 平均人気度
                pop_scores = [self.popqa_entities[qid]['pop'] for qid in entities if qid in self.popqa_entities]
                avg_popularity = sum(pop_scores) / len(pop_scores) if pop_scores else 0
                
                # 階層の深さ計算
                depth = self.calculate_type_depth(yago_type)
                
                # カテゴリ名の品質スコア
                name_quality = self.calculate_name_quality(yago_type)
                
                # 総合スコア計算
                if depth < self.min_tree_depth:
                    depth_penalty = (self.min_tree_depth - depth) * 2
                else:
                    depth_penalty = abs(depth - 3.5) * 0.5
                
                score = (entity_count * 0.4 + 
                        avg_popularity * 0.3 + 
                        name_quality * 0.2 - 
                        depth_penalty * 0.1)
                
                type_stats[yago_type] = {
                    'entity_count': entity_count,
                    'avg_popularity': avg_popularity,
                    'depth': depth,
                    'name_quality': name_quality,
                    'score': score,
                    'entities': entities
                }
        
        return type_stats
    
    def calculate_type_depth(self, yago_type: str) -> int:
        """タイプの階層の深さを計算"""
        def get_depth(type_name, visited=None):
            if visited is None:
                visited = set()
            if type_name in visited:
                return 0
            visited.add(type_name)
            
            max_child_depth = 0
            for parent, children in self.type_hierarchy.items():
                if type_name in children:
                    max_child_depth = max(max_child_depth, 1 + get_depth(parent, visited.copy()))
            
            return max_child_depth
        
        return get_depth(yago_type)
    
    def calculate_name_quality(self, yago_type: str) -> float:
        """カテゴリ名の品質スコア計算"""
        name = yago_type.lower()
        
        quality = 100.0
        
        # 長さペナルティ
        quality -= len(name) * 2
        
        # 数字やアンダースコアのペナルティ
        quality -= name.count('_') * 5
        quality -= sum(c.isdigit() for c in name) * 3
        
        # 一般的なカテゴリキーワードにボーナス
        good_keywords = ['person', 'place', 'organization', 'event', 'work', 'species', 'concept', 'thing', 'location', 'creative', 'schema']
        for keyword in good_keywords:
            if keyword in name:
                quality += 20
        
        # 特殊文字やWikipedia特有の接尾語にペナルティ
        bad_patterns = ['wikipage', 'redirect', 'disambiguation', 'category', 'wordnet']
        for pattern in bad_patterns:
            if pattern in name:
                quality -= 30
        
        return max(quality, 0)
    
    def select_representative_categories(self, type_stats: Dict[str, Dict], 
                                       num_categories: int = 20) -> List[str]:
        """代表的なカテゴリを選択"""
        print(f"Selecting top {num_categories} representative categories...")
        
        # スコア順でソート
        sorted_types = sorted(type_stats.items(), 
                             key=lambda x: x[1]['score'], 
                             reverse=True)
        
        selected = []
        selected_entities = set()
        
        for yago_type, stats in sorted_types:
            if len(selected) >= num_categories:
                break
            
            # 重複するエンティティが多すぎる場合はスキップ
            overlap = len(stats['entities'] & selected_entities)
            if len(stats['entities']) > 0 and overlap / len(stats['entities']) > 0.7:
                continue
            
            selected.append(yago_type)
            selected_entities.update(stats['entities'])
            
            print(f"Selected: {yago_type} "
                  f"(entities: {stats['entity_count']}, "
                  f"avg_pop: {stats['avg_popularity']:.2f}, "
                  f"depth: {stats['depth']}, "
                  f"score: {stats['score']:.2f})")
        
        return selected
    
    def build_taxonomy_tree(self, selected_categories: List[str]):
        """選択されたカテゴリで分類階層ツリーを構築（深さ3以上を確保）"""
        print("Building taxonomy tree with minimum depth 3...")
        
        self.selected_categories = set(selected_categories)
        
        # エンティティ情報を保存
        for qid in self.popqa_entities:
            if qid in self.entity_types:
                entity_categories = self.entity_types[qid] & self.selected_categories
                if entity_categories:
                    self.entity_info[qid] = {
                        'wiki_title': self.popqa_entities[qid]['label'],
                        'qid': qid,
                        'pop': self.popqa_entities[qid]['pop'],
                        'categories': list(entity_categories)
                    }
        
        # 階層の深さ分析とグループ化
        category_levels = self.analyze_category_levels(selected_categories)
        
        # 3レベル以上の階層を構築
        self.build_multilevel_hierarchy(category_levels)
        
        # 残ったカテゴリを適切なレベルに配置
        self.assign_remaining_categories(selected_categories)
    
    def analyze_category_levels(self, categories: List[str]) -> Dict[int, List[str]]:
        """カテゴリを階層レベルごとに分析・分類"""
        category_levels = defaultdict(list)
        category_depths = {}
        
        # 各カテゴリの階層深度を計算
        for category in categories:
            depth = self.calculate_detailed_type_depth(category)
            category_depths[category] = depth
        
        # 深度でグループ化（最低3レベルを確保）
        max_depth = max(category_depths.values()) if category_depths else 0
        min_levels = max(3, max_depth)
        
        for category, depth in category_depths.items():
            # 深度を正規化して3レベル以上に分散
            normalized_level = min(int((depth / max(max_depth, 1)) * (min_levels - 1)), min_levels - 1)
            category_levels[normalized_level].append(category)
        
        return category_levels
    
    def calculate_detailed_type_depth(self, yago_type: str) -> int:
        """タイプの詳細な階層深度を計算（サイクル検出付き）"""
        def get_depth_recursive(type_name, visited=None, depth=0):
            if visited is None:
                visited = set()
            if type_name in visited or depth > 10:
                return depth
            
            visited.add(type_name)
            max_parent_depth = depth
            
            # 親タイプを探索
            for parent, children in self.type_hierarchy.items():
                if type_name in children:
                    parent_depth = get_depth_recursive(parent, visited.copy(), depth + 1)
                    max_parent_depth = max(max_parent_depth, parent_depth)
            
            return max_parent_depth
        
        return get_depth_recursive(yago_type)
    
    def build_multilevel_hierarchy(self, category_levels: Dict[int, List[str]]):
        """複数レベルの階層を構築"""
        print(f"Building hierarchy with {len(category_levels)} levels...")
        
        sorted_levels = sorted(category_levels.items())
        
        for level, categories in sorted_levels:
            print(f"Level {level}: {len(categories)} categories")
            
            for category in categories:
                # このレベルのカテゴリに属するエンティティを追加
                entities_in_category = self.type_entities[category] & set(self.entity_info.keys())
                for entity in entities_in_category:
                    self.taxonomy_tree[category].add(entity)
                
                # 上位レベルとの階層関係を構築
                if level > 0:
                    self.connect_to_parent_level(category, sorted_levels[:level])
                
                # 下位レベルとの階層関係を構築
                remaining_levels = sorted_levels[level+1:]
                if remaining_levels:
                    self.connect_to_child_level(category, remaining_levels)
    
    def connect_to_parent_level(self, category: str, parent_levels: List[Tuple[int, List[str]]]):
        """カテゴリを上位レベルの適切な親に接続"""
        # YAGO階層情報を使用して親を探す
        for parent, children in self.type_hierarchy.items():
            if category in children and parent in self.selected_categories:
                # 親が上位レベルにあるかチェック
                for level, categories in parent_levels:
                    if parent in categories:
                        self.taxonomy_tree[parent].add(category)
                        return
        
        # YAGO階層で親が見つからない場合、最も関連性の高い上位カテゴリに接続
        if parent_levels:
            best_parent = self.find_best_semantic_parent(category, parent_levels[-1][1])
            if best_parent:
                self.taxonomy_tree[best_parent].add(category)
    
    def connect_to_child_level(self, category: str, child_levels: List[Tuple[int, List[str]]]):
        """カテゴリの下位レベルの子を接続"""
        for level, child_categories in child_levels[:1]:
            for child in child_categories:
                # YAGO階層で直接の子関係をチェック
                if child in self.type_hierarchy.get(category, set()):
                    self.taxonomy_tree[category].add(child)
    
    def find_best_semantic_parent(self, category: str, potential_parents: List[str]) -> Optional[str]:
        """セマンティックに最も適切な親カテゴリを見つける"""
        category_lower = category.lower()
        
        best_parent = None
        best_score = 0
        
        for parent in potential_parents:
            parent_lower = parent.lower()
            
            # 共通の単語数を計算
            category_words = set(category_lower.split('_'))
            parent_words = set(parent_lower.split('_'))
            common_words = category_words & parent_words
            
            if common_words:
                score = len(common_words) / max(len(category_words), len(parent_words))
                if score > best_score:
                    best_score = score
                    best_parent = parent
        
        return best_parent
    
    def assign_remaining_categories(self, selected_categories: List[str]):
        """階層に未割り当ての残ったカテゴリを適切に配置"""
        for category in selected_categories:
            if category not in self.taxonomy_tree:
                entities_in_category = self.type_entities[category] & set(self.entity_info.keys())
                if entities_in_category:
                    for entity in entities_in_category:
                        self.taxonomy_tree[category].add(entity)
                    
                    self.find_and_connect_parent(category)
    
    def find_and_connect_parent(self, category: str):
        """カテゴリの適切な親を見つけて接続"""
        # まずYAGO階層で親を探す
        for parent, children in self.type_hierarchy.items():
            if category in children and parent in self.selected_categories:
                if parent in self.taxonomy_tree:
                    self.taxonomy_tree[parent].add(category)
                    return
        
        # YAGO階層で見つからない場合、最も類似したカテゴリを親とする
        existing_categories = list(self.taxonomy_tree.keys())
        if existing_categories:
            best_parent = self.find_best_semantic_parent(category, existing_categories)
            if best_parent:
                self.taxonomy_tree[best_parent].add(category)
    
    def generate_tree_data(self) -> List[Dict]:
        """ツリー構造からJSONLデータを生成"""
        nodes = []
        tree_id = 1
        
        # すべてのノード（カテゴリ + エンティティ）
        all_nodes = set(self.selected_categories)
        for category in self.selected_categories:
            all_nodes.update(self.taxonomy_tree[category])
        
        for node in all_nodes:
            edges = []
            
            # 子ノードへのエッジ (P527: has part)
            if node in self.taxonomy_tree:
                for child in self.taxonomy_tree[node]:
                    edges.append({
                        "property": "P527",
                        "target_qid": child,
                        "target_label": self.get_node_label(child)
                    })
            
            # 親ノードへのエッジ (P361: part of)
            for parent, children in self.taxonomy_tree.items():
                if node in children:
                    edges.append({
                        "property": "P361",
                        "target_qid": parent,
                        "target_label": self.get_node_label(parent)
                    })
            
            # ノード情報を作成
            node_data = {
                "tree_id": tree_id,
                "wiki_title": self.get_node_label(node),
                "qid": node,
                "num_edges": len(edges),
                "source_props": list(set([edge["property"] for edge in edges]))
            }
            
            # PopQAエンティティの場合はpopを追加
            if node in self.entity_info:
                node_data["pop"] = self.entity_info[node]["pop"]
            
            if edges:
                node_data["edges"] = edges
            
            nodes.append(node_data)
        
        return nodes
    
    def get_node_label(self, node: str) -> str:
        """ノードのラベルを取得"""
        if node in self.entity_info:
            return self.entity_info[node]['wiki_title']
        else:
            # カテゴリ名をフォーマット
            return node.replace('_', ' ').replace('schema ', '').title()
    
    def save_dataset(self, output_file: str):
        """データセットをJSONL形式で保存"""
        print(f"Saving dataset to {output_file}...")
        
        nodes = self.generate_tree_data()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for node in nodes:
                f.write(json.dumps(node, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(nodes)} nodes to {output_file}")
    
    def create_dataset(self, yago_facts_file: str, yago_beyond_file: str, 
                      yago_taxonomy_file: str, output_file: str, num_categories: int):
        """完全なデータセット作成プロセス"""
        print("Starting taxonomy dataset creation...")
        
        # 1. PopQAエンティティを読み込み
        self.load_popqa_entities()

        # 2. rdf:type を解析（手書きパーサを呼ぶ）
        self.parse_yago_facts(yago_facts_file)
        
        # 3. YAGO beyond Wikipediaからタイプ情報を解析
        if yago_beyond_file:
            self.parse_yago_beyond_wikipedia(yago_beyond_file)
        
        # 4. YAGO分類階層を解析
        if yago_taxonomy_file:
            self.parse_yago_taxonomy(yago_taxonomy_file)
        
        # 5. タイプ統計を計算
        type_stats = self.calculate_type_statistics()
        
        if not type_stats:
            print("No valid categories found. Please check your YAGO files.")
            sys.exit(1)
        
        # 6. 代表的なカテゴリを選択
        selected_categories = self.select_representative_categories(type_stats, num_categories)
        
        if not selected_categories:
            print("No categories selected. Please adjust the parameters.")
            sys.exit(1)
        
        # 7. 分類階層ツリーを構築
        self.build_taxonomy_tree(selected_categories)
        
        # 8. データセットを保存
        self.save_dataset(output_file)
        
        print("Taxonomy dataset creation completed!")
        
        # 統計情報を表示
        print(f"\nDataset Statistics:")
        print(f"Total PopQA entities: {len(self.popqa_entities)}")
        print(f"Entities with types: {len(self.entity_types)}")
        print(f"Selected categories: {len(selected_categories)}")
        print(f"Entities in taxonomy: {len(self.entity_info)}")

def main():
    parser = argparse.ArgumentParser(description='Create taxonomy dataset from YAGO 4.5 dumps and PopQA entities')
    
    parser.add_argument('--yago-facts', required=True,
                       help='Path to YAGO facts file (yago-facts.ttl or .gz)')
    parser.add_argument('--yago-beyond', 
                       help='Path to YAGO beyond Wikipedia file (yago-beyond-wikipedia.ttl or .gz)')
    parser.add_argument('--yago-taxonomy', 
                       help='Path to YAGO taxonomy file (yago-taxonomy.ttl or .gz)')
    parser.add_argument('--output', default='taxonomy_dataset.jsonl',
                       help='Output JSONL file (default: taxonomy_dataset.jsonl)')
    parser.add_argument('--num-categories', type=int, default=20,
                       help='Number of representative categories to select (default: 20)')
    parser.add_argument('--min-entities', type=int, default=50,
                       help='Minimum entities per category (default: 50)')
    parser.add_argument('--max-depth', type=int, default=6,
                       help='Maximum category depth (default: 6)')
    parser.add_argument('--min-tree-depth', type=int, default=3,
                       help='Minimum tree depth (default: 3)')
    parser.add_argument('--popqa-dataset', default='masaki-sakata/popqa-unique-entities',
                       help='PopQA dataset name (default: masaki-sakata/popqa-unique-entities)')
    parser.add_argument('--yago-meta',
        required=True,
        help='Path to yago-meta-facts.ntx (contains hasWikidataId triples)')
    
    args = parser.parse_args()
    
    # ファイル存在チェック
    if not Path(args.yago_facts).exists():
        print(f"Error: YAGO facts file not found: {args.yago_facts}")
        sys.exit(1)
    
    if args.yago_beyond and not Path(args.yago_beyond).exists():
        print(f"Warning: YAGO beyond file not found: {args.yago_beyond}")
        args.yago_beyond = None
    
    if args.yago_taxonomy and not Path(args.yago_taxonomy).exists():
        print(f"Warning: YAGO taxonomy file not found: {args.yago_taxonomy}")
        args.yago_taxonomy = None
    
    if not Path(args.yago_meta).exists():
        print(f"Error: YAGO meta file not found: {args.yago_meta}")
        sys.exit(1)

    # データセット作成
    creator = TaxonomyDatasetCreator(
        min_entities_per_category=args.min_entities,
        max_category_depth=args.max_depth,
        min_tree_depth=args.min_tree_depth
    )
    
    creator.create_dataset(
        yago_facts_file=args.yago_facts,
        yago_beyond_file=args.yago_beyond,
        yago_taxonomy_file=args.yago_taxonomy,
        output_file=args.output,
        num_categories=args.num_categories
    )

if __name__ == "__main__":
    main()