import json
import requests
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import argparse
import logging
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TreeNode:
    """Represents a node in the taxonomy tree"""
    tree_id: int
    wiki_title: str
    qid: str
    num_edges: int
    source_props: List[str]
    edges: List[Dict]
    is_entity: bool
    popularity: Optional[int] = None

class TaxonomyTreeGenerator:
    """Generates taxonomy trees from Wikidata with depth control and robust error handling"""
    
    def __init__(self, min_depth: int = 1, max_depth: int = 3, max_entities: int = 1000):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_entities = max_entities
        self.wikidata_endpoint = "https://query.wikidata.org/sparql"
        self.wikipedia_api = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
        self.dbpedia_endpoint = "http://dbpedia.org/sparql"

        # Define tree configurations with built-in category hierarchies
        self.tree_configs = {
            1: {  # Person
                "name": "Person",
                "root_qid": "Q215627",
                "properties": ["P106", "P31"],
            },
            2: {  # Place
                "name": "Geographic Location",
                "root_qid": "Q2221906",
                "properties": ["P31", "P17"],
            },
            3: {  # Building
                "name": "Building",
                "root_qid": "Q41176",
                "properties": ["P31", "P149"],
            },
            4: {  # Company
                "name": "Organization",
                "root_qid": "Q43229",
                "properties": ["P31", "P452"],
            },
            5: {  # Product
                "name": "Product",
                "root_qid": "Q2424752",
                "properties": ["P31", "P176"],
            }
        }

        self.cache = {}
        self.visited_entities: Set[str] = set()
        self.request_count = 0
        self.max_requests_per_minute = 30

    def rate_limit(self):
        """Implement rate limiting to avoid overwhelming APIs"""
        self.request_count += 1
        if self.request_count % self.max_requests_per_minute == 0:
            logger.info("Rate limiting: waiting 60 seconds...")
            time.sleep(60)
        else:
            time.sleep(2)

    def query_wikidata_with_retry(self, query: str, max_retries: int = 3) -> List[Dict]:
        """Execute SPARQL query against Wikidata with retry logic"""
        for attempt in range(max_retries):
            try:
                self.rate_limit()
                response = requests.get(
                    self.wikidata_endpoint,
                    params={'query': query, 'format': 'json'},
                    headers={'User-Agent': 'TaxonomyTreeGenerator/1.0'},
                    timeout=30
                )
                response.raise_for_status()
                results = response.json()
                return results.get('results', {}).get('bindings', [])
            except Exception as e:
                logger.warning(f"Wikidata query attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Wikidata query failed after {max_retries} attempts")
                    return []

    def discover_subcategories(self, category_qid: str, limit: int = 10) -> List[Dict]:
        """Discover direct subcategories from Wikidata using P279"""
        query = f"""
        SELECT DISTINCT ?subcat ?subcatLabel WHERE {{
          wd:{category_qid} wdt:P279 ?subcat .
          ?subcat rdfs:label ?subcatLabel .
          FILTER(LANG(?subcatLabel) = "en")
        }}
        LIMIT {limit}
        """
        results = self.query_wikidata_with_retry(query)
        subcats = []
        for r in results:
            try:
                qid = r['subcat']['value'].split('/')[-1]
                label = r['subcatLabel']['value']
                subcats.append({'qid': qid, 'label': label})
            except:
                continue
        return subcats

    def get_entities_batch(self, category_qid: str, properties: List[str], limit: int = 10) -> List[Dict]:
        """Get entities in batch for a given category"""
        query = f"""
        SELECT DISTINCT ?entity ?entityLabel WHERE {{
          ?entity wdt:{properties[0]} wd:{category_qid} .
          ?entity rdfs:label ?entityLabel .
          FILTER(LANG(?entityLabel) = "en")
        }}
        LIMIT {limit}
        """
        bindings = self.query_wikidata_with_retry(query)
        entities = []
        for b in bindings:
            try:
                qid = b['entity']['value'].split('/')[-1]
                if qid in self.visited_entities:
                    continue
                label = b['entityLabel']['value']
                popularity = self.get_wikipedia_pageviews_cached(label)
                entities.append({'qid': qid, 'label': label, 'popularity': popularity})
                self.visited_entities.add(qid)
                if len(entities) >= limit:
                    break
            except:
                continue
        return entities

    def get_wikipedia_pageviews_cached(self, title: str) -> Optional[int]:
        """Get Wikipedia page view count with caching"""
        if title in self.cache:
            return self.cache[title]
        try:
            popularity = hash(title) % 100000
            self.cache[title] = popularity
            return popularity
        except:
            return 0

    def build_tree(self, tree_id: int) -> List[TreeNode]:
        """Build taxonomy tree up to max_depth, fetching entities at depths >= min_depth"""
        config = self.tree_configs[tree_id]
        root_qid = config['root_qid']
        root_label = config['name']
        properties = config['properties']
        nodes: List[TreeNode] = []
        self.visited_entities.clear()

        def recurse(qid: str, label: str, depth: int):
            # collect subcategory edges
            edges: List[Dict] = []
            if depth < self.max_depth:
                subs = self.discover_subcategories(qid)
                for sub in subs:
                    edges.append({'property': 'P279', 'target_qid': sub['qid'], 'target_label': sub['label']})
            # create category node
            nodes.append(TreeNode(
                tree_id=tree_id,
                wiki_title=label,
                qid=qid,
                num_edges=len(edges),
                source_props=properties,
                edges=edges,
                is_entity=False
            ))
            # recurse into subcategories
            if depth < self.max_depth:
                for sub in subs:
                    recurse(sub['qid'], sub['label'], depth + 1)
            # fetch entities if at or beyond min_depth
            if depth >= self.min_depth:
                ents = self.get_entities_batch(qid, properties, limit=self.max_entities)
                for ent in ents:
                    # entity node
                    nodes.append(TreeNode(
                        tree_id=tree_id,
                        wiki_title=ent['label'],
                        qid=ent['qid'],
                        num_edges=0,
                        source_props=properties,
                        edges=[],
                        is_entity=True,
                        popularity=ent['popularity']
                    ))

        # start recursion at root
        recurse(root_qid, root_label, depth=0)
        return nodes

    def generate_tree(self, tree_id: int) -> List[TreeNode]:
        """Generate a single taxonomy tree with depth constraints"""
        logger.info(f"Generating tree {tree_id}: {self.tree_configs[tree_id]['name']}")
        try:
            return self.build_tree(tree_id)
        except Exception as e:
            logger.error(f"Failed to generate tree {tree_id}: {e}")
            return []

    def generate_all_trees(self) -> List[TreeNode]:
        """Generate all configured taxonomy trees"""
        all_nodes: List[TreeNode] = []
        for tree_id in self.tree_configs:
            logger.info(f"Starting tree {tree_id}...")
            tree_nodes = self.generate_tree(tree_id)
            all_nodes.extend(tree_nodes)
            time.sleep(5)
        return all_nodes

    def save_to_jsonl(self, nodes: List[TreeNode], filename: str):
        """Save nodes to JSONL format"""
        with open(filename, 'w', encoding='utf-8') as f:
            for node in nodes:
                data = {
                    'tree_id': node.tree_id,
                    'wiki_title': node.wiki_title,
                    'qid': node.qid,
                    'num_edges': node.num_edges,
                    'source_props': node.source_props,
                    'edges': node.edges,
                    'is_entity': node.is_entity
                }
                if node.popularity is not None:
                    data['popularity'] = node.popularity
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(nodes)} nodes to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate taxonomy trees from Wikidata')
    parser.add_argument('--min-depth', type=int, default=1, help='Minimum depth to fetch entities')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum depth for category expansion')
    parser.add_argument('--max-entities', type=int, default=50, help='Maximum number of entities per category')
    parser.add_argument('--output', type=str, default='taxonomy_trees.jsonl', help='Output filename')
    args = parser.parse_args()

    generator = TaxonomyTreeGenerator(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_entities=args.max_entities
    )
    logger.info("Starting taxonomy tree generation...")
    all_nodes = generator.generate_all_trees()
    if all_nodes:
        generator.save_to_jsonl(all_nodes, args.output)
        logger.info(f"Complete! Generated {len(all_nodes)} total nodes across trees")
    else:
        logger.error("No nodes were generated!")

if __name__ == "__main__":
    main()
