output="/work03/masaki/data/taxonomy/taxonomy_from_popQA.jsonl"
cache_file="/work03/masaki/data/taxonomy/qid2type.json.gz"

min_entities=10
min_tree_depth=4
target_trees=1000
min_nodes_per_tree=8

min_parent_categories=1
min_leaf_categories=2
min_entities_per_leaf=3


poetry run python3 build_taxonomy_dataset_api.py \
  --output $output \
  --min-entities $min_entities \
  --min-tree-depth $min_tree_depth \
  --target-trees $target_trees \
  --min-nodes-per-tree   $min_nodes_per_tree \
  --min-parent-categories $min_parent_categories \
  --min-leaf-categories $min_leaf_categories \
  --min-entities-per-leaf $min_entities_per_leaf \
  --entity-type-cache $cache_file 


