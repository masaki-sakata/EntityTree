output="/work03/masaki/data/taxonomy/taxonomy_from_popQA.jsonl"

yago_facts="/work03/masaki/data/yago-4.5/yago-facts.ttl"
yago_beyond="/work03/masaki/data/yago-4.5/yago-beyond-wikipedia.ttl"
yago_taxonomy="/work03/masaki/data/yago-4.5/yago-taxonomy.ttl"


poetry run python3 build_taxonomy_dataset.py \
  --yago-facts $yago_facts \
  --yago-beyond $yago_beyond \
  --yago-taxonomy $yago_taxonomy \
  --output $output 