
output="/work03/masaki/data/zelda/partonomy/zelda_train_partonomy_from_wikidata.jsonl"


poetry run python3 build_partonomy_dataset.py \
    --output $output \
    --trees 35 \
    --max-depth 4 \
    --min-edges 12 \
    --user-agent "MyTree/0.2 (frisk0zisan@gmail.com)"