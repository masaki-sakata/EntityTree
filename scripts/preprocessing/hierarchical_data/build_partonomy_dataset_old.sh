
src="/work03/masaki/data/zelda/train_data/pre_processed/target_word_appearance_indices/remove_duplicate/repeat_sentence/all_data/aggregate/zelda_train_3types.jsonl"
dst="/work03/masaki/data/zelda/partonomy/zelda_train_partonomy.jsonl"

poetry run python3 build_partonomy_dataset.py \
  --src $src \
  --dst $dst 