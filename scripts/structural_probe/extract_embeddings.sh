# MODEL="gpt2"
MODEL="meta-llama/Meta-Llama-3-8B"
DATA_PATH="/home/masaki/hierarchical-repr/EntityTree/input/300people/train250.jsonl"
OUTPUT_DIR="/work03/masaki/model/hierarchical-repr/${MODEL}/train250/"
NUM_SPLITS=1000
device="cuda:1"

uv run python3 extract_embeddings.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --device $device \
    --model_name $MODEL \
    --split_tree \
    --num_splits $NUM_SPLITS \
    --is_verbose


DATA_PATH="/home/masaki/hierarchical-repr/EntityTree/input/300people/test50.jsonl"
OUTPUT_DIR="/work03/masaki/model/hierarchical-repr/${MODEL}/test50/"

uv run python3 extract_embeddings.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --device $device \
    --model_name $MODEL \
    --is_verbose