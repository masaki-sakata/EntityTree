#!/bin/bash

# Evaluation script for tree distances using TreeDist

INPUT="./input/taxonomy_person_test50.jsonl"
OUTPUT_DIR="./output/eval_results"

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# # Evaluate meta-llama/Meta-Llama-3-8B
# MODEL="meta-llama/Meta-Llama-3-8B"
# METHOD=last_token
# DEVICE=cuda:0
# OUTPUT="${OUTPUT_DIR}/eval_${MODEL##*/}_${METHOD}.csv"

# echo "Evaluating ${MODEL} with ${METHOD}..."
# poetry run python3 eval_tree.py \
#     --input ${INPUT} \
#     --output ${OUTPUT} \
#     --model ${MODEL} \
#     --method ${METHOD} \
#     --device ${DEVICE}

# Evaluate GPT-2
MODEL="gpt2"
METHOD=last_token
DEVICE=cuda:0
OUTPUT="${OUTPUT_DIR}/eval_${MODEL}_${METHOD}.csv"

echo "Evaluating ${MODEL} with ${METHOD}..."
poetry run python3 eval_tree.py \
    --input ${INPUT} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE}

# # Evaluate FastText
# MODEL="fasttext"
# METHOD=average
# DEVICE=cuda:0
# OUTPUT="${OUTPUT_DIR}/eval_${MODEL}_${METHOD}.csv"

# echo "Evaluating ${MODEL} with ${METHOD}..."
# poetry run python3 eval_tree.py \
#     --input ${INPUT} \
#     --output ${OUTPUT} \
#     --model ${MODEL} \
#     --method ${METHOD} \
#     --device ${DEVICE}

# echo "All evaluations complete. Results saved in ${OUTPUT_DIR}/"