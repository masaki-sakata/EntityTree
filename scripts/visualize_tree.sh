#!/bin/bash

# Common parameters
INPUT="../input/taxonomy_person_test50.jsonl"
OUTPUT_FILE_NAME="taxonomy_person_test50"
DEVICE=cuda:0

# Enable verbose mode for debugging token usage
# Set to empty string to disable: VERBOSE=""
VERBOSE="--verbose"

# Meta-Llama model
MODEL="meta-llama/Meta-Llama-3-8B"
METHOD=last_token
BASE_OUTPUT_DIR="../output/${MODEL}"

# echo "Running Meta-Llama-3-8B..."
# # Template 1: entity_only
# TEMPLATE="entity_only"
# OUTPUT_DIR="${BASE_OUTPUT_DIR}/template_${TEMPLATE}"

# echo "Template: ${TEMPLATE}"
# uv run python3 visualize_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --output_file_name ${OUTPUT_FILE_NAME} \
#     --model ${MODEL} \
#     --method ${METHOD} \
#     --device ${DEVICE} \
#     --template ${TEMPLATE} \
#     --export_png \
#     $VERBOSE

# Template 2: occupation_question 
TEMPLATE="occupation_question"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/template_${TEMPLATE}"

echo "Template: ${TEMPLATE}"
uv run python3 visualize_tree.py \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --output_file_name ${OUTPUT_FILE_NAME} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --template ${TEMPLATE} \
    --export_png \
    $VERBOSE

# Template 3: gift
TEMPLATE="gift"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/template_${TEMPLATE}"

echo "Template: ${TEMPLATE}"
uv run python3 visualize_tree.py \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --output_file_name ${OUTPUT_FILE_NAME} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --template ${TEMPLATE} \
    --export_png \
    $VERBOSE





# GPT-2 model
MODEL="gpt2"
METHOD=last_token
BASE_OUTPUT_DIR="../output/${MODEL}"

echo "Testing different templates with ${MODEL}..."
echo "=============================================="

# # Template 1: entity_only
# TEMPLATE="entity_only"
# OUTPUT_DIR="${BASE_OUTPUT_DIR}/template_${TEMPLATE}"

# echo "Template: ${TEMPLATE}"
# uv run python3 visualize_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --output_file_name ${OUTPUT_FILE_NAME} \
#     --model ${MODEL} \
#     --method ${METHOD} \
#     --device ${DEVICE} \
#     --template ${TEMPLATE} \
#     --export_png \
#     $VERBOSE

# Template 2: occupation_question 
TEMPLATE="occupation_question"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/template_${TEMPLATE}"

echo "Template: ${TEMPLATE}"
uv run python3 visualize_tree.py \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --output_file_name ${OUTPUT_FILE_NAME} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --template ${TEMPLATE} \
    --export_png \
    $VERBOSE

# Template 3: gift
TEMPLATE="gift"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/template_${TEMPLATE}"

echo "Template: ${TEMPLATE}"
uv run python3 visualize_tree.py \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --output_file_name ${OUTPUT_FILE_NAME} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --template ${TEMPLATE} \
    --export_png \
    $VERBOSE



# # FastText model
# MODEL="fasttext"
# METHOD=average
# BASE_OUTPUT_DIR="../output/${MODEL}"

# echo "Running FastText..."
# uv run python3 visualize_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --output_file_name ${OUTPUT_FILE_NAME} \
#     --model ${MODEL} \
#     --method ${METHOD} \
#     --device ${DEVICE} \
#     --export_png \
#     $VERBOSE

# # Random embeddings - Multiple configurations
# echo "Running Random Embeddings with different configurations..."

# # Configuration 1: Low dimensional random embeddings
# MODEL="random_emb"
# RANDOM_DIM=300
# RANDOM_STD=1.0
# RANDOM_SEED=42
# OUTPUT_DIR="../output/${MODEL}/dim_${RANDOM_DIM}_std_${RANDOM_STD}_seed_${RANDOM_SEED}"


# echo "Random Emb: dim=${RANDOM_DIM}, std=${RANDOM_STD}, seed=${RANDOM_SEED}..."
# uv run python3 visualize_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --output_file_name ${OUTPUT_FILE_NAME} \
#     --model ${MODEL} \
#     --device ${DEVICE} \
#     --random_dim ${RANDOM_DIM} \
#     --random_std ${RANDOM_STD} \
#     --random_seed ${RANDOM_SEED} \
#     --export_png \
#     $VERBOSE

# # Configuration 2: High dimensional random embeddings
# MODEL="random_emb"
# RANDOM_DIM=768
# RANDOM_STD=1.0
# RANDOM_SEED=42
# OUTPUT_DIR="../output/${MODEL}/dim_${RANDOM_DIM}_std_${RANDOM_STD}_seed_${RANDOM_SEED}"


# echo "Random Emb: dim=${RANDOM_DIM}, std=${RANDOM_STD}, seed=${RANDOM_SEED}..."
# uv run python3 visualize_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --output_file_name ${OUTPUT_FILE_NAME} \
#     --model ${MODEL} \
#     --device ${DEVICE} \
#     --random_dim ${RANDOM_DIM} \
#     --random_std ${RANDOM_STD} \
#     --random_seed ${RANDOM_SEED} \
#     --export_png \
#     $VERBOSE

# # Configuration 3: Very high dimensional random embeddings
# MODEL="random_emb"
# RANDOM_DIM=4096
# RANDOM_STD=1.0
# RANDOM_SEED=42
# OUTPUT_DIR="../output/${MODEL}/dim_${RANDOM_DIM}_std_${RANDOM_STD}_seed_${RANDOM_SEED}"


# echo "Random Emb: dim=${RANDOM_DIM}, std=${RANDOM_STD}, seed=${RANDOM_SEED}..."
# uv run python3 visualize_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --output_file_name ${OUTPUT_FILE_NAME} \
#     --model ${MODEL} \
#     --device ${DEVICE} \
#     --random_dim ${RANDOM_DIM} \
#     --random_std ${RANDOM_STD} \
#     --random_seed ${RANDOM_SEED} \
#     --export_png \
#     $VERBOSE