#!/bin/bash

# Common parameters
INPUT="../input/taxonomy_person_test50.jsonl"
OUTPUT_BASE_DIR="../output/eval_tree"
DEVICE=cuda:1

# Enable verbose mode for debugging token usage
# Set to empty string to disable: VERBOSE=""
VERBOSE=""

echo "=========================================="
echo "Tree Evaluation with Multiple Models"
echo "=========================================="

####################################
# Random Embeddings
####################################
echo "Running Random Embeddings..."
MODEL="random_emb"
RANDOM_DIM=768
RANDOM_STD=1.0
RANDOM_SEED=42
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}/dim_${RANDOM_DIM}_std_${RANDOM_STD}_seed_${RANDOM_SEED}"

uv run python3 eval_tree.py \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL} \
    --device ${DEVICE} \
    --export_visualizations \
    $VERBOSE

####################################
# FastText
####################################
echo "Running FastText..."
MODEL="fasttext"
METHOD="average"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}"

uv run python3 eval_tree.py \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --export_visualizations \
    $VERBOSE

####################################
# GPT-2 with different layers and templates
####################################
echo "Running GPT-2 with different layers and templates..."
MODEL="gpt2"
METHOD="last_token"

# Templates to test
TEMPLATES=("entity_only")
# TEMPLATES=("entity_only" "occupation_question" "gift")
# Layers to test
LAYERS=(0 2 4 12)

for TEMPLATE in "${TEMPLATES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        echo "GPT-2: Template=${TEMPLATE}, Layer=${LAYER}"
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}/template_${TEMPLATE}/layer_${LAYER}"
        
        uv run python3 eval_tree.py \
            --input ${INPUT} \
            --output_dir ${OUTPUT_DIR} \
            --model ${MODEL} \
            --method ${METHOD} \
            --layer ${LAYER} \
            --device ${DEVICE} \
            --template ${TEMPLATE} \
            --export_visualizations \
            $VERBOSE
    done
done

####################################
# Meta-Llama-3-8B with different layers and templates
####################################
echo "Running Meta-Llama-3-8B with different layers and templates..."
MODEL="meta-llama/Meta-Llama-3-8B"
METHOD="last_token"

# Templates to test
TEMPLATES=("entity_only")
# TEMPLATES=("entity_only" "occupation_question" "gift")
# Layers to test (Llama-3-8B has 32 layers)
LAYERS=(0 2 5 10 15 20 25 32)

for TEMPLATE in "${TEMPLATES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        echo "Llama-3-8B: Template=${TEMPLATE}, Layer=${LAYER}"
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}/template_${TEMPLATE}/layer_${LAYER}"
        
        uv run python3 eval_tree.py \
            --input ${INPUT} \
            --output_dir ${OUTPUT_DIR} \
            --model ${MODEL} \
            --method ${METHOD} \
            --layer ${LAYER} \
            --device ${DEVICE} \
            --template ${TEMPLATE} \
            --export_visualizations \
            $VERBOSE
    done
done

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: ${OUTPUT_BASE_DIR}"
echo "=========================================="