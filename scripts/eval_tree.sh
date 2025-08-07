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

# ####################################
# # Random Embeddings
# ####################################
# echo "Running Random Embeddings..."
# MODEL="random_emb"
# RANDOM_DIM=8192
# RANDOM_STD=1.0
# RANDOM_SEED=42
# OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}/dim_${RANDOM_DIM}_std_${RANDOM_STD}_seed_${RANDOM_SEED}"

# uv run python3 eval_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --model ${MODEL} \
#     --device ${DEVICE} \
#     --export_visualizations \
#     --random_dim ${RANDOM_DIM} \
#     --random_std ${RANDOM_STD} \
#     --random_seed ${RANDOM_SEED} \
#     --verbose 

###################################
# gold_binary
###################################
# echo "Running gold_binary..."
# MODEL="gold_binary_left"
# METHOD="average"
# OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}"
# echo "Left-leaning:"
# uv run python3 eval_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --model ${MODEL} \
#     --method ${METHOD} \
#     --device ${DEVICE} \
#     --export_visualizations \
#     --verbose 

# MODEL="gold_binary_balanced"
# OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}"
# echo "Balanced:"
# uv run python3 eval_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --model ${MODEL} \
#     --method ${METHOD} \
#     --device ${DEVICE} \
#     --export_visualizations \
#     --verbose 



# ###################################
# # FastText
# ###################################
# echo "Running FastText..."
# MODEL="fasttext"
# METHOD="average"
# OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}"

# uv run python3 eval_tree.py \
#     --input ${INPUT} \
#     --output_dir ${OUTPUT_DIR} \
#     --model ${MODEL} \
#     --method ${METHOD} \
#     --device ${DEVICE} \
#     --export_visualizations \
#     --verbose 


# ####################################
# # GPT-2 with different layers and templates
# ####################################
# echo "Running GPT-2 with different layers and templates..."
# MODEL="gpt2"
# METHOD="last_token"

# # Templates to test
# # TEMPLATES=("entity_only")
# TEMPLATES=("entity_only" "occupation_question" "gift")
# # Layers to test
# LAYERS=(0 2 4 6 8 12)
# # LAYERS=(0 2 6)

# for TEMPLATE in "${TEMPLATES[@]}"; do
#     for LAYER in "${LAYERS[@]}"; do
#         echo "GPT-2: Template=${TEMPLATE}, Layer=${LAYER}"
#         OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL}/template_${TEMPLATE}/layer_${LAYER}"
        
#         uv run python3 eval_tree.py \
#             --input ${INPUT} \
#             --output_dir ${OUTPUT_DIR} \
#             --model ${MODEL} \
#             --method ${METHOD} \
#             --layer ${LAYER} \
#             --device ${DEVICE} \
#             --template ${TEMPLATE} \
#             --export_visualizations \
#             --verbose 
#     done
# done

####################################
# Meta-Llama-3-8B with different layers and templates
####################################
echo "Running Meta-Llama-3-8B with different layers and templates..."
MODEL="meta-llama/Meta-Llama-3-8B"
METHOD="last_token"

# Templates to test
# TEMPLATES=("entity_only")
TEMPLATES=("entity_only" "occupation_question" "gift")
# Layers to test (Llama-3-8B has 32 layers)
# LAYERS=(0 2 6 10 15 20 25 32)
LAYERS=(6)

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
            --verbose 
    done
done

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: ${OUTPUT_BASE_DIR}"
echo "=========================================="