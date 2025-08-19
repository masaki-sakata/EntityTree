#!/bin/bash

# Common parameters
INPUT="../input/300people/train250.jsonl"
OUTPUT_BASE_DIR="../models/lda"
DEVICE=cuda:0

echo "=========================================="
echo "Tree Evaluation with Multiple Models"
echo "=========================================="
echo "Input file: ${INPUT}"
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo "Device: ${DEVICE}"
echo "=========================================="


# ###################################
# # FastText
# ###################################
# echo "Running FastText..."
# MODEL="fasttext"
# METHOD="average"
# TEMPLATE="entity_only"
# LAYER=0

# uv run python3 train_lda_classifier.py \
#     --input ${INPUT} \
#     --outdir ${OUTPUT_BASE_DIR} \
#     --model ${MODEL} \
#     --method ${METHOD} \
#     --template ${TEMPLATE} \
#     --layer ${LAYER} \
#     --device ${DEVICE} \
#     --solver svd \
#     --n_components 5 \
#     --shrinkage none \
#     --verbose



# ####################################
# # GPT-2 with different layers and templates
# ####################################
# echo "Running GPT-2 with different layers and templates..."
# MODEL="gpt2"
# METHOD="last_token"
# # TEMPLATES=("entity_only")
# TEMPLATES=("entity_only" "occupation_question" "occupation_simple" "profession_query" "professional_intro" "gift")
# LAYERS=('all')

# for TEMPLATE in "${TEMPLATES[@]}"; do
#     for LAYER in "${LAYERS[@]}"; do
#         echo "Model=${MODEL}: Template=${TEMPLATE}, Layer=${LAYER}"
#         uv run python3 train_lda_classifier.py \
#         --input ${INPUT} \
#         --outdir ${OUTPUT_BASE_DIR} \
#         --model ${MODEL} \
#         --method ${METHOD} \
#         --template ${TEMPLATE} \
#         --layer ${LAYER} \
#         --device ${DEVICE} \
#         --solver svd \
#         --n_components 5 \
#         --shrinkage none \
#         --verbose
#      done
# done




####################################
# Meta-Llama-3-8B with different layers and templates
####################################
echo "Running Meta-Llama-3-8B with different layers and templates..."
MODEL="meta-llama/Meta-Llama-3-8B"
METHOD="last_token"
# TEMPLATES=("entity_only")
TEMPLATES=("entity_only" "occupation_question" "occupation_simple" "profession_query" "professional_intro" "gift")
# Layers to test (Llama-3-8B has 32 layers)
# LAYERS=(0 2 4 6 8 10 12 14 15 18 20 25 30 32)
LAYERS=('all')

for TEMPLATE in "${TEMPLATES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        echo "Model=${MODEL}: Template=${TEMPLATE}, Layer=${LAYER}"
        uv run python3 train_lda_classifier.py \
        --input ${INPUT} \
        --outdir ${OUTPUT_BASE_DIR} \
        --model ${MODEL} \
        --method ${METHOD} \
        --template ${TEMPLATE} \
        --layer ${LAYER} \
        --device ${DEVICE} \
        --solver svd \
        --n_components 5 \
        --shrinkage none \
        --verbose
     done
done

