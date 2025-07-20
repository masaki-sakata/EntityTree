
# MODEL="gpt2"
MODEL="meta-llama/Meta-Llama-3-8B"
LAYER=6
METHOD=last_token
SCALE=1.2
DEVICE=cuda:0


INPUT="./input/sample_text.txt"
OUTPUT="./output/${MODEL}/${LAYER}/scale${SCALE}_tree.html"


poetry run python3 visualize_story_tree.py \
    --input ${INPUT} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --method ${METHOD} \
    --layer ${LAYER} \
    --device ${DEVICE} \
    --scale ${SCALE}
