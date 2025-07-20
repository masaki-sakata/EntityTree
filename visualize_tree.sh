
# MODEL="gpt2"
MODEL="meta-llama/Meta-Llama-3-8B"
METHOD=last_token

DEVICE=cuda:0

INPUT="./input/sample_text.txt"
OUTPUT="./output/${MODEL}"

poetry run python3 visualize_tree.py \
    --input ${INPUT} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} 
    



MODEL="gpt2"
METHOD=last_token
DEVICE=cuda:0
INPUT="./input/sample_text.txt"
OUTPUT="./output/${MODEL}"
poetry run python3 visualize_tree.py \
    --input ${INPUT} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} 
    



MODEL="fasttext"
METHOD=average

DEVICE=cuda:0
INPUT="./input/sample_text.txt"
OUTPUT="./output/${MODEL}"
poetry run python3 visualize_tree.py \
    --input ${INPUT} \
    --output ${OUTPUT} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} 
    
