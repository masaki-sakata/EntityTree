
MODEL="meta-llama/Meta-Llama-3-8B"
METHOD=last_token
DEVICE=cuda:0

INPUT="./input/taxonomy_person_test50.jsonl"
OUTPUT_DIR="./output/${MODEL}"
OUTPUT_FILE_NAME="taxonomy_person_test50"


poetry run python3 visualize_tree.py \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --output_file_name ${OUTPUT_FILE_NAME} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --export_png
    



MODEL="gpt2"
METHOD=last_token
DEVICE=cuda:0

INPUT="./input/taxonomy_person_test50.jsonl"
OUTPUT_DIR="./output/${MODEL}"
OUTPUT_FILE_NAME="taxonomy_person_test50"
poetry run python3 visualize_tree.py \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --output_file_name ${OUTPUT_FILE_NAME} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --export_png



MODEL="fasttext"
METHOD=average

DEVICE=cuda:0
INPUT="./input/taxonomy_person_test50.jsonl"
OUTPUT_DIR="./output/${MODEL}"
OUTPUT_FILE_NAME="taxonomy_person_test50"
poetry run python3 visualize_tree.py \
    --input ${INPUT} \
    --output_dir ${OUTPUT_DIR} \
    --output_file_name ${OUTPUT_FILE_NAME} \
    --model ${MODEL} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --export_png
    
