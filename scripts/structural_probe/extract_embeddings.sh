
# # partonomy data embeddings extraction script
# data_type="test_50"
# # text_mode="title_desc_title"
# text_mode="title"


# # # model_name="meta-llama/Meta-Llama-3-8B"
# # model_name="gpt2"
# # model_type="hf"

# # data_path="/work03/masaki/data/zelda/partonomy/${data_type}.jsonl"
# # output_dir="/work03/masaki/model/partonomy/embeddings/${data_type}/$model_name"
# # device="cuda:3"

# # poetry run python3 extract_embeddings.py \
# #     --data_path $data_path \
# #     --output_dir $output_dir \
# #     --model_name $model_name \
# #     --device $device \
# #     --text_mode $text_mode \
# #     --model_type $model_type \
# #     --is_verbose 


# model_type="fasttext"
# model_name="fasttext"
# vector_path="/work03/masaki/model/fastText_vec/wiki-news-300d-1M-subword.bin"
# data_path="/work03/masaki/data/zelda/partonomy/${data_type}.jsonl"
# output_dir="/work03/masaki/model/partonomy/embeddings/${data_type}/$model_name"
# device="cuda:3"
# poetry run python3 extract_embeddings.py \
#     --data_path $data_path \
#     --output_dir $output_dir \
#     --model_name $model_name \
#     --device $device \
#     --text_mode $text_mode \
#     --model_type $model_type \
#     --vector_path $vector_path \
#     --is_verbose 




# # taxonomy data embeddings extraction script

text_mode="title"

# # model_name="meta-llama/Meta-Llama-3-8B"
# model_name="gpt2"
# model_type="hf"

# data_path="/work03/masaki/data/taxonomy/taxonomy_from_popQA.jsonl"
# output_dir="/work03/masaki/model/taxonomy/embeddings/$model_name"
# device="cuda:0"

# poetry run python3 extract_embeddings.py \
#     --data_path $data_path \
#     --output_dir $output_dir \
#     --model_name $model_name \
#     --device $device \
#     --text_mode $text_mode \
#     --model_type $model_type \
#     --is_verbose 


model_type="fasttext"
model_name="fasttext"
vector_path="/work03/masaki/model/fastText_vec/wiki-news-300d-1M-subword.bin"
data_path="/work03/masaki/data/taxonomy/taxonomy_from_popQA.jsonl"
output_dir="/work03/masaki/model/taxonomy/embeddings/$model_name"
device="cuda:3"
poetry run python3 extract_embeddings.py \
    --data_path $data_path \
    --output_dir $output_dir \
    --model_name $model_name \
    --device $device \
    --text_mode $text_mode \
    --model_type $model_type \
    --vector_path $vector_path \
    --is_verbose 

