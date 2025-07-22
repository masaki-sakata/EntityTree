#!/usr/bin/env bash
set -eu


# taxonomy title

text_mode="title"


model_name="meta-llama/Meta-Llama-3-8B"
probe_rank=300
layer=6
device="cuda:0"
seed=41

probe_path="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/probe/probe_layer${layer}_rank${probe_rank}.pt"

embedding_dir="/work03/masaki/model/taxonomy/embeddings/${model_name}/${text_mode}"
save_dir="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/"

poetry run python3 eval_probe.py \
    --probe_path    "${probe_path}" \
    --embedding_dir "${embedding_dir}" \
    --layer         "${layer}" \
    --device        "${device}" \
    --probe_rank    "${probe_rank}" \
    --seed          "${seed}" \
    --save_dir      "${save_dir}"


# taxonomy_icl_nl
# text_mode="icl_NL"


# model_name="meta-llama/Meta-Llama-3-8B"
# probe_rank=300
# layer=6
# device="cuda:0"
# seed=41
# icl_repeat_num=1

# probe_path="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/Repeat_${icl_repeat_num}/probe/probe_layer${layer}_rank${probe_rank}.pt"

# embedding_dir="/work03/masaki/model/taxonomy/embeddings/${model_name}/${text_mode}"
# # save_dir="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/Repeat_${icl_repeat_num}"

# poetry run python3 eval_probe.py \
#     --probe_path    "${probe_path}" \
#     --embedding_dir "${embedding_dir}" \
#     --layer         "${layer}" \
#     --device        "${device}" \
#     --probe_rank    "${probe_rank}" \
#     --seed          "${seed}" \
#     --icl_repeat_num "${icl_repeat_num}"
