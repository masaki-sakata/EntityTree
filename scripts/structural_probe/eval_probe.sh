# model="meta-llama/Meta-Llama-3-8B"
model="gpt2"
embedding_dir="/work03/masaki/model/hierarchical-repr/${model}/test50/title"
gold_data_path="/home/masaki/hierarchical-repr/EntityTree/input/300people/test50.jsonl"
layer=6
probe_rank=768
probe_type="distance"
probe_path="../../models/structural_probe/${model}/train250_title/n_comp${probe_rank}/layer${layer}/probe_${probe_type}_layer${layer}_rank${probe_rank}.pt"
save_dir="./probe_results/${model}/"
vis_path="./probe_results/${model}/vis/n_comp${probe_rank}/layer${layer}/vis_layer${layer}_rank${probe_rank}.html"
device="cuda:0"
seed=42

uv run python3 eval_probe.py \
    --model_name ${model} \
    --embedding_dir ${embedding_dir} \
    --gold_data_path ${gold_data_path} \
    --save_dir ${save_dir} \
    --layer ${layer} \
    --probe_type ${probe_type} \
    --probe_rank ${probe_rank} \
    --probe_path ${probe_path} \
    --vis_path ${vis_path} \
    --device ${device} \
    --seed ${seed} 




# #!/usr/bin/env bash
# set -euo pipefail

# # ===== Fixed parameters =====
# model="gpt2"
# embedding_dir="/work03/masaki/model/hierarchical-repr/${model}/test50/title"
# probe_type="distance"
# save_dir="./probe_results/${model}/"
# device="cuda:0"
# seed=42

# # ===== Parameters for parallelization =====
# probe_rank_list=(5 50 100 300 768)
# # Generate 0..12 as an array
# layer_list=($(seq 0 12))

# # ===== Check GNU Parallel availability =====
# command -v parallel >/dev/null 2>&1 || {
#   echo "ERROR: GNU parallel not found. Please install it." >&2
#   exit 1
# }

# mkdir -p "${save_dir}" "${save_dir}/vis"

# # ===== Function to run one job =====
# run_one() {
#   local layer="$1"
#   local probe_rank="$2"

#   local probe_path="../../models/structural_probe/${model}/train250_title/comp${probe_rank}_L${layer}/probe_${probe_type}_layer${layer}_rank${probe_rank}.pt"
#   local vis_path="${save_dir}/vis/vis_layer${layer}_rank${probe_rank}.html"

#   # Optional: print job info
#   echo "[RUN] layer=${layer} rank=${probe_rank}"

#   uv run python3 eval_probe.py \
#     --embedding_dir "${embedding_dir}" \
#     --save_dir "${save_dir}" \
#     --layer "${layer}" \
#     --probe_type "${probe_type}" \
#     --probe_rank "${probe_rank}" \
#     --probe_path "${probe_path}" \
#     --vis_path "${vis_path}" \
#     --device "${device}" \
#     --seed "${seed}"
# }
# export -f run_one
# export model embedding_dir probe_type save_dir device seed

# # ===== Run all combinations in parallel =====
# # -j 0 means "use all available CPU cores". Adjust if needed (e.g., -j 4).
# parallel --will-cite -j 0 run_one {1} {2} ::: "${layer_list[@]}" ::: "${probe_rank_list[@]}"

# # To check the commands before execution (dry run):
# # parallel --will-cite --dry-run run_one {1} {2} ::: "${layer_list[@]}" ::: "${probe_rank_list[@]}"











# #!/usr/bin/env bash
# set -eu


# # taxonomy title

# text_mode="title"


# model_name="meta-llama/Meta-Llama-3-8B"
# probe_rank=300
# layer=6
# device="cuda:0"
# seed=41

# probe_path="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/probe/probe_layer${layer}_rank${probe_rank}.pt"

# embedding_dir="/work03/masaki/model/taxonomy/embeddings/${model_name}/${text_mode}"
# save_dir="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/"

# poetry run python3 eval_probe.py \
#     --probe_path    "${probe_path}" \
#     --embedding_dir "${embedding_dir}" \
#     --layer         "${layer}" \
#     --device        "${device}" \
#     --probe_rank    "${probe_rank}" \
#     --seed          "${seed}" \
#     --save_dir      "${save_dir}"


# # taxonomy_icl_nl
# # text_mode="icl_NL"


# # model_name="meta-llama/Meta-Llama-3-8B"
# # probe_rank=300
# # layer=6
# # device="cuda:0"
# # seed=41
# # icl_repeat_num=1

# # probe_path="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/Repeat_${icl_repeat_num}/probe/probe_layer${layer}_rank${probe_rank}.pt"

# # embedding_dir="/work03/masaki/model/taxonomy/embeddings/${model_name}/${text_mode}"
# # # save_dir="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/Repeat_${icl_repeat_num}"

# # poetry run python3 eval_probe.py \
# #     --probe_path    "${probe_path}" \
# #     --embedding_dir "${embedding_dir}" \
# #     --layer         "${layer}" \
# #     --device        "${device}" \
# #     --probe_rank    "${probe_rank}" \
# #     --seed          "${seed}" \
# #     --icl_repeat_num "${icl_repeat_num}"
