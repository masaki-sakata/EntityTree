#!/usr/bin/env bash
set -eu

########################
#  固定パラメータ
########################
text_mode="icl_NL"

# 並列に回したいリスト
model_list=(
  "meta-llama/Meta-Llama-3-8B"
)

probe_rank_list=(300)
icl_repeat_num_list=(1 4 8)
layer_max=32

epochs=1000
device="cuda:0"
batch_size=64
seed=41
n_jobs=3

########################
#  parallel で一気に実行
########################
export epochs device batch_size seed text_mode   # 共有パラメータ

parallel -j "${n_jobs}" --progress --eta --line-buffer --verbose '
  model_name={1}
  probe_rank={2}
  layer={3}
  icl_repeat_num={4}
  batch_size="${batch_size}"

  embedding_dir="/work03/masaki/model/taxonomy/embeddings/${model_name}/${text_mode}"
  save_dir="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/Repeat_${icl_repeat_num}"

  poetry run python3 train_probe_icl.py \
      --embedding_dir "${embedding_dir}" \
      --save_dir      "${save_dir}" \
      --layer         "${layer}" \
      --epochs        "${epochs}" \
      --device        "${device}" \
      --batch_size    "${batch_size}" \
      --probe_rank    "${probe_rank}" \
      --seed          "${seed}" \
      --icl_repeat_num "${icl_repeat_num}"
' ::: "${model_list[@]}" \
  ::: "${probe_rank_list[@]}" \
  ::: $(seq 0 "${layer_max}") \
  ::: "${icl_repeat_num_list[@]}"