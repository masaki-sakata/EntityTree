#!/usr/bin/env bash
set -euo pipefail

# ====== 共通設定 ======
# model="gpt2"
model="meta-llama/Meta-Llama-3-8B"
embedding_dir="/work03/masaki/model/hierarchical-repr/${model}/train250/title"
probe_type="distance"
epochs=1000
lr=1e-3
device="cuda:1"
parallel_jobs=1

# 探索するハイパラ
# probe_rank_list=(5 50 100 300 768)
probe_rank_list=(768)

# layer_list=({0..12})
# layer_list=({0..32})
layer_list=($(seq 0 4 32))


# ====== 環境変数として export（GNU parallel に渡す用）======
export model embedding_dir probe_type epochs lr device

# 1ジョブの実行関数
run_job() {
  local pr="$1"
  local la="$2"
  local SAVE_DIR="../../models/structural_probe/${model}/train250_title/n_comp${pr}/layer${la}"
  mkdir -p "$SAVE_DIR"
  echo "[INFO] Starting probe_rank=${pr} layer=${la} -> ${SAVE_DIR}"

  uv run python3 train_probe.py \
    --embedding_dir "$embedding_dir" \
    --layer "$la" \
    --probe_type "$probe_type" \
    --probe_rank "$pr" \
    --epochs "$epochs" \
    --lr "$lr" \
    --device "$device" \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee "$SAVE_DIR/train.log"
}

export -f run_job

# (probe_rank -> layer) の順で組を生成
generate_pairs() {
  for pr in "${probe_rank_list[@]}"; do
    for la in "${layer_list[@]}"; do
      echo "$pr $la"
    done
  done
}

# ====== 並列で実行 ======
generate_pairs | parallel -j "$parallel_jobs" --lb --colsep ' ' run_job {1} {2}




# # #!/usr/bin/env bash
# set -eu  # 事故防止

# ########################
# #  固定パラメータ
# # ########################
# text_mode="title"

# # 並列に回したいリスト
# model_list=(
#   # "meta-llama/Meta-Llama-3-8B"
#   "gpt2"
# )

# # probe_rank_list=(1 2 5 10 30 40 50 60 80 100 200 300 500 768 1000 2000 3000)
# # probe_rank_list=(1 2 5 10 30 40 50 60 80 100 200 300 500 768)
# probe_rank_list=(300)
# layer_max=12

# epochs=1000
# device="cuda:0"
# batch_size=256
# seed=41
# n_jobs=3                 # 並列数
# lr=1e-3

# ########################
# #  parallel で一気に実行
# ########### #############
# # 先に共有パラメータを export しておくと {= =} 展開の中で普通に参照できる
# export epochs device batch_size seed text_mode lr

# parallel -j "${n_jobs}" --progress --eta --line-buffer --verbose \
#   '
#   model_name={1}
#   probe_rank={2}
#   layer={3}
#   batch_size="${batch_size}"
#   lr="${lr}"

#   embedding_dir="/work03/masaki/model/taxonomy/embeddings/${model_name}/${text_mode}"
#   save_dir="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/lr${lr}_epochs${epochs}_batch${batch_size}"

#   poetry run python3 train_probe.py \
#       --embedding_dir "${embedding_dir}" \
#       --save_dir      "${save_dir}" \
#       --layer         "${layer}" \
#       --epochs        "${epochs}" \
#       --device        "${device}" \
#       --batch_size    "${batch_size}" \
#       --probe_rank    "${probe_rank}" \
#       --lr            "${lr}" \
#       --seed          "${seed}"
#   ' ::: "${model_list[@]}" \
#     ::: "${probe_rank_list[@]}" \
#     ::: $(seq 0 "${layer_max}")



# # # # fastText

# # #!/usr/bin/env bash
# # set -eu  # 事故防止
# # ########################
# # #  固定パラメータ
# # ########################
# # text_mode="title"

# # # 並列に回したいリスト
# # model_name="fasttext"
# # layer=0                    
# # # probe_rank_list=(1 2 5 10 30 40 50 60 80 100 200 300)
# # probe_rank_list=(300)

# # epochs=1000
# # lr=1e-3
# # batch_size=64
# # seed=41
# # n_jobs=1                    # 並列数

# # ########################
# # #  parallel で一気に実行
# # ########################
# # # 先に共有パラメータを export しておくと {= =} 展開の中で普通に参照できる
# # export data_type epochs batch_size seed model_name text_mode layer lr

# # parallel -j "${n_jobs}" --progress --eta --line-buffer --verbose \
# #   '
# #   probe_rank={1}
# #   batch_size="${batch_size}"
# #   lr="${lr}"

# #   embedding_dir="/work03/masaki/model/taxonomy/embeddings/${model_name}/${text_mode}"
# #   save_dir="/work03/masaki/result/taxonomy/${model_name}/${text_mode}/lr${lr}_epochs${epochs}_batch${batch_size}"

# #   poetry run python3 train_probe.py \
# #       --embedding_dir "${embedding_dir}" \
# #       --save_dir      "${save_dir}" \
# #       --layer         "${layer}" \
# #       --epochs        "${epochs}" \
# #       --batch_size    "${batch_size}" \
# #       --probe_rank    "${probe_rank}" \
# #       --seed          "${seed}"
# #       --lr            "${lr}"
# #   ' ::: "${probe_rank_list[@]}" 




# # # random emb

# # #!/usr/bin/env bash
# # set -eu  # 事故防止

# # ########################
# # #  固定パラメータ
# # ########################

# # text_mode="title"

# # # 並列に回したいリスト
# # layer=0
# # model_name="random_emb"
# # probe_rank_list=(1 2 5 10 30 40 50 60 80 100 200 300 500 768 2000 3000)

# # epochs=1000
# # device="cuda:0"
# # batch_size=64
# # seed=41
# # n_jobs=2                    # 並列数
# # lr=1e-3

# # ########################
# # #  parallel で一気に実行
# # ########################
# # # 先に共有パラメータを export しておくと {= =} 展開の中で普通に参照できる
# # export epochs batch_size seed model_name text_mode layer device lr

# # parallel -j "${n_jobs}" --progress --eta --line-buffer --verbose \
# #   '
# #   probe_rank={1}

# #   embedding_dir="/work03/masaki/model/taxonomy/embeddings/${model_name}/dim_${probe_rank}/${text_mode}"
# #   save_dir="/work03/masaki/result/taxonomy/${model_name}/dim_${probe_rank}/${text_mode}/lr${lr}_epochs${epochs}_batch${batch_size}"

# #   poetry run python3 train_probe.py \
# #       --embedding_dir "${embedding_dir}" \
# #       --save_dir      "${save_dir}" \
# #       --layer         "${layer}" \
# #       --epochs        "${epochs}" \
# #       --device        "${device}" \
# #       --batch_size    "${batch_size}" \
# #       --probe_rank    "${probe_rank}" \
# #       --lr            "${lr}" \
# #       --seed          "${seed}"
# #   ' ::: "${probe_rank_list[@]}" 