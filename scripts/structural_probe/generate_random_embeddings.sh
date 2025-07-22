
# partonomyのランダム埋め込みを生成するスクリプト
#!/bin/bash

# 共通パラメータ
model_name="random_emb"  
data_type="test_15"
text_mode="title"
random_seed=42
data_path="/work03/masaki/data/zelda/partonomy/${data_type}.jsonl"
device="cuda:0"

# 実行する次元数のリスト
embedding_dim_list=(1 2 5 10 30 40 50 60 80 100 200 300 500 768 1000 2000 3000)

# 並列実行用の関数を定義
run_embedding_generation() {
    local embedding_dim=$1
    local output_dir="/work03/masaki/model/partonomy/embeddings/${data_type}/${model_name}/dim_${embedding_dim}"
    
    echo "Starting embedding generation for dim=${embedding_dim}"
    
    poetry run python3 generate_random_embeddings.py \
        --data_path "$data_path" \
        --output_dir "$output_dir" \
        --embedding_dim "$embedding_dim" \
        --text_mode "$text_mode" \
        --random_seed "$random_seed" \
        --device "$device" \
        --is_verbose
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed dim=${embedding_dim}"
    else
        echo "✗ Failed for dim=${embedding_dim}"
        return 1
    fi
}

# 関数をexportして並列実行で使えるようにする
export -f run_embedding_generation
export data_type text_mode random_seed data_path device

echo "Starting parallel execution for ${#embedding_dim_list[@]} different embedding dimensions"
echo "Embedding dimensions: ${embedding_dim_list[*]}"
echo "Data type: $data_type"
echo "Text mode: $text_mode"
echo "Random seed: $random_seed"
echo "Device: $device"
echo "Data path: $data_path"
echo ""

# GNU parallelで並列実行
# -j 0: 利用可能なCPUコア数を自動で使用
# --progress: 進行状況を表示
# --joblog: ジョブログを出力
# --resume: 失敗したジョブのみを再実行（オプション）
parallel -j 0 --progress --joblog "parallel_log_${data_type}_$(date +%Y%m%d_%H%M%S).log" \
    run_embedding_generation ::: "${embedding_dim_list[@]}"

echo ""
echo "All parallel jobs completed!"
echo "Check the log file for detailed results."



# taxonomyのランダム埋め込みを生成するスクリプト
