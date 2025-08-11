#!/usr/bin/env bash
# tda_persim.sh
# Run tda_persim.py over:
#  - gpt2 all layers
#  - meta-llama/Meta-Llama-3-8B all layers
#  - fasttext (layer 0)
#  - random_emb (layer 0)

set -euo pipefail

# ---------- defaults ----------
INPUT="../input/taxonomy_person_test50.jsonl"
OUT_ROOT="../output/persistent_diagrams"
METRIC="euclidean"        # or: cosine
MAXDIM=1               # usually 1 (H0/H1)
THRESH="auto"          # or numeric (e.g., 1.5)
DEVICE="cuda:0"
TEMPLATE="entity_only"

# Layer counts (override if needed): export GPT2_LAYERS=12, LLAMA3_LAYERS=32
GPT2_LAYERS="${GPT2_LAYERS:-12}"
LLAMA3_LAYERS="${LLAMA3_LAYERS:-32}"

# Allow selecting subset: export ONLY="gpt2|llama|fasttext|random"
ONLY="${ONLY:-}"

# ---------- parse args ----------
usage () {
  cat <<EOF
Usage:
  bash tda_persim.sh --input data/gold.jsonl --out out/tda_persim_runs \\
                     [--metric cosine|euclidean] [--maxdim 1] [--thresh auto|NUM] \\
                     [--device cuda|cpu] [--template entity_only]

Env overrides:
  GPT2_LAYERS (default ${GPT2_LAYERS})
  LLAMA3_LAYERS (default ${LLAMA3_LAYERS})
  ONLY=gpt2|llama|fasttext|random  # run a subset

Requires:
  tda_persim.py in the current directory
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)    INPUT="$2"; shift 2;;
    --out)      OUT_ROOT="$2"; shift 2;;
    --metric)   METRIC="$2"; shift 2;;
    --maxdim)   MAXDIM="$2"; shift 2;;
    --thresh)   THRESH="$2"; shift 2;;
    --device)   DEVICE="$2"; shift 2;;
    --template) TEMPLATE="$2"; shift 2;;
    -h|--help)  usage; exit 0;;
    *) echo "[ERR] Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -z "$INPUT" || -z "$OUT_ROOT" ]] && { usage; exit 1; }

# ---------- checks ----------
[[ -f "tda_persim.py" ]] || { echo "[ERR] tda_persim.py not found."; exit 1; }
uv run python3 - <<'PY' >/dev/null 2>&1 || { echo "[ERR] Please: pip install ripser persim matplotlib"; exit 1; }
import importlib; importlib.import_module("ripser"); importlib.import_module("persim"); importlib.import_module("matplotlib")
PY

# ---------- helpers ----------
run_one () {
  local MODEL="$1"
  local LAYER="$2"     # integer
  local METHOD="$3"    # "last_token" or "average"
  local OUTDIR="$4"

  mkdir -p "$OUTDIR"
  echo "[RUN] model=${MODEL} layer=${LAYER} -> ${OUTDIR}"

  uv run python3 tda_persim.py \
    --input "$INPUT" \
    --out_dir "$OUTDIR" \
    --model "$MODEL" \
    --method "$METHOD" \
    --layer "$LAYER" \
    --metric "$METRIC" \
    --maxdim "$MAXDIM" \
    --thresh "$THRESH" \
    >/dev/null
}

# ---------- gpt2 all layers ----------
if [[ -z "$ONLY" || "$ONLY" == "gpt2" ]]; then
  for ((L=0; L<GPT2_LAYERS; L++)); do
    run_one "gpt2" "$L" "last_token" "${OUT_ROOT}/gpt2/L${L}"
  done
fi

# ---------- llama3 8B all layers ----------
if [[ -z "$ONLY" || "$ONLY" == "llama" ]]; then
  for ((L=0; L<LLAMA3_LAYERS; L++)); do
    run_one "meta-llama/Meta-Llama-3-8B" "$L" "last_token" "${OUT_ROOT}/llama3-8b/L${L}"
  done
fi

# ---------- fastText (no layers; use 0) ----------
if [[ -z "$ONLY" || "$ONLY" == "fasttext" ]]; then
  run_one "fasttext" 0 "average" "${OUT_ROOT}/fasttext/L0"
fi

# ---------- random embeddings (no layers; use 0) ----------
if [[ -z "$ONLY" || "$ONLY" == "random" ]]; then
  run_one "random_emb" 0 "average" "${OUT_ROOT}/random_emb/L0"
fi

echo "[DONE] Results under: ${OUT_ROOT}"
