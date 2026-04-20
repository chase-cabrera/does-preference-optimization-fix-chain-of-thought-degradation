#!/bin/bash
# =============================================================================
# Train + evaluate the PEMDAS-only naturalistic DPO ablation.
#
# Purpose: addresses the reviewer concern that Multi-Domain DPO is not a clean
# test of "data coverage" because the dataset is dominated by CoinFlip pairs.
# This ablation removes that confound by training on PEMDAS preferences only,
# matching as closely as possible the Multi-Domain DPO recipe (same trainer,
# same hparams, same eval).
#
# Data sources (already on disk):
#   ../data/pemdas_preferences.json           143 pairs (LLaMA-2-7B)
#   ../data_llama31/pemdas_preferences.json   225 pairs (Llama-3.1-8B)
#
# Usage (from this directory, i.e. code/):
#   ./run_pemdas_only.sh train 7b
#   ./run_pemdas_only.sh train 31
#   ./run_pemdas_only.sh eval  7b
#   ./run_pemdas_only.sh eval  31
#   ./run_pemdas_only.sh all   7b      # train + eval
#   ./run_pemdas_only.sh all   31
#
# Wall-clock estimates (Apple Silicon M-series, 128 GB):
#   train 7b:   ~25-35 min  (143 pairs, LoRA r=8, 3 epochs, fp32)
#   train 31:   ~20-30 min  (225 pairs)
#   eval  7b:   ~2 h 45     (290 PEMDAS + 280 CoinFlip, two strategies each)
#   eval  31:   ~1 h 30
# =============================================================================
set -e
cd "$(dirname "$0")"

MODEL_7B="../Project Topic/CPO/model/Llama-2-7b-hf"
MODEL_31="../models/Llama-3.1-8B"

DATA_7B="../data/pemdas_preferences.json"
DATA_31="../data_llama31/pemdas_preferences.json"

OUT_7B="../results/dpo_pemdas_only"
OUT_31="../results_llama31/dpo_pemdas_only"

LOGDIR="../logs"
mkdir -p "$LOGDIR"

train_one() {
    local mtag="$1"
    local model dataset outdir
    if [ "$mtag" = "7b" ]; then
        model="$MODEL_7B"; dataset="$DATA_7B"; outdir="$OUT_7B"
    else
        model="$MODEL_31"; dataset="$DATA_31"; outdir="$OUT_31"
    fi

    if [ ! -f "$dataset" ]; then
        echo "ERROR: $dataset not found." >&2; exit 1
    fi

    echo "=========================================================="
    echo "  Train PEMDAS-only DPO ($mtag)"
    echo "  base model: $model"
    echo "  dataset:    $dataset"
    echo "  output:     $outdir"
    echo "=========================================================="
    python3 train_dpo.py \
        --base-model "$model" \
        --dataset "$dataset" \
        --output-dir "$outdir" \
        2>&1 | tee "$LOGDIR/train_${mtag}_pemdas_only.log"
}

eval_one() {
    local mtag="$1"
    local model resultsdir prefix outdir
    if [ "$mtag" = "7b" ]; then
        model="$MODEL_7B"; resultsdir="../results"; prefix=""; outdir="$OUT_7B"
    else
        model="$MODEL_31"; resultsdir="../results_llama31"; prefix="llama31_"; outdir="$OUT_31"
    fi

    local final="$outdir/final_checkpoint"
    if [ ! -d "$final" ]; then
        echo "ERROR: $final not found. Run 'train $mtag' first." >&2; exit 1
    fi

    echo "=========================================================="
    echo "  Evaluate PEMDAS-only DPO ($mtag)"
    echo "  LoRA: $final"
    echo "=========================================================="
    python3 evaluate_benchmarks.py \
        --base-model "$model" \
        --model-tag "${prefix}dpo_pemdas_only" \
        --lora "$final" \
        --output-dir "$resultsdir" \
        2>&1 | tee "$LOGDIR/eval_${mtag}_pemdas_only.log"
}

case "${1:-}" in
    train) train_one "${2:?model tag 7b|31}" ;;
    eval)  eval_one  "${2:?model tag 7b|31}" ;;
    all)
        train_one "${2:?model tag 7b|31}"
        eval_one  "$2"
        ;;
    *)
        echo "Usage: $0 {train|eval|all} [7b|31]"
        exit 1
        ;;
esac
