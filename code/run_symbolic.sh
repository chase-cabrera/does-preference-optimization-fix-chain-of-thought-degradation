#!/bin/bash
# =============================================================================
# Run symbolic_verifier.py end-to-end across both base models and all five
# variants (Base + 4 DPO variants), in BOTH verify-only and verify-and-correct
# modes. Replaces the prior post-hoc summary that hard-coded PEMDAS symbolic
# accuracy to 100%.
#
# Domains: pemdas (verify-and-correct is the interesting case)
#          coinflip (sym_acc is 100% by construction; reported as sanity check)
#
# Usage (from this directory, i.e. code/):
#   ./run_symbolic.sh smoke                  # 5-instance dry run on base 7B
#   ./run_symbolic.sh one base 7b correct    # one variant, one model, one mode
#   ./run_symbolic.sh all 7b                 # all 5 variants on LLaMA-2-7B
#   ./run_symbolic.sh all 31                 # all 5 variants on Llama-3.1-8B
#   ./run_symbolic.sh all both               # everything (sequential, ~6h)
#
# Wall-clock estimates (Apple Silicon M-series, 128 GB):
#   one variant on 7B:    ~30-40 min  (290 PEMDAS + 280 CoinFlip @ 256 tok)
#   one variant on 31:    ~20-30 min  (faster tokenizer)
#   all 5 variants 7B:    ~3 h
#   all 5 variants 31:    ~2 h
# =============================================================================
set -e
cd "$(dirname "$0")"

MODEL_7B="../Project Topic/CPO/model/Llama-2-7b-hf"
MODEL_31="../models/Llama-3.1-8B"

# Variant -> (lora_path_7b, lora_path_31, tag_suffix). Use "" for no LoRA.
LORA_BASE_7B=""
LORA_MULTI_7B="../results/dpo_multidomain/final_checkpoint"
LORA_CORRUPTED_7B="../results/dpo_corrupted/final_checkpoint"
LORA_STEPLEVEL_7B="../results/dpo_steplevel/final_checkpoint"
LORA_SYNTHETIC_7B="../results/dpo_synthetic/final_checkpoint"
LORA_CORRUPTED_V2_7B="../results/dpo_corrupted_v2/final_checkpoint"
LORA_PEMDAS_ONLY_7B="../results/dpo_pemdas_only/final_checkpoint"

LORA_BASE_31=""
LORA_MULTI_31="../results_llama31/dpo_multidomain/final_checkpoint"
LORA_CORRUPTED_31="../results_llama31/dpo_corrupted/final_checkpoint"
LORA_STEPLEVEL_31="../results_llama31/dpo_steplevel/final_checkpoint"
LORA_SYNTHETIC_31="../results_llama31/dpo_synthetic/final_checkpoint"
LORA_CORRUPTED_V2_31="../results_llama31/dpo_corrupted_v2/final_checkpoint"
LORA_PEMDAS_ONLY_31="../results_llama31/dpo_pemdas_only/final_checkpoint"

LOGDIR="../logs"
mkdir -p "$LOGDIR"

run_one() {
    # run_one <variant> <model_tag: 7b|31> <mode: only|correct>
    local variant="$1"
    local mtag="$2"
    local mode_short="$3"

    local model resultsdir prefix lora_var
    if [ "$mtag" = "7b" ]; then
        model="$MODEL_7B"
        resultsdir="../results"
        prefix=""
        lora_var="LORA_$(echo "$variant" | tr '[:lower:]' '[:upper:]')_7B"
    else
        model="$MODEL_31"
        resultsdir="../results_llama31"
        prefix="llama31_"
        lora_var="LORA_$(echo "$variant" | tr '[:lower:]' '[:upper:]')_31"
    fi
    local lora="${!lora_var}"

    local mode
    if [ "$mode_short" = "only" ]; then mode="verify-only"; else mode="verify-and-correct"; fi

    local tag
    if [ "$variant" = "base" ]; then
        tag="${prefix}base"
    else
        tag="${prefix}dpo_${variant}"
    fi

    if [ -n "$lora" ] && [ ! -d "$lora" ]; then
        echo "[SKIP] $tag $mode: LoRA dir not found: $lora"
        return 0
    fi

    echo
    echo "=========================================================="
    echo "  $tag  /  $mode"
    echo "=========================================================="
    local lora_arg=""
    [ -n "$lora" ] && lora_arg="--lora $lora"

    python3 symbolic_verifier.py \
        --base-model "$model" \
        --model-tag "$tag" \
        $lora_arg \
        --mode "$mode" \
        --output-dir "$resultsdir" \
        2>&1 | tee "$LOGDIR/sym_${tag}_${mode//-/_}.log"
}

VARIANTS_DEFAULT="base multi corrupted steplevel synthetic corrupted_v2 pemdas_only"

run_all_for_model() {
    local mtag="$1"
    for variant in $VARIANTS_DEFAULT; do
        for mode in only correct; do
            run_one "$variant" "$mtag" "$mode" || true
        done
    done
}

case "${1:-}" in
    smoke)
        echo "Smoke test: base 7B, verify-and-correct, PEMDAS only, first 5 instances."
        echo "(This still loads the full model; expect ~3 min.)"
        python3 -c "
import json
with open('../benchmarks/pemdas.json') as f: b = json.load(f)
b['instances'] = b['instances'][:5]
import os; os.makedirs('/tmp/sym_smoke', exist_ok=True)
with open('/tmp/sym_smoke/pemdas.json', 'w') as f: json.dump(b, f)
"
        python3 symbolic_verifier.py \
            --base-model "$MODEL_7B" \
            --model-tag base_smoke \
            --benchmarks-dir /tmp/sym_smoke \
            --domains pemdas \
            --mode verify-and-correct \
            --output-dir /tmp/sym_smoke
        ;;
    one)
        run_one "${2:?variant}" "${3:?model 7b|31}" "${4:?mode only|correct}"
        ;;
    all)
        case "${2:?model 7b|31|both}" in
            7b)   run_all_for_model 7b ;;
            31)   run_all_for_model 31 ;;
            both) run_all_for_model 7b; run_all_for_model 31 ;;
            *)    echo "model must be 7b, 31, or both"; exit 1 ;;
        esac
        ;;
    *)
        echo "Usage: $0 {smoke|one <variant> <7b|31> <only|correct>|all <7b|31|both>}"
        echo
        echo "Variants: $VARIANTS_DEFAULT"
        exit 1
        ;;
esac
