# Does Preference Optimization Fix Chain-of-Thought Degradation?

CS652 Spring 2026 | Chase Cabrera | San Diego State University

## Overview

This project asks whether Direct Preference Optimization (DPO) can fix the complexity-scaling degradation observed in Chain-of-Thought (CoT) reasoning, and whether any improvement reflects reasoning or pattern matching. We evaluate five interventions across two open bases (LLaMA-2-7B and Llama 3.1-8B) on PEMDAS and CoinFlip:

1. **Step-Level DPO** — per-step symbolically-verified preference pairs.
2. **Multi-Domain DPO** — joint PEMDAS+CoinFlip naturalistic pairs (with a PEMDAS-only ablation).
3. **Synthetic-CoT vs. Incorrect-CoT DPO** — matched diagnostic with identical final answers but correct vs. scrambled intermediates.
4. **Canonical CPO** — Tree-of-Thought search at training time scored by the model's own self-evaluation (Zhang et al., NeurIPS 2024).
5. **Symbolic verify-and-correct at inference** — post-hoc check on the model's own CoT.

## Key Findings

- Multi-Domain DPO modestly improves PEMDAS CoT accuracy on both bases (25.5→32.4% on LLaMA-2-7B, 32.1→34.8% on Llama 3.1-8B), but its effect on the complexity curve splits sign: degradation $\Delta$ halves on LLaMA-2-7B and roughly doubles on Llama 3.1-8B.
- The matched Synthetic-vs-Incorrect-CoT diagnostic shows a base-model interaction: a 6.5 pp gap on LLaMA-2-7B but only 1.3 pp on Llama 3.1-8B, consistent with the stronger base learning format rather than reasoning.
- Step-Level DPO (verifier-graded) yields only 8/15 pairs on the two bases and leaves the LoRA adapter near identity; canonical CPO with ToT search yields 353/145 pairs and produces only a small movement on 7B and bit-identical accuracy and degradation on 8B. The bottleneck is how many parseable step candidates the base can generate, not the choice of scoring function.
- Symbolic verify-and-correct over the model's own CoT recovers **zero** additional accuracy on every (model, variant) combination: 94–98% of responses contain no parseable arithmetic line, and where a step is emitted it is already correct. The gap to a perfect solver is a generation bottleneck, not an execution one.
- Trace correctness stays below 5% across all variants of both models.

## Structure

```
paper/              LaTeX source and figures
code/               All experiment scripts
benchmarks/         PEMDAS, CoinFlip benchmark instances
data/               LLaMA-2-7B preference pairs
data_llama31/       Llama 3.1-8B preference pairs
results/            LLaMA-2-7B evaluation results + DPO checkpoints
results_llama31/    Llama 3.1-8B evaluation results + DPO checkpoints
```

## Reproducing Results

### Setup

```bash
pip install -r code/requirements.txt
```

Download models from Hugging Face:
- LLaMA-2-7B to `models/Llama-2-7b-hf/`
- Llama 3.1-8B to `models/Llama-3.1-8B/` (requires access approval)

The remaining steps assume `cd code` and run each block on whichever base you are reproducing. Replace `../models/<base>` and `../data{,_llama31}` / `../results{,_llama31}` accordingly.

### Step 1: Generate preference data

```bash
# Naturalistic full-trace pairs (PEMDAS, CoinFlip, and the joint Multi-Domain set)
python3 generate_multidomain_preferences.py --base-model ../models/Llama-2-7b-hf --output-dir ../data

# Per-step symbolically-verified pairs
python3 generate_step_preferences.py --base-model ../models/Llama-2-7b-hf --output ../data/step_preferences.json

# Synthetic chosen traces (left-to-right reduction) and matched corrupted variant
python3 generate_pemdas_synthetic_preferences.py --output ../data/pemdas_synthetic_preferences.json
python3 generate_corrupted_preferences.py --input ../data/pemdas_synthetic_preferences.json --output ../data/pemdas_synthetic_preferences_corrupted_v2.json

# Canonical CPO: ToT expansion + model self-evaluation as value function
python3 generate_cpo_preferences.py --base-model ../models/Llama-2-7b-hf --output ../data/pemdas_cpo_preferences.json
```

For Llama 3.1-8B, swap the base-model path and write to `../data_llama31/`. The 8B CPO run uses two seeds (e.g. `--seed 42` and `--seed 200`) to compensate for its lower per-instance yield; merge the two JSONs before training.

### Step 2: Train DPO variants

```bash
python3 train_dpo.py --base-model ../models/Llama-2-7b-hf --dataset ../data/multidomain_preferences.json     --output-dir ../results/dpo_multidomain
python3 train_dpo.py --base-model ../models/Llama-2-7b-hf --dataset ../data/pemdas_preferences.json          --output-dir ../results/dpo_pemdas_only
python3 train_dpo.py --base-model ../models/Llama-2-7b-hf --dataset ../data/step_preferences.json            --output-dir ../results/dpo_steplevel
python3 train_dpo.py --base-model ../models/Llama-2-7b-hf --dataset ../data/pemdas_synthetic_preferences.json --output-dir ../results/dpo_synthetic
python3 train_dpo.py --base-model ../models/Llama-2-7b-hf --dataset ../data/pemdas_synthetic_preferences_corrupted_v2.json --output-dir ../results/dpo_corrupted_v2
python3 train_dpo.py --base-model ../models/Llama-2-7b-hf --dataset ../data/pemdas_cpo_preferences.json      --output-dir ../results/dpo_cpo
```

Mirror the same six commands against `../models/Llama-3.1-8B`, `../data_llama31/`, and `../results_llama31/` for the 8B sweep.

### Step 3: Evaluate

For each variant (and the bare base), evaluate on PEMDAS + CoinFlip:

```bash
# Base
python3 evaluate_benchmarks.py --base-model ../models/Llama-2-7b-hf --model-tag base --output-dir ../results

# DPO variants — repeat for {dpo_multidomain, dpo_pemdas_only, dpo_steplevel,
#                            dpo_synthetic, dpo_corrupted_v2, dpo_cpo}
python3 evaluate_benchmarks.py --base-model ../models/Llama-2-7b-hf --model-tag dpo_multidomain \
    --lora ../results/dpo_multidomain/final_checkpoint --output-dir ../results
```

For the 8B sweep, prefix every `--model-tag` with `llama31_` and write to `../results_llama31/`.

### Step 4: Symbolic post-hoc verify-and-correct

This runs in seconds per file because no neural inference is required:

```bash
python3 symbolic_verify_posthoc.py \
  --inputs ../results/pemdas_*.json ../results_llama31/pemdas_llama31_*.json \
  --out ../results/symbolic_verify_posthoc_summary.json
```

### Step 5: Figures and tables

```bash
python3 analyze_results.py --plot --latex
```

This writes the degradation curves (`paper/figures/pemdas_*.png`) and the trace-correctness breakdown (`paper/figures/faithfulness_breakdown.png`), and prints LaTeX rows for Table 1.

## Hardware

All experiments ran on Apple Silicon M5 Max with 128 GB unified memory using the MPS backend. DPO trained in fp32; evaluation generation in fp16, greedy decoding, max 256 new tokens, single seed (42 for the 90/10 train/eval split).

## References

- Stechly, Valmeekam, Kambhampati. *Chain of Thoughtlessness? An Analysis of CoT in Planning.* NeurIPS 2024.
- Zhang et al. *Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs.* NeurIPS 2024.
- Zhao et al. *Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens.* arXiv:2508.01191, 2025.
- Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
- Lanham et al. *Measuring Faithfulness in Chain-of-Thought Reasoning.* arXiv:2307.13702, 2023.
- Touvron et al. *LLaMA 2: Open Foundation and Fine-Tuned Chat Models.* arXiv:2307.09288, 2023.
- Grattafiori et al. *The Llama 3 Herd of Models.* arXiv:2407.21783, 2024.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
