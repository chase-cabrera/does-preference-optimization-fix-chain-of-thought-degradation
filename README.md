# Does Preference Optimization Fix Chain of Thought Degradation?

CS652 Spring 2026 | Chase Cabrera | San Diego State University

## Overview

This project investigates whether Direct Preference Optimization (DPO) can fix the complexity scaling degradation observed in Chain of Thought (CoT) reasoning. We test four interventions across two models (LLaMA 2 7B and Llama 3.1 8B):

1. Multi domain DPO (preference pairs from PEMDAS and CoinFlip)
2. Step level DPO (per step symbolic ground truth pairs)
3. Incorrect CoT DPO (corrupted reasoning traces with correct answers)
4. Symbolic verification at inference

## Key Findings

- Multi domain DPO improves PEMDAS CoT accuracy on both models (25.5% to 32.4% on LLaMA 2, 32.1% to 34.8% on Llama 3.1)
- Incorrect CoT DPO performs comparably to correct CoT DPO on both models, proving improvement is pattern matching
- Symbolic verification achieves 100% accuracy on all variants of both models
- Step level DPO produces too few pairs on both models (8 on LLaMA 2, 15 on Llama 3.1)
- A stronger base model improves the ceiling but does not change the nature of the improvement

## Structure

```
paper/              LaTeX source and figures
code/               All experiment scripts
benchmarks/         PEMDAS, CoinFlip benchmark instances
data/               LLaMA 2 7B preference pairs
data_llama31/       Llama 3.1 8B preference pairs
results/            LLaMA 2 7B evaluation results
results_llama31/    Llama 3.1 8B evaluation results
```

## Reproducing Results

### Setup

```bash
pip install -r code/requirements.txt
```

Download models from Hugging Face:
- LLaMA 2 7B to `models/Llama-2-7b-hf/`
- Llama 3.1 8B to `models/Llama-3.1-8B/` (requires access approval)

### Step 1: Generate preference data

```bash
cd code

# LLaMA 2 7B
python3 generate_multidomain_preferences.py --base-model ../models/Llama-2-7b-hf --output-dir ../data
python3 generate_step_preferences.py --base-model ../models/Llama-2-7b-hf --output ../data/step_preferences.json
python3 generate_corrupted_preferences.py --input ../data/pemdas_preferences.json

# Llama 3.1 8B
python3 generate_multidomain_preferences.py --base-model ../models/Llama-3.1-8B --output-dir ../data_llama31
python3 generate_step_preferences.py --base-model ../models/Llama-3.1-8B --output ../data_llama31/step_preferences.json
python3 generate_corrupted_preferences.py --input ../data_llama31/pemdas_preferences.json --output ../data_llama31/pemdas_preferences_corrupted.json
```

### Step 2: Train DPO variants

```bash
# LLaMA 2 7B
python3 train_dpo.py --base-model ../models/Llama-2-7b-hf --dataset ../data/multidomain_preferences.json --output-dir ../results/dpo_multidomain
python3 train_dpo.py --base-model ../models/Llama-2-7b-hf --dataset ../data/pemdas_preferences_corrupted.json --output-dir ../results/dpo_corrupted

# Llama 3.1 8B
python3 train_dpo.py --base-model ../models/Llama-3.1-8B --dataset ../data_llama31/multidomain_preferences.json --output-dir ../results_llama31/dpo_multidomain
python3 train_dpo.py --base-model ../models/Llama-3.1-8B --dataset ../data_llama31/pemdas_preferences_corrupted.json --output-dir ../results_llama31/dpo_corrupted
```

### Step 3: Evaluate

```bash
# LLaMA 2 7B
python3 evaluate_benchmarks.py --base-model ../models/Llama-2-7b-hf --model-tag base --output-dir ../results
python3 evaluate_benchmarks.py --base-model ../models/Llama-2-7b-hf --model-tag dpo_multidomain --lora ../results/dpo_multidomain/final_checkpoint --output-dir ../results
python3 evaluate_benchmarks.py --base-model ../models/Llama-2-7b-hf --model-tag dpo_corrupted --lora ../results/dpo_corrupted/final_checkpoint --output-dir ../results

# Llama 3.1 8B
python3 evaluate_benchmarks.py --base-model ../models/Llama-3.1-8B --model-tag llama31_base --output-dir ../results_llama31
python3 evaluate_benchmarks.py --base-model ../models/Llama-3.1-8B --model-tag llama31_dpo_multidomain --lora ../results_llama31/dpo_multidomain/final_checkpoint --output-dir ../results_llama31
python3 evaluate_benchmarks.py --base-model ../models/Llama-3.1-8B --model-tag llama31_dpo_corrupted --lora ../results_llama31/dpo_corrupted/final_checkpoint --output-dir ../results_llama31
```

### Step 4: Generate figures

```bash
python3 analyze_results.py --plot
```

## Hardware

All experiments ran on Apple Silicon M5 Max with 128 GB unified memory using MPS backend.

## References

- Stechly et al. "Chain of Thoughtlessness?" NeurIPS 2024
- Zhang et al. "Chain of Preference Optimization" NeurIPS 2024
- Zhao et al. "Is CoT Reasoning a Mirage?" arXiv 2024
- Rafailov et al. "Direct Preference Optimization" NeurIPS 2023
- Grattafiori et al. "The Llama 3 Herd of Models" arXiv 2024
