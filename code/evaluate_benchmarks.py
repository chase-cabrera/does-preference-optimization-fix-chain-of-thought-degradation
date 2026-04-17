"""
Evaluate LLM variants on synthetic benchmarks with step level metrics.

Metrics: overall accuracy, degradation delta, step accuracy, faithfulness.

Usage:
  python3 evaluate_benchmarks.py --base-model models/Llama-2-7b-hf --model-tag base
  python3 evaluate_benchmarks.py --base-model models/Llama-3.1-8B --model-tag llama31_base
  python3 evaluate_benchmarks.py --base-model models/Llama-2-7b-hf --model-tag dpo_multi --lora results/dpo_multidomain/final_checkpoint
"""
import os
import json
import re
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--model-tag", default="base")
    p.add_argument("--benchmarks-dir", default="../benchmarks")
    p.add_argument("--output-dir", default="../results")
    p.add_argument("--domains", nargs="+", default=["pemdas", "coinflip"])
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return p.parse_args()


def load_model(args):
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
        model = model.merge_and_unload()
    model = model.to(args.device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model, tokenizer


def generate(model, tokenizer, prompt, device, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=2,
        )
    new_tokens = output[0][input_ids.shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    del output, input_ids, attention_mask, inputs
    if device == "mps":
        torch.mps.empty_cache()
    return result


def make_cot_prompt(direct_prompt):
    if "[Answer]" in direct_prompt:
        return direct_prompt.replace("\n[Answer]\n", "\nLet's think step by step.\n\n[Answer]\n")
    return direct_prompt + "\nLet's think step by step.\n"


def parse_answer_pemdas(response):
    last_num = None
    for line in response.strip().split("\n"):
        if not line.strip():
            continue
        for match in re.finditer(r"-?\d+\.?\d*", line):
            last_num = match.group()
        if last_num is not None:
            break
    if last_num:
        try:
            return str(int(float(last_num))) if "." in last_num else last_num
        except ValueError:
            return last_num
    return response.strip().split("\n")[0].strip()


def parse_answer_coinflip(response):
    first_line = response.strip().lower().split("\n")[0]
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return "yes" in response.lower()


PARSERS = {"pemdas": parse_answer_pemdas, "coinflip": parse_answer_coinflip}


def check_correct(domain, parsed, ground_truth):
    if domain == "coinflip":
        return parsed == ground_truth
    return str(parsed).strip().lower() == str(ground_truth).strip().lower()


def compute_step_metrics(response):
    steps = []
    for line in response.strip().split("\n"):
        match = re.search(r"(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)", line.strip())
        if match:
            expr = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            try:
                expected = eval(expr)
                actual = float(match.group(4))
                steps.append({"step_correct": abs(expected - actual) < 0.01})
            except (ValueError, ZeroDivisionError, SyntaxError):
                steps.append({"step_correct": False})
    n_correct = sum(1 for s in steps if s["step_correct"])
    return {
        "n_steps_found": len(steps),
        "n_steps_correct": n_correct,
        "step_accuracy": n_correct / len(steps) if steps else 0.0,
        "all_steps_correct": all(s["step_correct"] for s in steps) if steps else False,
    }


def evaluate_domain(model, tokenizer, args, domain, benchmark):
    parser_fn = PARSERS[domain]
    instances = benchmark["instances"]
    results = []

    for strategy in ["direct", "zero-shot-cot"]:
        correct_count = 0
        total = 0
        start = time.time()

        for i, inst in enumerate(instances):
            prompt = inst["prompt"] if strategy == "direct" else make_cot_prompt(inst["prompt"])
            response = generate(model, tokenizer, prompt, args.device, args.max_tokens)
            parsed = parser_fn(response)
            is_correct = check_correct(domain, parsed, inst["ground_truth"])
            correct_count += int(is_correct)
            total += 1

            result = {
                "domain": domain, "strategy": strategy, "model_tag": args.model_tag,
                "instance_id": inst["instance_id"], "steps_to_solve": inst["steps_to_solve"],
                "ground_truth": inst["ground_truth"],
                "parsed_answer": parsed if not isinstance(parsed, bool) else str(parsed).lower(),
                "raw_response": response[:500], "correct": is_correct,
            }

            if strategy == "zero-shot-cot" and domain == "pemdas":
                sm = compute_step_metrics(response)
                result.update(sm)
                result["faithful"] = sm["all_steps_correct"] and is_correct

            results.append(result)

            if (i + 1) % 20 == 0 or i == len(instances) - 1:
                elapsed = time.time() - start
                print(f"  [{strategy:15s}] {i+1:4d}/{len(instances)}  acc={correct_count/total*100:5.1f}%  elapsed={elapsed:.0f}s")

        print(f"  {strategy:15s} FINAL: {correct_count}/{total} = {correct_count/total*100:.1f}%")
    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = load_model(args)
    all_results = []

    for domain in args.domains:
        bench_path = os.path.join(args.benchmarks_dir, f"{domain}.json")
        if not os.path.exists(bench_path):
            continue
        with open(bench_path) as f:
            benchmark = json.load(f)
        print(f"\n{'='*60}\n  {benchmark['label']} ({benchmark['total_instances']} instances)\n{'='*60}")
        results = evaluate_domain(model, tokenizer, args, domain, benchmark)
        all_results.extend(results)

        with open(os.path.join(args.output_dir, f"{domain}_{args.model_tag}.json"), "w") as f:
            json.dump(results, f, indent=2)

    with open(os.path.join(args.output_dir, f"all_{args.model_tag}.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.output_dir}/all_{args.model_tag}.json")


if __name__ == "__main__":
    main()
