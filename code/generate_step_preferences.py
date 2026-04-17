"""
Generate step level preference pairs for DPO training.

Instead of pairing whole correct vs incorrect traces, this creates
preference pairs at each intermediate reasoning step using symbolically
computed correct answers as the chosen response.

Usage:
  python3 generate_step_preferences.py --base-model models/Llama-2-7b-hf
  python3 generate_step_preferences.py --base-model models/Llama-3.1-8B --output ../data_llama31/step_preferences.json
"""
import os
import json
import re
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def make_cot_prompt(direct_prompt):
    if "[Answer]" in direct_prompt:
        return direct_prompt.replace(
            "\n[Answer]\n",
            "\nLet's think step by step.\n\n[Answer]\n"
        )
    return direct_prompt + "\nLet's think step by step.\n"


def generate_response(model, tokenizer, prompt, temperature=1.0, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(DEVICE),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_k=50,
            pad_token_id=2,
        )
    new_tokens = output[0][input_ids.shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    del output, input_ids, inputs
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return result


def parse_reasoning_steps(response):
    """Parse a CoT response into individual arithmetic steps."""
    steps = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.search(
            r"((?:-?\d+\.?\d*\s*[+\-*/]\s*(?:\(?\s*-?\d+\.?\d*\s*\)?))(?:\s*[+\-*/]\s*(?:\(?\s*-?\d+\.?\d*\s*\)?))*)\s*=\s*(-?\d+\.?\d*)",
            line
        )
        if match:
            expr_str = match.group(1).strip()
            result_str = match.group(2).strip()
            try:
                expected = eval(expr_str)
                actual = float(result_str)
                steps.append({
                    "text": line,
                    "expression": expr_str,
                    "result": result_str,
                    "expected": expected,
                    "correct": abs(expected - actual) < 0.01,
                })
            except (ValueError, ZeroDivisionError, SyntaxError):
                steps.append({
                    "text": line, "expression": expr_str,
                    "result": result_str, "expected": None, "correct": False,
                })
    return steps


def generate_step_pairs_for_instance(model, tokenizer, inst, args):
    """Generate step level pairs using symbolic ground truth.

    For each incorrect model step, pair it against the symbolically
    computed correct answer.
    """
    cot_prompt = make_cot_prompt(inst["prompt"])

    all_traces = []
    for _ in range(args.n_samples):
        raw = generate_response(model, tokenizer, cot_prompt, args.temperature)
        lines = []
        for line in raw.split("\n"):
            lines.append(line)
            if any(m in line.lower() for m in ["[answer]", "[question]"]):
                break
        response = "\n".join(lines[:6])
        steps = parse_reasoning_steps(response)
        all_traces.append({"response": response, "steps": steps})

    step_pairs = []
    seen_pairs = set()

    for trace in all_traces:
        for step_idx, step in enumerate(trace["steps"]):
            if step["correct"] or step["expected"] is None:
                continue

            prefix_correct = all(trace["steps"][j]["correct"] for j in range(step_idx))
            if not prefix_correct:
                continue

            prefix_text = cot_prompt
            for j in range(step_idx):
                prefix_text += "\n" + trace["steps"][j]["text"]

            expected_str = str(int(step["expected"])) if step["expected"] == int(step["expected"]) else str(step["expected"])
            chosen_text = f"{step['expression']} = {expected_str}"
            rejected_text = step["text"]

            pair_key = (chosen_text, rejected_text, step_idx)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            step_pairs.append({
                "prompt": prefix_text,
                "chosen": " " + chosen_text,
                "rejected": " " + rejected_text,
                "step_index": step_idx,
                "steps_to_solve": inst["steps_to_solve"],
                "correct_result": expected_str,
                "incorrect_result": step["result"],
            })

    return step_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--benchmarks-dir", default="../benchmarks")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--output", default="../data/step_preferences.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    with open(os.path.join(args.benchmarks_dir, "pemdas.json")) as f:
        bench = json.load(f)

    instances = [i for i in bench["instances"]
                 if args.min_steps <= i["steps_to_solve"] <= args.max_steps]
    print(f"Using {len(instances)} PEMDAS instances (steps {args.min_steps}-{args.max_steps})")

    all_pairs = []
    start = time.time()

    for idx, inst in enumerate(instances):
        pairs = generate_step_pairs_for_instance(model, tokenizer, inst, args)
        all_pairs.extend(pairs)

        elapsed = time.time() - start
        if (idx + 1) % 10 == 0 or idx == 0:
            rate = (idx + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(instances) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(instances)}] pairs={len(all_pairs)} "
                  f"({rate:.1f} q/min, ETA={eta:.0f}min)")
            with open(args.output, "w") as f:
                json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    # Deduplicate
    deduped = []
    seen = set()
    for p in all_pairs:
        key = (p["chosen"][:80], p["rejected"][:80], p["step_index"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    with open(args.output, "w") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"\nInstances: {len(instances)}, Raw: {len(all_pairs)}, Deduped: {len(deduped)}")
    print(f"Elapsed: {(time.time()-start)/60:.1f} min")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
