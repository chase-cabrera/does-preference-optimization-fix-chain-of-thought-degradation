"""
Generate preference pairs for DPO training across all benchmark domains
(PEMDAS, LastLetterConcatenation, CoinFlip).

For each domain, the base model is sampled multiple times per instance.
Correct responses are paired against incorrect ones to form DPO training data.

Usage:
  python3 generate_multidomain_preferences.py --base-model models/Llama-2-7b-hf
  python3 generate_multidomain_preferences.py --base-model models/Llama-3.1-8B --output-dir ../data_llama31
  python3 generate_multidomain_preferences.py --domains pemdas coinflip
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
            pass
    return None


def parse_answer_lastletterconcat(response):
    response = response.strip().lower()
    for line in response.split("\n"):
        line = line.strip()
        if line and re.match(r"^[a-z]+$", line):
            return line
    cleaned = re.sub(r"[^a-z]", "", response.split("\n")[0].lower())
    return cleaned if cleaned else None


def parse_answer_coinflip(response):
    first_line = response.strip().lower().split("\n")[0]
    if "yes" in first_line:
        return "yes"
    if "no" in first_line:
        return "no"
    if "yes" in response.lower():
        return "yes"
    if "no" in response.lower():
        return "no"
    return None


DOMAIN_CONFIG = {
    "pemdas": {
        "parse": parse_answer_pemdas,
        "check": lambda p, gt: p is not None and str(p).strip() == str(gt).strip(),
        "max_new_tokens": 128,
        "truncate_lines": 3,
    },
    "lastletterconcat": {
        "parse": parse_answer_lastletterconcat,
        "check": lambda p, gt: p is not None and p.lower() == str(gt).strip().lower(),
        "max_new_tokens": 256,
        "truncate_lines": 5,
    },
    "coinflip": {
        "parse": parse_answer_coinflip,
        "check": lambda p, gt: p is not None and p == ("yes" if gt is True else "no"),
        "max_new_tokens": 256,
        "truncate_lines": 5,
    },
}


def truncate_response(raw, max_lines):
    lines = []
    for line in raw.split("\n"):
        lines.append(line)
        if any(marker in line.lower() for marker in ["[answer]", "[question]", "[instructions]"]):
            break
    return "\n".join(lines[:max_lines])


def generate_pairs_for_domain(model, tokenizer, domain, instances, args):
    config = DOMAIN_CONFIG[domain]
    paired_data = []
    total_correct = 0
    total_wrong = 0
    start = time.time()

    for idx, inst in enumerate(instances):
        cot_prompt = make_cot_prompt(inst["prompt"])
        ground_truth = inst["ground_truth"]
        correct_responses = []
        wrong_responses = []

        for _ in range(args.n_samples):
            raw = generate_response(model, tokenizer, cot_prompt, args.temperature, config["max_new_tokens"])
            response = truncate_response(raw, config["truncate_lines"])
            parsed = config["parse"](response)

            if config["check"](parsed, ground_truth):
                correct_responses.append(response.strip())
            else:
                wrong_responses.append(response.strip())

        total_correct += len(correct_responses)
        total_wrong += len(wrong_responses)

        if correct_responses and wrong_responses:
            for chosen in correct_responses:
                for rejected in wrong_responses[:3]:
                    if chosen != rejected:
                        paired_data.append({
                            "prompt": cot_prompt,
                            "chosen": " " + chosen,
                            "rejected": " " + rejected,
                            "domain": domain,
                            "steps_to_solve": inst["steps_to_solve"],
                        })

        elapsed = time.time() - start
        if (idx + 1) % 10 == 0 or idx == 0:
            rate = (idx + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(instances) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{domain}] [{idx+1}/{len(instances)}] "
                  f"correct={total_correct} wrong={total_wrong} "
                  f"pairs={len(paired_data)} "
                  f"({rate:.1f} q/min, ETA={eta:.0f}min)")

    return paired_data, total_correct, total_wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--benchmarks-dir", default="../benchmarks")
    parser.add_argument("--domains", nargs="+", default=["pemdas", "lastletterconcat", "coinflip"])
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--output-dir", default="../data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    all_pairs = []
    summary = {}

    for domain in args.domains:
        bench_path = os.path.join(args.benchmarks_dir, f"{domain}.json")
        if not os.path.exists(bench_path):
            print(f"Skipping {domain}: {bench_path} not found")
            continue

        with open(bench_path) as f:
            bench = json.load(f)

        instances = [i for i in bench["instances"] if i["steps_to_solve"] <= args.max_steps]
        print(f"\n{'='*60}")
        print(f"  {domain}: {len(instances)} instances (steps <= {args.max_steps})")
        print(f"{'='*60}")

        pairs, n_correct, n_wrong = generate_pairs_for_domain(model, tokenizer, domain, instances, args)

        deduped = []
        seen = set()
        for p in pairs:
            key = (p["domain"], p["chosen"][:80], p["rejected"][:80])
            if key not in seen:
                seen.add(key)
                deduped.append(p)

        domain_path = os.path.join(args.output_dir, f"{domain}_preferences.json")
        with open(domain_path, "w") as f:
            json.dump(deduped, f, ensure_ascii=False, indent=2)

        all_pairs.extend(deduped)
        summary[domain] = {"deduped_pairs": len(deduped), "total_correct": n_correct, "total_wrong": n_wrong}
        print(f"  {domain}: {len(deduped)} pairs saved to {domain_path}")

    if len(args.domains) > 1:
        combined_path = os.path.join(args.output_dir, "multidomain_preferences.json")
        with open(combined_path, "w") as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    for domain, s in summary.items():
        print(f"  {domain:20s}: {s['deduped_pairs']:4d} pairs")
    print(f"  {'TOTAL':20s}: {len(all_pairs):4d} pairs")


if __name__ == "__main__":
    main()
