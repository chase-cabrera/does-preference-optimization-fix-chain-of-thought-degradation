"""
Symbolic verification at inference time for CoT responses.

Two modes:
  - verify-only:        generate CoT, check each PEMDAS step symbolically,
                        report faithfulness; for CoinFlip / LastLetterConcat,
                        compare the model's answer against the symbolic answer.
  - verify-and-correct: same as verify-only, but for each parseable arithmetic
                        step, replace the model's stated result with the true
                        symbolic result before re-checking the final answer.

For PEMDAS, "verify-and-correct" yields an upper bound on what symbolic
post-hoc patching can recover *given the model's own decomposition* (it does
NOT solve the problem from scratch). For CoinFlip and LastLetterConcat, the
symbolic answer is computed directly from the structured raw_instance, so
"sym_acc" is by construction 100% and is reported only as a sanity check.

Usage:
  python3 symbolic_verifier.py \\
      --base-model "../Project Topic/CPO/model/Llama-2-7b-hf" \\
      --model-tag base \\
      --mode verify-only \\
      --output-dir ../results

  python3 symbolic_verifier.py \\
      --base-model "../Project Topic/CPO/model/Llama-2-7b-hf" \\
      --model-tag dpo_multidomain \\
      --lora ../results/dpo_multidomain/final_checkpoint \\
      --mode verify-and-correct \\
      --output-dir ../results
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
    p.add_argument("--model-tag", required=True,
                   help="Used to name output files, e.g. 'base', 'dpo_multidomain', "
                        "'llama31_base', 'llama31_dpo_multidomain'")
    p.add_argument("--benchmarks-dir", default="../benchmarks")
    p.add_argument("--output-dir", default="../results")
    p.add_argument("--domains", nargs="+",
                   default=["pemdas", "coinflip"])
    p.add_argument("--mode", choices=["verify-only", "verify-and-correct"],
                   default="verify-and-correct")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--device",
                   default="mps" if torch.backends.mps.is_available() else "cpu")
    return p.parse_args()


def load_model(args):
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
        model = model.merge_and_unload()
    model = model.to(args.device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params"
          + (f" (+ LoRA from {args.lora})" if args.lora else ""))
    return model, tokenizer


def generate(model, tokenizer, prompt, device, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=2,
        )
    new_tokens = output[0][input_ids.shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    del output, input_ids, inputs
    if device == "mps":
        torch.mps.empty_cache()
    return result


def make_cot_prompt(direct_prompt):
    if "[Answer]" in direct_prompt:
        return direct_prompt.replace(
            "\n[Answer]\n",
            "\nLet's think step by step.\n\n[Answer]\n"
        )
    return direct_prompt + "\nLet's think step by step.\n"


# ========== PEMDAS Symbolic Verifier ==========

_PEMDAS_STEP_RE = re.compile(
    r"(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)"
)


def verify_pemdas_response(response, ground_truth):
    """Check each arithmetic step symbolically; report per-step correctness."""
    steps = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _PEMDAS_STEP_RE.search(line)
        if match:
            expr = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            model_result = match.group(4)
            try:
                expected = eval(expr)  # noqa: S307 -- expr is digits + operator only
                actual = float(model_result)
                correct = abs(expected - actual) < 0.01
                expected_str = (
                    str(int(expected)) if expected == int(expected) else str(expected)
                )
            except (ValueError, ZeroDivisionError, SyntaxError):
                correct = False
                expected_str = "ERROR"
            steps.append({
                "line": line,
                "expression": expr,
                "model_result": model_result,
                "expected": expected_str,
                "step_correct": correct,
            })

    final_answer = None
    for line in response.strip().split("\n"):
        for match in re.finditer(r"-?\d+\.?\d*", line):
            final_answer = match.group()
        if final_answer:
            break

    return {
        "steps": steps,
        "n_steps": len(steps),
        "n_correct": sum(1 for s in steps if s["step_correct"]),
        "all_correct": all(s["step_correct"] for s in steps) if steps else False,
        "model_answer": final_answer,
        "ground_truth": str(ground_truth),
        "answer_correct": final_answer is not None
        and str(final_answer).strip() == str(ground_truth).strip(),
    }


def correct_pemdas_response(response):
    """Replace each arithmetic step's stated result with the symbolically
    computed correct value, leaving everything else unchanged."""
    lines = response.strip().split("\n")
    corrected_lines = []
    for line in lines:
        match = _PEMDAS_STEP_RE.search(line)
        if match:
            expr = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            model_result = match.group(4)
            try:
                expected = eval(expr)  # noqa: S307
                expected_str = (
                    str(int(expected)) if expected == int(expected) else str(expected)
                )
                corrected = line.replace(f"= {model_result}", f"= {expected_str}")
                corrected_lines.append(corrected)
            except (ValueError, ZeroDivisionError, SyntaxError):
                corrected_lines.append(line)
        else:
            corrected_lines.append(line)
    return "\n".join(corrected_lines)


# ========== CoinFlip Symbolic Verifier ==========

def verify_coinflip_response(response, ground_truth, raw_instance):
    """Compute the true coin state from the structured instance; compare
    against the model's parsed yes/no answer."""
    heads = True
    step_results = []
    for name, flips in raw_instance:
        if flips:
            heads = not heads
        step_results.append({
            "person": name,
            "flips": flips,
            "state_after": "heads" if heads else "tails",
        })
    symbolic_answer = heads  # True = still heads up

    response_lower = response.strip().lower()
    first_line = response_lower.split("\n")[0]
    if "yes" in first_line:
        model_answer = True
    elif "no" in first_line:
        model_answer = False
    else:
        model_answer = "yes" in response_lower

    return {
        "steps": step_results,
        "symbolic_answer": symbolic_answer,
        "model_answer": model_answer,
        "answer_correct": model_answer == ground_truth,
        "symbolic_correct": symbolic_answer == ground_truth,
    }


# ========== LastLetterConcat Symbolic Verifier ==========

def verify_lastletter_response(response, ground_truth, raw_instance):
    """Concatenate last letters from the structured words list; compare
    against the model's parsed string answer."""
    words = []
    for item in raw_instance:
        words.extend(item.split())
    last_letters = [w[-1].lower() for w in words]
    symbolic_answer = "".join(last_letters)

    response_lower = response.strip().lower()
    model_answer = ""
    for line in response_lower.split("\n"):
        line = line.strip()
        if line and re.match(r"^[a-z]+$", line):
            model_answer = line
            break
    else:  # no break
        model_answer = re.sub(r"[^a-z]", "", response_lower.split("\n")[0])

    return {
        "words": words,
        "expected_letters": last_letters,
        "symbolic_answer": symbolic_answer,
        "model_answer": model_answer,
        "answer_correct": model_answer == str(ground_truth).strip().lower(),
        "symbolic_correct": symbolic_answer == str(ground_truth).strip().lower(),
    }


# ========== Main evaluation loop ==========

def evaluate_with_verification(model, tokenizer, args, domain, benchmark):
    instances = benchmark["instances"]
    results = []
    start = time.time()

    correct_model = 0
    correct_symbolic = 0
    correct_after_correction = 0
    total = 0

    for i, inst in enumerate(instances):
        prompt = make_cot_prompt(inst["prompt"])
        response = generate(model, tokenizer, prompt, args.device, args.max_tokens)

        result = {
            "domain": domain,
            "mode": args.mode,
            "model_tag": args.model_tag,
            "instance_id": inst["instance_id"],
            "steps_to_solve": inst["steps_to_solve"],
            "ground_truth": inst["ground_truth"],
            "raw_response": response[:500],
        }

        if domain == "pemdas":
            verification = verify_pemdas_response(response, inst["ground_truth"])
            result.update({
                "model_correct": verification["answer_correct"],
                "n_steps": verification["n_steps"],
                "n_steps_correct": verification["n_correct"],
                "all_steps_correct": verification["all_correct"],
                "faithful": verification["all_correct"] and verification["answer_correct"],
            })
            if args.mode == "verify-and-correct":
                corrected = correct_pemdas_response(response)
                corrected_verification = verify_pemdas_response(
                    corrected, inst["ground_truth"]
                )
                result["corrected_response"] = corrected[:500]
                result["corrected_correct"] = corrected_verification["answer_correct"]
                correct_after_correction += int(corrected_verification["answer_correct"])
            correct_model += int(verification["answer_correct"])

        elif domain == "coinflip":
            verification = verify_coinflip_response(
                response, inst["ground_truth"], inst["raw_instance"]
            )
            result.update({
                "model_correct": verification["answer_correct"],
                "symbolic_answer": verification["symbolic_answer"],
            })
            if args.mode == "verify-and-correct":
                result["corrected_correct"] = verification["symbolic_correct"]
                correct_after_correction += int(verification["symbolic_correct"])
            correct_model += int(verification["answer_correct"])
            correct_symbolic += int(verification["symbolic_correct"])

        elif domain == "lastletterconcat":
            verification = verify_lastletter_response(
                response, inst["ground_truth"], inst["raw_instance"]
            )
            result.update({
                "model_correct": verification["answer_correct"],
                "symbolic_answer": verification["symbolic_answer"],
            })
            if args.mode == "verify-and-correct":
                result["corrected_correct"] = verification["symbolic_correct"]
                correct_after_correction += int(verification["symbolic_correct"])
            correct_model += int(verification["answer_correct"])
            correct_symbolic += int(verification["symbolic_correct"])

        results.append(result)
        total += 1

        if (i + 1) % 20 == 0 or i == len(instances) - 1:
            elapsed = time.time() - start
            model_acc = correct_model / total * 100
            if args.mode == "verify-and-correct":
                corr_acc = correct_after_correction / total * 100
                print(f"  [{i+1:4d}/{len(instances)}] "
                      f"model={model_acc:5.1f}%  corrected={corr_acc:5.1f}%  "
                      f"elapsed={elapsed:.0f}s")
            else:
                print(f"  [{i+1:4d}/{len(instances)}] "
                      f"model={model_acc:5.1f}%  elapsed={elapsed:.0f}s")

    model_acc = correct_model / total * 100 if total else 0
    print(f"\n  {domain} ({args.model_tag}, {args.mode}):")
    print(f"    Model accuracy:     {correct_model}/{total} = {model_acc:.1f}%")
    if domain in ["coinflip", "lastletterconcat"]:
        sym_acc = correct_symbolic / total * 100 if total else 0
        print(f"    Symbolic accuracy:  {correct_symbolic}/{total} = {sym_acc:.1f}%")
    if args.mode == "verify-and-correct":
        corr_acc = correct_after_correction / total * 100 if total else 0
        print(f"    After correction:   {correct_after_correction}/{total} = {corr_acc:.1f}%")

    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model ({args.model_tag})...")
    model, tokenizer = load_model(args)

    all_results = []
    mode_suffix = args.mode.replace("-", "_")  # verify_only / verify_and_correct

    for domain in args.domains:
        bench_path = os.path.join(args.benchmarks_dir, f"{domain}.json")
        if not os.path.exists(bench_path):
            print(f"Skipping {domain}: benchmark not found at {bench_path}")
            continue

        with open(bench_path) as f:
            benchmark = json.load(f)

        print(f"\n{'='*70}")
        print(f"  {benchmark['label']} -- Symbolic Verification ({args.mode})")
        print(f"{'='*70}")

        results = evaluate_with_verification(model, tokenizer, args, domain, benchmark)
        all_results.extend(results)

        out_path = os.path.join(
            args.output_dir,
            f"{domain}_{args.model_tag}_{mode_suffix}.json"
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {out_path}")

    combined = os.path.join(
        args.output_dir,
        f"all_{args.model_tag}_{mode_suffix}.json"
    )
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined}")


if __name__ == "__main__":
    main()
