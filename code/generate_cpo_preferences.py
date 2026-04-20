"""CPO preference-pair generation: ToT expansion + model self-eval scoring."""
import os
import re
import json
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ARITH_LINE_RE = re.compile(
    r"((?:-?\d+\.?\d*\s*[+\-*/]\s*(?:\(?\s*-?\d+\.?\d*\s*\)?))"
    r"(?:\s*[+\-*/]\s*(?:\(?\s*-?\d+\.?\d*\s*\)?))*)\s*=\s*(-?\d+\.?\d*)"
)
ANSWER_LINE_RE = re.compile(r"^\s*answer\s*[:=]\s*(-?\d+\.?\d*)\s*$", re.IGNORECASE)
_PROBLEM_RE = re.compile(r"into a single number:\s*([^\n]+)")


def make_base_cot_prompt(direct_prompt):
    # strip the single-number directive (fights the tree expansion) and go
    # minimal; few-shot exemplars drove llama-2-7b into repetition loops.
    m = _PROBLEM_RE.search(direct_prompt)
    expr = m.group(1).strip() if m else direct_prompt.strip().split("\n")[-1]
    return (
        f"Compute the following expression step by step. Each step has the "
        f"form 'a op b = c'. End with a line 'Answer: <number>'.\n"
        f"Expression: {expr}\n"
    )


def assemble_prompt_with_prefix(base_prompt, chosen_steps, depth, force_answer=False):
    parts = [base_prompt]
    for i, step in enumerate(chosen_steps):
        parts.append(f"Step {i+1}: {step}\n")
    parts.append("Answer:" if force_answer else f"Step {depth+1}: ")
    return "".join(parts)


def parse_step_line(line):
    line = line.strip()
    if not line:
        return {"kind": "junk", "text": "", "correct": False}

    am = ANSWER_LINE_RE.match(line)
    if am:
        return {"kind": "answer", "text": line, "value": am.group(1), "correct": None}

    m = ARITH_LINE_RE.search(line)
    if not m:
        return {"kind": "junk", "text": line, "correct": False}

    expr_str = m.group(1).strip()
    result_str = m.group(2).strip()
    try:
        expected = eval(expr_str)
        actual = float(result_str)
        ok = abs(expected - actual) < 0.01
    except Exception:
        return {"kind": "junk", "text": line, "correct": False}

    return {
        "kind": "arith",
        "text": line,
        "expression": expr_str,
        "result": result_str,
        "expected": expected,
        "correct": ok,
    }


@torch.no_grad()
def batched_sample(model, tokenizer, prompt, n, temperature, max_new_tokens):
    # mps+fp16 occasionally yields inf/nan in the sampling dist, so retry cooler.
    enc = tokenizer([prompt] * n, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(DEVICE)
    attn = enc["attention_mask"].to(DEVICE)

    last_err = None
    for t in (temperature, max(temperature * 0.6, 0.3), 0.2):
        try:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(t, 0.05),
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(
                out[:, input_ids.shape[1]:], skip_special_tokens=True
            )
            del out, input_ids, attn, enc
            if DEVICE == "mps":
                torch.mps.empty_cache()
            return decoded
        except RuntimeError as e:
            last_err = e
            if "inf" not in str(e) and "nan" not in str(e):
                raise
            if DEVICE == "mps":
                torch.mps.empty_cache()
    raise last_err


_PROSE_RE = re.compile(
    r"\b(subtract|add|multiply|divide|both\s+sides|to\s+see|to\s+make|"
    r"because|therefore|thao|swing|Seks|감사|that|step|note|answer\s+is"
    r"|you\s|let\s|first|second|third|note\s)",
    re.IGNORECASE,
)


def _strip_to_arith_charset(text):
    out = []
    for ch in text:
        if ch.isdigit() or ch in " +-*/=().,:" or ch.isascii() and ch.isalpha():
            out.append(ch)
        elif ch == "\n":
            out.append(ch)
    return "".join(out)


def trim_to_first_line(text):
    text = _strip_to_arith_charset(text)
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.count("!") >= 3 or abs(s.count("(") - s.count(")")) > 1:
            continue
        if _PROSE_RE.search(s):
            continue
        non_arith = sum(1 for ch in s if not (ch.isdigit() or ch in " +-*/=().,:Answer"))
        if len(s) > 0 and non_arith / len(s) > 0.45:
            continue
        return s
    return ""


_LEADING_NOISE_RE = re.compile(r"^[^0-9\-(]+")
_STEP_PREFIX_RE = re.compile(r"^Step\s*\d+\s*[:.]?\s*", re.IGNORECASE)


def clean_chosen_step(text):
    text = _STEP_PREFIX_RE.sub("", text.strip())
    cleaned = _LEADING_NOISE_RE.sub("", text)
    if "=" in cleaned:
        cleaned = cleaned.strip()
    else:
        cleaned = text.strip()
    return cleaned or text.strip()


def build_self_eval_prompt(problem_expr, prior_steps_text, candidate_step):
    # "useful next step toward solving" rewards decomposition over early Answer:
    prior = prior_steps_text.strip() if prior_steps_text.strip() else "(no steps yet)"
    return (
        f"Problem: simplify {problem_expr}\n"
        f"Steps so far:\n{prior}\n"
        f"Proposed next step: {candidate_step}\n"
        f"Is the proposed next step a correct and useful partial reduction "
        f"that moves the problem closer to its final value? "
        f"Answer with Yes or No.\nAnswer:"
    )


@torch.no_grad()
def batched_self_eval(model, tokenizer, prompts, yes_id, no_id):
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(DEVICE)
    attn = enc["attention_mask"].to(DEVICE)
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    last_idx = attn.sum(dim=1) - 1
    next_logits = out.logits[torch.arange(out.logits.size(0)), last_idx, :]
    score = (next_logits[:, yes_id] - next_logits[:, no_id]).float().cpu().tolist()
    del out, next_logits, input_ids, attn, enc
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return score


def find_yes_no_token_ids(tokenizer):
    def single(cands):
        for c in cands:
            ids = tokenizer.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return tokenizer.encode(cands[0], add_special_tokens=False)[0]
    return (
        single([" Yes", "Yes", " yes", "yes"]),
        single([" No", "No", " no", "no"]),
    )


def extract_problem_expression(prompt):
    m = re.search(r"into a single number:\s*([^\n]+)", prompt)
    return m.group(1).strip() if m else prompt[-200:]


def expand_tree_for_instance(model, tokenizer, inst, args, yes_id, no_id):
    base_prompt = make_base_cot_prompt(inst["prompt"])
    problem_expr = extract_problem_expression(inst["prompt"])

    if args.debug:
        print(f"\n=== INSTANCE {inst.get('instance_id')} "
              f"(steps_to_solve={inst.get('steps_to_solve')}) ===")
        print(f"  expr: {problem_expr}")

    pairs = []
    chosen_steps = []
    chosen_arith_correct_chain = []
    expanded_depth = 0
    stop_reason = "max_depth"

    for depth in range(args.max_depth):
        force_answer_prompt = (
            depth >= args.min_arith_before_answer
            and depth + 1 >= len(chosen_steps) + 1
        )

        full_prompt_step = assemble_prompt_with_prefix(
            base_prompt, chosen_steps, depth, force_answer=False
        )

        if force_answer_prompt:
            # split beam across 'Step N:' and 'Answer:' forcings
            n_step = max(1, args.beam_width // 2)
            n_answer = args.beam_width - n_step
            raw_step = batched_sample(
                model, tokenizer, full_prompt_step, n=n_step,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens_per_step,
            )
            full_prompt_answer = assemble_prompt_with_prefix(
                base_prompt, chosen_steps, depth, force_answer=True
            )
            raw_answer = batched_sample(
                model, tokenizer, full_prompt_answer, n=n_answer,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens_per_step,
            )
            raw_candidates = [("step", c) for c in raw_step] + \
                             [("answer", c) for c in raw_answer]
        else:
            raw_step = batched_sample(
                model, tokenizer, full_prompt_step, n=args.beam_width,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens_per_step,
            )
            raw_candidates = [("step", c) for c in raw_step]

        first_lines = []
        for kind_force, c in raw_candidates:
            line = trim_to_first_line(c)
            if not line:
                continue
            if kind_force == "answer" and not line.lower().startswith("answer"):
                line = "Answer: " + line
            first_lines.append(line)

        seen = set()
        unique = []
        for line in first_lines:
            if line not in seen:
                seen.add(line)
                unique.append(line)

        parsed = [parse_step_line(line) for line in unique]

        if args.debug:
            print(f"  depth {depth}: {len(unique)} unique candidates "
                  f"(force_answer={force_answer_prompt})")
            for line, p in zip(unique, parsed):
                print(f"    [{p['kind']:6s}] {line[:80]}")

        # require arithmetic below min_arith_before_answer so the tree can't
        # terminate before any decomposition has happened
        if depth < args.min_arith_before_answer:
            valid_indices = [i for i, p in enumerate(parsed) if p["kind"] == "arith"]
        else:
            valid_indices = [i for i, p in enumerate(parsed)
                             if p["kind"] in ("arith", "answer")]

        if not valid_indices:
            stop_reason = "no_parseable_candidates"
            break

        eval_prompts = [
            build_self_eval_prompt(
                problem_expr,
                "\n".join(f"Step {j+1}: {s}" for j, s in enumerate(chosen_steps)),
                unique[i],
            )
            for i in valid_indices
        ]
        scores = batched_self_eval(model, tokenizer, eval_prompts, yes_id, no_id)

        best_local = max(range(len(scores)), key=lambda k: scores[k])
        best_idx = valid_indices[best_local]
        best_score = scores[best_local]
        chosen_step = unique[best_idx]
        chosen_meta = parsed[best_idx]

        if args.debug:
            filt = "arith-only" if depth < args.min_arith_before_answer else "arith+answer"
            print(f"  scored {len(valid_indices)} valid candidates (filter: {filt})")
            for k, vi in enumerate(valid_indices):
                marker = "*" if k == best_local else " "
                print(f"    {marker} score={scores[k]:+.3f} "
                      f"[{parsed[vi]['kind']:6s}] {unique[vi][:70]}")

        for k, vi in enumerate(valid_indices):
            if k == best_local:
                continue
            if scores[k] >= best_score - args.score_margin:
                continue
            cand_kind = parsed[vi]["kind"]

            # anti-poisoning: drop pairs where self-eval ranked a symbolically
            # wrong step above a symbolically right one
            if (not args.keep_misleading_pairs
                and chosen_meta["kind"] == "arith"
                and parsed[vi]["kind"] == "arith"
                and chosen_meta.get("correct") is False
                and parsed[vi].get("correct") is True):
                continue

            chosen_force = "answer" if chosen_meta["kind"] == "answer" else "step"
            rejected_force = "answer" if cand_kind == "answer" else "step"
            chosen_prompt = assemble_prompt_with_prefix(
                base_prompt, chosen_steps, depth,
                force_answer=(chosen_force == "answer"),
            )
            if chosen_force == rejected_force:
                pair_prompt = chosen_prompt
                chosen_text = (
                    chosen_step[len("Answer:"):].lstrip()
                    if chosen_force == "answer"
                    and chosen_step.lower().startswith("answer:")
                    else chosen_step
                )
                rejected_text = (
                    unique[vi][len("Answer:"):].lstrip()
                    if rejected_force == "answer"
                    and unique[vi].lower().startswith("answer:")
                    else unique[vi]
                )
            else:
                pair_prompt = chosen_prompt
                chosen_text = chosen_step
                rejected_text = unique[vi]

            pairs.append({
                "prompt": pair_prompt,
                "chosen": " " + chosen_text,
                "rejected": " " + rejected_text,
                "depth": depth,
                "domain": "pemdas",
                "steps_to_solve": inst.get("steps_to_solve"),
                "chosen_score": float(best_score),
                "rejected_score": float(scores[k]),
                "chosen_kind": chosen_meta["kind"],
                "rejected_kind": cand_kind,
                "chosen_arith_correct": (
                    chosen_meta.get("correct") if chosen_meta["kind"] == "arith" else None
                ),
                "rejected_arith_correct": (
                    parsed[vi].get("correct") if parsed[vi]["kind"] == "arith" else None
                ),
            })

        chosen_steps.append(
            clean_chosen_step(chosen_step) if chosen_meta["kind"] == "arith"
            else chosen_step
        )
        if chosen_meta["kind"] == "arith":
            chosen_arith_correct_chain.append(bool(chosen_meta["correct"]))
        else:
            chosen_arith_correct_chain.append(None)
        expanded_depth = depth + 1

        if chosen_meta["kind"] == "answer":
            stop_reason = "answer_emitted"
            break

    return pairs, {
        "instance_id": inst.get("instance_id"),
        "steps_to_solve": inst.get("steps_to_solve"),
        "depth_reached": expanded_depth,
        "stop_reason": stop_reason,
        "n_pairs": len(pairs),
        "chosen_steps": chosen_steps,
        "chosen_arith_correct_chain": chosen_arith_correct_chain,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--benchmarks-dir", default="../benchmarks")
    p.add_argument("--output", required=True)
    p.add_argument("--meta-output", default=None)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens-per-step", type=int, default=40)
    p.add_argument("--score-margin", type=float, default=0.0)
    p.add_argument("--max-instances", type=int, default=None)
    p.add_argument("--min-steps", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--min-arith-before-answer", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--keep-misleading-pairs", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()

    yes_id, no_id = find_yes_no_token_ids(tokenizer)

    with open(os.path.join(args.benchmarks_dir, "pemdas.json")) as f:
        bench = json.load(f)
    instances = [i for i in bench["instances"]
                 if args.min_steps <= i["steps_to_solve"] <= args.max_steps]
    if args.max_instances is not None:
        instances = instances[:args.max_instances]
    print(f"{len(instances)} instances ({args.min_steps}-{args.max_steps} steps)")

    all_pairs, all_meta = [], []
    start = time.time()

    def save():
        with open(args.output, "w") as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)
        if args.meta_output:
            with open(args.meta_output, "w") as f:
                json.dump(all_meta, f, ensure_ascii=False, indent=2)

    for idx, inst in enumerate(instances):
        try:
            pairs, meta = expand_tree_for_instance(
                model, tokenizer, inst, args, yes_id, no_id
            )
        except Exception as e:
            print(f"  [{idx+1}/{len(instances)}] EXCEPTION: {e}")
            continue
        all_pairs.extend(pairs)
        all_meta.append(meta)

        if (idx + 1) % 5 == 0 or idx == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(instances) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(instances)}] pairs={len(all_pairs)} "
                  f"depth={meta['depth_reached']} stop={meta['stop_reason']} "
                  f"({rate:.2f} q/min, ETA={eta:.0f}min)")
            save()

    seen = set()
    deduped = []
    for pr in all_pairs:
        key = (pr["prompt"][-120:], pr["chosen"][:80], pr["rejected"][:80])
        if key not in seen:
            seen.add(key)
            deduped.append(pr)

    with open(args.output, "w") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    if args.meta_output:
        with open(args.meta_output, "w") as f:
            json.dump(all_meta, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for pr in deduped if pr["chosen_arith_correct"] is True)
    n_wrong = sum(1 for pr in deduped if pr["chosen_arith_correct"] is False)
    n_rej_ok = sum(1 for pr in deduped if pr["rejected_arith_correct"] is True)
    print(f"done: {len(deduped)} pairs ({len(all_pairs)} raw), "
          f"chosen ok/wrong={n_ok}/{n_wrong}, rejected-ok={n_rej_ok}, "
          f"{(time.time()-start)/60:.1f}min -> {args.output}")


if __name__ == "__main__":
    main()
