"""
Synthesise PEMDAS preference pairs with multi-step arithmetic traces.
Chosen = correct step-by-step; rejected = same format, wrong final answer.

Usage:
  python3 generate_pemdas_synthetic_preferences.py \\
      --benchmark ../benchmarks/pemdas.json \\
      --output ../data/pemdas_synthetic_preferences.json
"""
import os
import re
import json
import random
import argparse


EXPR_RE = re.compile(r"expression into a single number:\s*([^\n]+)")
# One binary operation between two numbers or parenthesised sub-expressions.
# We greedily find the first INNERMOST parenthesised arithmetic expression,
# reduce it, and substitute the value back.
INNER_PAREN = re.compile(r"\(([^()]*)\)")
# A single binary op inside a flat (paren-free) string.
BINARY_OP = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)")


def _fmt(n):
    if isinstance(n, float) and n == int(n):
        return str(int(n))
    if isinstance(n, float):
        return f"{n:g}"
    return str(n)


def _reduce_flat(expr):
    """Reduce a paren-free arithmetic string respecting precedence."""
    steps = []
    tokens = re.findall(r"-?\d+(?:\.\d+)?|[+\-*/]", expr)

    def one_pass(ops):
        nonlocal tokens
        i = 1
        while i < len(tokens) - 1:
            op = tokens[i]
            if op in ops:
                a, b = tokens[i-1], tokens[i+1]
                try:
                    a_n = float(a)
                    b_n = float(b)
                    if op == "+": v = a_n + b_n
                    elif op == "-": v = a_n - b_n
                    elif op == "*": v = a_n * b_n
                    elif op == "/":
                        if b_n == 0:
                            return False
                        v = a_n / b_n
                except ValueError:
                    return False
                v_s = _fmt(v)
                steps.append(f"{_fmt(a_n if not a_n.is_integer() else int(a_n))} {op} "
                             f"{_fmt(b_n if not b_n.is_integer() else int(b_n))} = {v_s}")
                tokens = tokens[:i-1] + [v_s] + tokens[i+2:]
                i = 1
            else:
                i += 2
        return True

    if not one_pass({"*", "/"}):
        return None, None
    if not one_pass({"+", "-"}):
        return None, None

    if len(tokens) != 1:
        return None, None
    return steps, tokens[0]


def reduce_expression(expr):
    """Reduce expression innermost-parens-first. Returns (steps, final_val) or (None, None)."""
    all_steps = []
    work = expr.strip()

    while True:
        m = INNER_PAREN.search(work)
        if not m:
            break
        inner = m.group(1)
        # Bare number in parens — just strip.
        if re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", inner):
            val = inner.strip()
            work = work[:m.start()] + val + work[m.end():]
            continue
        steps, val = _reduce_flat(inner)
        if steps is None:
            return None, None
        all_steps.extend(steps)
        work = work[:m.start()] + val + work[m.end():]

    # Reduce remaining flat expression.
    steps, val = _reduce_flat(work)
    if steps is None:
        return None, None
    all_steps.extend(steps)
    return all_steps, val


def _perturb_answer(value_str):
    try:
        n = float(value_str)
    except ValueError:
        return value_str + "x"
    offset = random.choice([-3, -2, -1, 1, 2, 3])
    new = n + offset
    if new == n:
        new = n + 1
    return _fmt(int(new) if new == int(new) else new)


def make_chosen(steps, final_val):
    lines = [f"  {s}" for s in steps]
    lines.append(f"Answer: {final_val}")
    return "\n".join(lines)


def make_rejected(steps, final_val):
    """Same format but with a wrong final answer."""
    wrong = _perturb_answer(final_val)
    lines = [f"  {s}" for s in steps]
    lines.append(f"Answer: {wrong}")
    return "\n".join(lines)


def make_cot_prompt(direct_prompt):
    if "[Answer]" in direct_prompt:
        return direct_prompt.replace(
            "\n[Answer]\n",
            "\nLet's think step by step. Show each arithmetic step on its own line "
            "as 'a op b = c', then write the final answer on a line 'Answer: <number>'.\n\n[Answer]\n"
        )
    return direct_prompt + ("\nLet's think step by step. Show each arithmetic step on its "
                            "own line as 'a op b = c', then write the final answer on a "
                            "line 'Answer: <number>'.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="../benchmarks/pemdas.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=12,
                        help="Only use instances with steps_to_solve <= this.")
    parser.add_argument("--min-arith", type=int, default=2,
                        help="Drop traces with fewer arithmetic lines than this.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.benchmark) as f:
        bench = json.load(f)

    pairs = []
    skipped_parse = 0
    skipped_answer_mismatch = 0
    skipped_short = 0

    for inst in bench["instances"]:
        if inst["steps_to_solve"] > args.max_steps:
            continue
        m = EXPR_RE.search(inst["prompt"])
        if not m:
            skipped_parse += 1
            continue
        expr = m.group(1).strip()
        steps, final_val = reduce_expression(expr)
        if steps is None:
            skipped_parse += 1
            continue
        if len(steps) < args.min_arith:
            skipped_short += 1
            continue

        # Sanity-check: our computed value should match ground truth
        try:
            if abs(float(final_val) - float(inst["ground_truth"])) > 1e-6:
                skipped_answer_mismatch += 1
                continue
        except ValueError:
            skipped_answer_mismatch += 1
            continue

        prompt = make_cot_prompt(inst["prompt"])
        chosen = make_chosen(steps, final_val)
        rejected = make_rejected(steps, final_val)
        pairs.append({
            "prompt": prompt,
            "chosen": " " + chosen,
            "rejected": " " + rejected,
            "domain": "pemdas",
            "steps_to_solve": inst["steps_to_solve"],
            "n_arith_steps": len(steps),
        })

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    n = len(pairs)
    mean_steps = sum(p["n_arith_steps"] for p in pairs) / max(n, 1)
    print(f"Instances considered: {sum(1 for i in bench['instances'] if i['steps_to_solve'] <= args.max_steps)}")
    print(f"Pairs written:        {n}")
    print(f"Mean arith steps:     {mean_steps:.1f}")
    print(f"Skipped (parse):      {skipped_parse}")
    print(f"Skipped (mismatch):   {skipped_answer_mismatch}")
    print(f"Skipped (short):      {skipped_short}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
