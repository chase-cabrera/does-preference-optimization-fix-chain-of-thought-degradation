"""
Corrupt preference pairs for the incorrect-CoT DPO diagnostic.

Scrambles intermediate arithmetic steps while preserving the final answer.

Usage:
  python3 generate_corrupted_preferences.py --input ../data/pemdas_preferences.json
"""
import os
import json
import re
import random
import argparse
from copy import deepcopy


ARITH_RE = re.compile(
    r"(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)"
)
COMMUTATIVE = {"+", "*"}
ALL_OPS = ["+", "-", "*", "/"]


def _parse_num(s):
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return None


def _format_num(n):
    """Format n as int if whole, else compact float."""
    if isinstance(n, int) or (isinstance(n, float) and n == int(n)):
        return str(int(n))
    return f"{n:g}"


def _perturb_result(result_str):
    n = _parse_num(result_str)
    if n is None:
        return result_str, False
    offset = random.choice([-3, -2, -1, 1, 2, 3])
    new = n + offset
    new_str = _format_num(new)
    return new_str, new_str != result_str


def _pick_wrong_op(lhs, rhs, current_op, target_result):
    """Pick an operator that produces a different result."""
    candidates = [o for o in ALL_OPS if o != current_op]
    random.shuffle(candidates)
    for new_op in candidates:
        try:
            val = eval(f"{lhs} {new_op} {rhs}")
            if abs(val - target_result) > 1e-6:
                return new_op
        except (ZeroDivisionError, SyntaxError):
            continue
    return candidates[0] if candidates else current_op


def _corrupt_line(line, match, protect_result):
    """Try to corrupt one arithmetic line. Returns (new_line, changed)."""
    lhs_s, op, rhs_s, result_s = match.group(1), match.group(2), match.group(3), match.group(4)
    lhs = _parse_num(lhs_s)
    rhs = _parse_num(rhs_s)
    result = _parse_num(result_s)
    orig_expr = f"{lhs_s} {op} {rhs_s}"

    strategies = []
    if op not in COMMUTATIVE and lhs_s != rhs_s:
        strategies.append("swap_operands")
    if lhs is not None and rhs is not None and result is not None:
        strategies.append("wrong_op")
    if not protect_result:
        strategies.append("wrong_result")

    random.shuffle(strategies)
    for strategy in strategies:
        if strategy == "swap_operands":
            new_expr = f"{rhs_s} {op} {lhs_s}"
            if new_expr != orig_expr:
                return line.replace(orig_expr, new_expr, 1), True
        elif strategy == "wrong_op":
            new_op = _pick_wrong_op(lhs, rhs, op, result)
            if new_op != op:
                return line.replace(orig_expr, f"{lhs_s} {new_op} {rhs_s}", 1), True
        elif strategy == "wrong_result":
            new_result, changed = _perturb_result(result_s)
            if changed:
                return line.replace(f"= {result_s}", f"= {new_result}", 1), True
    return line, False


def corrupt_pemdas_reasoning(response_text):
    """Returns (new_text, n_arith_lines, n_corrupted_lines)."""
    lines = response_text.split("\n")
    arith_idxs = [i for i, l in enumerate(lines) if ARITH_RE.search(l)]
    if not arith_idxs:
        return response_text, 0, 0

    last_arith = arith_idxs[-1]
    out = list(lines)
    corrupted = 0
    for i in arith_idxs:
        m = ARITH_RE.search(lines[i])
        protect = (i == last_arith)
        new_line, did = _corrupt_line(lines[i], m, protect_result=protect)
        if did:
            out[i] = new_line
            corrupted += 1

    # Fallback: ensure at least one line is corrupted.
    if corrupted == 0:
        for i in arith_idxs:
            m = ARITH_RE.search(lines[i])
            lhs_s, op, rhs_s = m.group(1), m.group(2), m.group(3)
            for new_op in [o for o in ALL_OPS if o != op]:
                new_line = lines[i].replace(
                    f"{lhs_s} {op} {rhs_s}", f"{lhs_s} {new_op} {rhs_s}", 1
                )
                if new_line != lines[i]:
                    out[i] = new_line
                    corrupted += 1
                    break
            if corrupted > 0:
                break

    return "\n".join(out), len(arith_idxs), corrupted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-noncorruptable", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_corrupted{ext}"

    random.seed(args.seed)

    with open(args.input) as f:
        pairs = json.load(f)

    corrupted_pairs = []
    n_pairs_with_arith = 0
    n_pairs_corrupted = 0
    total_arith = 0
    total_corrupted_steps = 0
    dropped = 0

    for pair in pairs:
        new_chosen, n_arith, n_corr = corrupt_pemdas_reasoning(pair["chosen"])
        total_arith += n_arith
        total_corrupted_steps += n_corr

        if n_arith > 0:
            n_pairs_with_arith += 1
        if n_corr > 0:
            n_pairs_corrupted += 1

        if args.drop_noncorruptable and n_corr == 0:
            dropped += 1
            continue

        out = deepcopy(pair)
        if new_chosen != pair["chosen"]:
            out["chosen"] = new_chosen
        corrupted_pairs.append(out)

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(corrupted_pairs, f, ensure_ascii=False, indent=2)

    print(f"Input pairs:                 {len(pairs)}")
    print(f"Pairs with arithmetic lines: {n_pairs_with_arith}  "
          f"({n_pairs_with_arith/max(len(pairs),1)*100:.1f}%)")
    print(f"Pairs actually corrupted:    {n_pairs_corrupted}  "
          f"({n_pairs_corrupted/max(len(pairs),1)*100:.1f}%)")
    print(f"Total arithmetic steps:      {total_arith}")
    print(f"Corrupted steps:             {total_corrupted_steps}  "
          f"({total_corrupted_steps/max(total_arith,1)*100:.1f}%)")
    if args.drop_noncorruptable:
        print(f"Dropped non-corruptable:     {dropped}")
    print(f"Output pairs:                {len(corrupted_pairs)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
