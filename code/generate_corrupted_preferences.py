"""
Generate corrupted preference pairs for the incorrect CoT DPO diagnostic.

Takes correct preference pairs and scrambles intermediate reasoning steps
while preserving the correct final answer. If DPO trained on corrupted pairs
performs similarly to correct CoT DPO, the improvement is pattern matching.

Usage:
  python3 generate_corrupted_preferences.py --input ../data/pemdas_preferences.json
"""
import os
import json
import re
import random
import argparse
from copy import deepcopy


def corrupt_pemdas_reasoning(response_text):
    lines = response_text.strip().split("\n")
    corrupted_lines = []

    for i, line in enumerate(lines):
        match = re.search(
            r"(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)",
            line
        )
        if match and i < len(lines) - 1:
            lhs, op, rhs, result = match.group(1), match.group(2), match.group(3), match.group(4)
            strategy = random.choice(["wrong_result", "swap_operands", "wrong_op"])

            if strategy == "wrong_result":
                try:
                    offset = random.choice([-3, -2, -1, 1, 2, 3])
                    corrupted = line.replace(f"= {result}", f"= {int(result) + offset}")
                except ValueError:
                    corrupted = line
            elif strategy == "swap_operands":
                corrupted = line.replace(f"{lhs} {op} {rhs}", f"{rhs} {op} {lhs}")
            else:
                new_op = random.choice([o for o in ["+", "-", "*"] if o != op])
                corrupted = line.replace(f"{lhs} {op} {rhs}", f"{lhs} {new_op} {rhs}")
            corrupted_lines.append(corrupted)
        else:
            corrupted_lines.append(line)

    return "\n".join(corrupted_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_corrupted{ext}"

    random.seed(args.seed)

    with open(args.input) as f:
        pairs = json.load(f)

    corrupted_pairs = []
    n_changed = 0
    for pair in pairs:
        corrupted = deepcopy(pair)
        new_chosen = corrupt_pemdas_reasoning(pair["chosen"])
        if new_chosen != pair["chosen"]:
            corrupted["chosen"] = new_chosen
            n_changed += 1
        corrupted_pairs.append(corrupted)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(corrupted_pairs, f, ensure_ascii=False, indent=2)

    print(f"Input: {len(pairs)} pairs")
    print(f"Changed: {n_changed} ({n_changed/len(pairs)*100:.0f}%)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
