"""Post-hoc symbolic verify-and-correct over saved per-instance eval JSONs."""
import os
import re
import json
import argparse
import glob

PEMDAS_STEP_RE = re.compile(
    r"(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)"
)
_ANSWER_LINE_RE = re.compile(
    r"^\s*answer\s*[:=]\s*(-?\d+\.?\d*)", re.IGNORECASE | re.MULTILINE
)


def correct_pemdas_response(response):
    lines = response.strip().split("\n")
    out = []
    n_steps = 0
    n_step_correct = 0
    for line in lines:
        m = PEMDAS_STEP_RE.search(line)
        if not m:
            out.append(line)
            continue
        expr = f"{m.group(1)} {m.group(2)} {m.group(3)}"
        stated = m.group(4)
        try:
            expected = eval(expr)  # noqa: S307 -- digits + operator only
            expected_str = (
                str(int(expected)) if expected == int(expected) else str(expected)
            )
        except (ValueError, ZeroDivisionError, SyntaxError):
            out.append(line)
            continue
        n_steps += 1
        try:
            if abs(float(stated) - expected) < 0.01:
                n_step_correct += 1
        except ValueError:
            pass
        out.append(line.replace(f"= {stated}", f"= {expected_str}"))
    return "\n".join(out), n_steps, n_step_correct


def extract_first_number(text):
    for line in text.strip().split("\n"):
        for m in re.finditer(r"-?\d+\.?\d*", line):
            return m.group()
    return None


def normalize_answer(s):
    # mirror evaluate_benchmarks.parse_answer_pemdas: float -> int-truncate, strip+lower
    if s is None:
        return None
    s = str(s).strip()
    if "." in s:
        try:
            return str(int(float(s)))
        except ValueError:
            return s.lower()
    return s.lower()


def extract_answer_line(text):
    # last "Answer: X" wins; decomposed CoT ends with one but the eval harness
    # stops at the first numeric line and misses it
    matches = _ANSWER_LINE_RE.findall(text)
    return matches[-1] if matches else None


def process_file(path):
    with open(path) as f:
        records = json.load(f)
    records = [r for r in records
               if r.get("strategy") == "zero-shot-cot" and r.get("domain") == "pemdas"]
    n = len(records)
    if n == 0:
        return None

    n_correct = 0
    n_correct_after = 0
    n_has_step = 0
    n_has_answer_line = 0
    n_correct_answer_aware = 0
    n_changed = 0

    for r in records:
        resp = r.get("raw_response", "")
        gt = normalize_answer(r.get("ground_truth"))

        if r.get("correct"):
            n_correct += 1

        ans_line = extract_answer_line(resp)
        if ans_line is not None:
            n_has_answer_line += 1
            if normalize_answer(ans_line) == gt:
                n_correct_answer_aware += 1
        elif r.get("correct"):
            n_correct_answer_aware += 1

        corrected_resp, n_steps, _ = correct_pemdas_response(resp)
        if n_steps > 0:
            n_has_step += 1

        # parse_answer_pemdas: last number on the first nonempty line
        new_answer = None
        for line in corrected_resp.strip().split("\n"):
            if not line.strip():
                continue
            last = None
            for m in re.finditer(r"-?\d+\.?\d*", line):
                last = m.group()
            if last is not None:
                new_answer = last
                break
        if normalize_answer(new_answer) == gt:
            n_correct_after += 1
        if corrected_resp != resp:
            n_changed += 1

    return {
        "n": n,
        "model_acc": round(100 * n_correct / n, 1),
        "has_step_share_pct": round(100 * n_has_step / n, 1),
        "has_answer_line_pct": round(100 * n_has_answer_line / n, 1),
        "answer_aware_acc": round(100 * n_correct_answer_aware / n, 1),
        "verify_and_correct_acc": round(100 * n_correct_after / n, 1),
        "responses_changed": n_changed,
    }


def derive_tag(path):
    # e.g. results_llama31/pemdas_dpo_cpo.json -> llama31_dpo_cpo
    base = os.path.basename(path).replace(".json", "")
    if base.startswith("pemdas_"):
        base = base[len("pemdas_"):]
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent == "results_llama31" and not base.startswith("llama31"):
        base = "llama31_" + base
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = []
    for pat in args.inputs:
        if any(c in pat for c in "*?["):
            files.extend(glob.glob(pat))
        else:
            files.append(pat)
    files = sorted(set(files))

    summary = {}
    for path in files:
        if not os.path.exists(path):
            print(f"  [skip] {path}: not found")
            continue
        s = process_file(path)
        if s is None:
            print(f"  [skip] {path}: no PEMDAS zero-shot-cot records")
            continue
        summary[derive_tag(path)] = s

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{len(summary)} variants -> {args.out}\n")
    print(f"{'variant':30s} {'n':>4} {'cot':>7} {'step%':>7} "
          f"{'ans%':>7} {'ans-aware':>10} {'v&c':>7} {'changed':>8}")
    for tag, s in summary.items():
        print(f"{tag:30s} {s['n']:4d} "
              f"{s['model_acc']:6.1f}% {s['has_step_share_pct']:6.1f}% "
              f"{s['has_answer_line_pct']:6.1f}% {s['answer_aware_acc']:9.1f}% "
              f"{s['verify_and_correct_acc']:6.1f}% {s['responses_changed']:8d}")


if __name__ == "__main__":
    main()
