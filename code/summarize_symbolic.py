"""
Aggregate per-instance symbolic_verifier.py outputs into a single summary
JSON suitable for Table 3 in the paper. Replaces the prior post-hoc
summary that hard-coded PEMDAS symbolic accuracy to 100%.

For PEMDAS we report:
  - n              : number of instances
  - model_acc      : zero-shot CoT accuracy (matches Table 1)
  - faithful       : count of fully-faithful traces (all_steps_correct AND answer_correct)
  - sym_corrected  : accuracy after replacing each parseable arithmetic step's
                     stated result with the symbolic result. This is an upper
                     bound on what symbolic post-hoc patching can recover
                     *given the model's own decomposition*; it is NOT 100%
                     because the model frequently fails to decompose at all,
                     omits the final combination step, or emits non-canonical
                     traces that the regex cannot patch.

For CoinFlip we report:
  - n, model_acc, sym_acc (= 100% by construction; sanity check only)

Usage (from this directory, i.e. code/):
  python3 summarize_symbolic.py --results-dir ../results --out ../results/symbolic_verification_table.json
  python3 summarize_symbolic.py --results-dir ../results_llama31 --out ../results_llama31/symbolic_verification_table.json
"""
import os
import json
import argparse
import glob


def summarize_pemdas(records):
    n = len(records)
    if n == 0:
        return None
    model_correct = sum(1 for r in records if r.get("model_correct"))
    faithful = sum(1 for r in records if r.get("faithful"))
    has_corrected = any("corrected_correct" in r for r in records)
    out = {
        "domain": "pemdas",
        "n": n,
        "model_acc": round(model_correct / n * 100, 1),
        "faithful": faithful,
        "faithful_pct": round(faithful / n * 100, 2),
    }
    if has_corrected:
        sym_corr = sum(1 for r in records if r.get("corrected_correct"))
        out["sym_corrected_n"] = sym_corr
        out["sym_corrected_acc"] = round(sym_corr / n * 100, 1)
    return out


def summarize_coinflip(records):
    n = len(records)
    if n == 0:
        return None
    model_correct = sum(1 for r in records if r.get("model_correct"))
    sym_correct = sum(1 for r in records if r.get("corrected_correct", r.get("symbolic_answer") == r.get("ground_truth")))
    return {
        "domain": "coinflip",
        "n": n,
        "model_acc": round(model_correct / n * 100, 1),
        "sym_acc": round(sym_correct / n * 100, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    summary = {}
    pattern = os.path.join(args.results_dir, "*_verify_and_correct.json")
    files = sorted(glob.glob(pattern))
    if not files:
        # Fall back to verify_only
        pattern = os.path.join(args.results_dir, "*_verify_only.json")
        files = sorted(glob.glob(pattern))

    for path in files:
        name = os.path.basename(path).replace(".json", "")
        with open(path) as f:
            records = json.load(f)

        # Group by domain
        by_domain = {}
        for r in records:
            by_domain.setdefault(r["domain"], []).append(r)

        for domain, recs in by_domain.items():
            if domain == "pemdas":
                s = summarize_pemdas(recs)
            elif domain == "coinflip":
                s = summarize_coinflip(recs)
            else:
                continue
            if s is not None:
                # Use just the model_tag, not the verify suffix
                tag = recs[0].get("model_tag", name)
                summary[f"{tag}_{domain}"] = s

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary for {len(summary)} (variant, domain) pairs to {args.out}")
    for k, v in summary.items():
        if v["domain"] == "pemdas":
            extra = f"  sym_corrected={v.get('sym_corrected_acc', 'N/A')}%" \
                    if "sym_corrected_acc" in v else ""
            print(f"  {k:50s}  acc={v['model_acc']:5.1f}%  faithful={v['faithful']:3d}/{v['n']}{extra}")
        else:
            print(f"  {k:50s}  acc={v['model_acc']:5.1f}%  sym_acc={v['sym_acc']:5.1f}%")


if __name__ == "__main__":
    main()
