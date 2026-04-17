"""
Analyze evaluation results and generate degradation curve figures.

Usage:
  python3 analyze_results.py --plot
  python3 analyze_results.py --plot --latex
"""
import json
import os
import argparse
from collections import defaultdict

RESULTS_DIR = "../results"
FIGURES_DIR = "../paper/figures"

MODEL_TAGS = ["base", "dpo_multidomain", "dpo_steplevel", "dpo_corrupted"]

DOMAIN_LABELS = {"pemdas": "PEMDAS", "coinflip": "CoinFlip"}

TAG_COLORS = {
    "base": "#1f77b4", "dpo_multidomain": "#2ca02c",
    "dpo_steplevel": "#9467bd", "dpo_corrupted": "#ff7f0e",
}

TAG_LABELS = {
    "base": "Base", "dpo_multidomain": "Multi Domain DPO",
    "dpo_steplevel": "Step Level DPO", "dpo_corrupted": "Incorrect CoT DPO",
}


def load_results(domain, tag):
    path = os.path.join(RESULTS_DIR, f"{domain}_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def accuracy_by_steps(results):
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        key = (r["strategy"], r["steps_to_solve"])
        groups[key]["total"] += 1
        if r["correct"]:
            groups[key]["correct"] += 1
    return {k: v["correct"] / v["total"] * 100 for k, v in groups.items() if v["total"]}


def overall_accuracy(results):
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        groups[r["strategy"]]["total"] += 1
        if r["correct"]:
            groups[r["strategy"]]["correct"] += 1
    return {s: c["correct"] / c["total"] * 100 for s, c in groups.items() if c["total"]}


def degradation(acc_by_steps, strategy):
    steps = sorted([s for (st, s) in acc_by_steps if st == strategy])
    if len(steps) < 3:
        return None
    n = len(steps)
    low = steps[:n // 3]
    high = steps[-(n // 3):]
    low_acc = sum(acc_by_steps[(strategy, s)] for s in low) / len(low)
    high_acc = sum(acc_by_steps[(strategy, s)] for s in high) / len(high)
    return {"low_acc": low_acc, "high_acc": high_acc, "delta": high_acc - low_acc}


def plot_domain(domain, available):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for tag in available:
        results = load_results(domain, tag)
        if not results:
            continue
        acc = accuracy_by_steps(results)
        color = TAG_COLORS.get(tag, "black")
        label = TAG_LABELS.get(tag, tag)
        data = sorted([(s, a) for (st, s), a in acc.items() if st == "zero-shot-cot"])
        if data:
            xs, ys = zip(*data)
            ax.plot(xs, ys, color=color, marker="o", markersize=4, label=f"{label} (CoT)", alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Problem Complexity (steps to solve)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"{DOMAIN_LABELS.get(domain, domain)}: Degradation Curves", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    out = os.path.join(FIGURES_DIR, f"{domain}_all_variants.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    available = [t for t in MODEL_TAGS
                 if any(os.path.exists(os.path.join(RESULTS_DIR, f"{d}_{t}.json")) for d in DOMAIN_LABELS)]

    if not available:
        print("No results found.")
        return

    print(f"Models found: {available}\n")

    for domain in DOMAIN_LABELS:
        print(f"{'='*60}\n  {DOMAIN_LABELS[domain]}\n{'='*60}")
        for tag in available:
            results = load_results(domain, tag)
            if not results:
                continue
            oa = overall_accuracy(results)
            abs_ = accuracy_by_steps(results)
            print(f"\n  {TAG_LABELS.get(tag, tag)}:")
            for strategy in ["direct", "zero-shot-cot"]:
                if strategy not in oa:
                    continue
                deg = degradation(abs_, strategy)
                deg_str = f"  delta={deg['delta']:+.1f}%" if deg else ""
                print(f"    {strategy:20s}  {oa[strategy]:5.1f}%{deg_str}")
        if args.plot:
            plot_domain(domain, available)

    if args.latex:
        print("\n% LaTeX table")
        print("\\begin{tabular}{lrrrr}")
        print("\\toprule")
        print("Model & PEMDAS Acc & $\\Delta$ & CoinFlip Acc & $\\Delta$ \\\\")
        print("\\midrule")
        for tag in available:
            cells = []
            for domain in DOMAIN_LABELS:
                results = load_results(domain, tag)
                if not results:
                    cells.append("-- & --")
                    continue
                oa = overall_accuracy(results)
                abs_ = accuracy_by_steps(results)
                acc = oa.get("zero-shot-cot", 0)
                deg = degradation(abs_, "zero-shot-cot")
                d = f"{deg['delta']:+.1f}" if deg else "--"
                cells.append(f"{acc:.1f}\\% & {d}\\%")
            print(f"{TAG_LABELS.get(tag, tag)} & {' & '.join(cells)} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
