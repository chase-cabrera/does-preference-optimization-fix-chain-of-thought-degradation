"""Degradation curves, trace-correctness bars, and LaTeX row dumps from eval JSONs."""
import json
import os
import argparse
from collections import defaultdict

FIGURES_DIR = "../paper/figures"
DOMAIN_LABELS = {"pemdas": "PEMDAS", "coinflip": "CoinFlip"}
MODEL_TAGS = ["base", "dpo_multidomain", "dpo_steplevel",
              "dpo_synthetic", "dpo_corrupted_v2", "dpo_cpo"]

TAG_COLORS = {
    "base": "#1f77b4", "dpo_multidomain": "#2ca02c",
    "dpo_steplevel": "#9467bd", "dpo_synthetic": "#d62728",
    "dpo_corrupted_v2": "#ff7f0e", "dpo_cpo": "#17becf",
}
TAG_LABELS = {
    "base": "Base", "dpo_multidomain": "Multi-Domain DPO",
    "dpo_steplevel": "Step-Level DPO", "dpo_synthetic": "Synthetic-CoT DPO",
    "dpo_corrupted_v2": "Incorrect-CoT DPO", "dpo_cpo": "CPO",
}

# (results_dir, file_pattern, model_label, figure_suffix)
MODEL_CONFIGS = {
    "llama2":  ("../results",          "{domain}_{tag}.json",          "LLaMA-2-7B",  "all_variants"),
    "llama31": ("../results_llama31",  "{domain}_llama31_{tag}.json",  "Llama 3.1-8B", "llama31_variants"),
}


def load_results(results_dir, pattern, domain, tag):
    path = os.path.join(results_dir, pattern.format(domain=domain, tag=tag))
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


def faithfulness_counts(results):
    cot = [r for r in results if r["strategy"] == "zero-shot-cot"]
    total = len(cot)
    faithful = sum(1 for r in cot if r.get("faithful", False))
    wrong_steps_right_answer = sum(1 for r in cot if r["correct"] and not r.get("faithful", False))
    right_steps_wrong_answer = sum(1 for r in cot if not r["correct"] and r.get("all_steps_correct", False))
    wrong_both = total - faithful - wrong_steps_right_answer - right_steps_wrong_answer
    return {
        "total": total,
        "faithful": faithful,
        "wrong_steps_right_answer": wrong_steps_right_answer,
        "right_steps_wrong_answer": right_steps_wrong_answer,
        "wrong_both": wrong_both,
    }


def plot_domain(domain, available_tags, results_dir, pattern, model_label, fig_suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for tag in available_tags:
        results = load_results(results_dir, pattern, domain, tag)
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
    ax.set_title(f"{model_label} {DOMAIN_LABELS.get(domain, domain)}: Degradation Curves", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    out = os.path.join(FIGURES_DIR, f"{domain}_{fig_suffix}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_faithfulness():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(FIGURES_DIR, exist_ok=True)

    categories = ["Trace-correct", "Wrong-trace, right answer",
                   "Right steps, wrong answer", "Wrong-trace, wrong answer"]
    faith_tags = ["base", "dpo_multidomain", "dpo_synthetic", "dpo_corrupted_v2", "dpo_cpo"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, (model_key, (results_dir, pattern, model_label, _)) in zip(axes, MODEL_CONFIGS.items()):
        x = np.arange(len(categories))
        width = 0.18
        offsets = np.arange(len(faith_tags)) - (len(faith_tags) - 1) / 2

        for i, tag in enumerate(faith_tags):
            results = load_results(results_dir, pattern, "pemdas", tag)
            if not results:
                continue
            fc = faithfulness_counts(results)
            total = fc["total"]
            vals = [
                fc["faithful"] / total * 100,
                fc["wrong_steps_right_answer"] / total * 100,
                fc["right_steps_wrong_answer"] / total * 100,
                fc["wrong_both"] / total * 100,
            ]
            color = TAG_COLORS.get(tag, "black")
            label = TAG_LABELS.get(tag, tag)
            bars = ax.bar(x + offsets[i] * width, vals, width, label=label, color=color, alpha=0.8)
            for bar, v in zip(bars, vals):
                if v > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel("Percentage of Instances (%)", fontsize=11)
        ax.set_title(f"{model_label}: PEMDAS Trace-Correctness Breakdown (CoT)", fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "faithfulness_breakdown.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    for model_key, (results_dir, pattern, model_label, fig_suffix) in MODEL_CONFIGS.items():
        available = [t for t in MODEL_TAGS
                     if any(os.path.exists(os.path.join(results_dir, pattern.format(domain=d, tag=t)))
                            for d in DOMAIN_LABELS)]
        if not available:
            print(f"No results found for {model_label}.")
            continue

        print(f"\n== {model_label} ({', '.join(available)}) ==")

        for domain in DOMAIN_LABELS:
            print(f"\n[{DOMAIN_LABELS[domain]}]")
            for tag in available:
                results = load_results(results_dir, pattern, domain, tag)
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
                plot_domain(domain, available, results_dir, pattern, model_label, fig_suffix)

    if args.plot:
        plot_faithfulness()

    if args.latex:
        print("\n% LaTeX table")
        for model_key, (results_dir, pattern, model_label, _) in MODEL_CONFIGS.items():
            print(f"\n% {model_label}")
            for tag in MODEL_TAGS:
                cells = []
                for domain in DOMAIN_LABELS:
                    results = load_results(results_dir, pattern, domain, tag)
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


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
