"""Agent 4: router + end-to-end evaluation.

For each problem the predictor chooses a budget, then the harness runs at that
budget (cached). Because the baseline sweep already ran every problem at every
budget, router evaluation reuses those cached results and makes ZERO new API
calls.

Evaluated on the held-out TEST split (problems the learned predictor never
trained on). Produces final_pareto.png and RESULTS.md with the headline:
tokens saved vs. always-high at matched accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness import run_problem
from predictor import (
    get_predictor,
    load_baseline_correctness,
    oracle_budget,
    BUDGETS,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_problems_by_id() -> dict[str, dict]:
    with open(ROOT / "data" / "sample.jsonl", encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def load_test_problems() -> list[dict]:
    with open(RESULTS / "split.json") as f:
        test_ids = json.load(f)["test_ids"]
    by_id = load_problems_by_id()
    return [by_id[i] for i in test_ids]


def eval_point(problems: list[dict], choose_budget) -> dict:
    """choose_budget(problem) -> budget. Returns accuracy + mean output tokens."""
    n = len(problems)
    correct = 0
    out_tokens = 0
    budget_hist = {b: 0 for b in BUDGETS}
    for p in problems:
        b = choose_budget(p)
        budget_hist[b] += 1
        rec = run_problem(p, b)  # cached
        correct += rec["correct"]
        out_tokens += rec["output_tokens"]
    return {
        "accuracy": correct / n,
        "mean_output_tokens": out_tokens / n,
        "n": n,
        "budget_histogram": budget_hist,
    }


def baseline_points(problems: list[dict]) -> dict[str, dict]:
    pts = {}
    for b in BUDGETS:
        pts[b] = eval_point(problems, lambda p, b=b: b)
    return pts


def interp_baseline_accuracy(baselines: dict[str, dict], x: float) -> float:
    """Linear interpolation of the baseline accuracy-vs-tokens frontier at token
    cost x (used to test whether a router point sits above the baseline line)."""
    pts = sorted(
        ((baselines[b]["mean_output_tokens"], baselines[b]["accuracy"]) for b in BUDGETS),
        key=lambda t: t[0],
    )
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def plot_final(baselines, routers, oracle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    bx = [baselines[b]["mean_output_tokens"] for b in BUDGETS]
    by = [baselines[b]["accuracy"] for b in BUDGETS]
    ax.plot(bx, by, "o-", color="#555", label="fixed-budget baselines", zorder=2)
    for b, x, y in zip(BUDGETS, bx, by):
        ax.annotate(b, (x, y), textcoords="offset points", xytext=(8, -10), color="#555")

    colors = {"heuristic": "#1f77b4", "learned": "#d62728"}
    for name, pt in routers.items():
        ax.scatter(
            pt["mean_output_tokens"], pt["accuracy"], s=110, marker="*",
            color=colors.get(name, "#2ca02c"), zorder=4, label=f"router ({name})"
        )

    if oracle:
        ax.scatter(
            oracle["mean_output_tokens"], oracle["accuracy"], s=130, marker="P",
            color="#2ca02c", zorder=4, label="oracle (post-hoc min budget)"
        )

    ax.set_xlabel("mean output tokens / problem")
    ax.set_ylabel("accuracy")
    ax.set_title("Adaptive reasoning-budget router vs fixed baselines (held-out test)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "final_pareto.png", dpi=130)
    print(f"  wrote {RESULTS / 'final_pareto.png'}")


def main():
    test_problems = load_test_problems()
    print(f"held-out test problems: {len(test_problems)}")

    baselines = baseline_points(test_problems)

    heur = get_predictor("heuristic")
    learned = get_predictor("learned")
    routers = {
        "heuristic": eval_point(test_problems, heur.predict),
        "learned": eval_point(test_problems, learned.predict),
    }

    correctness = load_baseline_correctness()
    oracle = eval_point(
        test_problems,
        lambda p: oracle_budget(correctness.get(p["id"], {})),
    )

    high = baselines["high"]
    report = {"baselines": baselines, "routers": routers, "oracle": oracle}

    # headline numbers vs always-high
    lines = []
    for name, pt in routers.items():
        tok_saved = high["mean_output_tokens"] - pt["mean_output_tokens"]
        tok_saved_pct = 100 * tok_saved / high["mean_output_tokens"]
        acc_lost = high["accuracy"] - pt["accuracy"]
        base_at_x = interp_baseline_accuracy(baselines, pt["mean_output_tokens"])
        above = pt["accuracy"] - base_at_x
        report[f"{name}_vs_high"] = {
            "tokens_saved_per_problem": tok_saved,
            "tokens_saved_pct": tok_saved_pct,
            "accuracy_lost_vs_high": acc_lost,
            "baseline_accuracy_at_same_tokens": base_at_x,
            "router_accuracy_above_baseline_line": above,
        }
        lines.append((name, pt, tok_saved_pct, acc_lost, above))

    with open(RESULTS / "router_eval.json", "w") as f:
        json.dump(report, f, indent=2)

    plot_final(baselines, routers, oracle)

    # console
    print("\n=== ROUTER EVAL (held-out test) ===")
    print(f"  {'point':<22} acc      mean_out_tokens")
    for b in BUDGETS:
        s = baselines[b]
        print(f"  baseline:{b:<13} {s['accuracy']:.3f}    {s['mean_output_tokens']:.0f}")
    for name, pt in routers.items():
        print(f"  router:{name:<15} {pt['accuracy']:.3f}    {pt['mean_output_tokens']:.0f}  hist={pt['budget_histogram']}")
    print(f"  oracle{'':<16} {oracle['accuracy']:.3f}    {oracle['mean_output_tokens']:.0f}")

    write_results_md(baselines, routers, oracle, report)
    print(f"\n  wrote {RESULTS / 'RESULTS.md'}")


def write_results_md(baselines, routers, oracle, report):
    high = baselines["high"]
    low = baselines["low"]
    lines = []
    lines.append("# Adaptive Reasoning-Budget Router — Results\n")
    lines.append(f"Held-out test split: **{routers['learned']['n']} problems**. "
                 f"Router evaluation reused cached baseline calls (zero new API spend).\n")
    lines.append("## Pareto points (accuracy vs mean output tokens/problem)\n")
    lines.append("| point | accuracy | mean output tokens |")
    lines.append("|---|---|---|")
    for b in BUDGETS:
        s = baselines[b]
        lines.append(f"| baseline: {b} | {s['accuracy']:.3f} | {s['mean_output_tokens']:.0f} |")
    for name, pt in routers.items():
        lines.append(f"| **router: {name}** | {pt['accuracy']:.3f} | {pt['mean_output_tokens']:.0f} |")
    lines.append(f"| oracle (post-hoc) | {oracle['accuracy']:.3f} | {oracle['mean_output_tokens']:.0f} |")
    lines.append("")

    lines.append("## Headline: router vs always-high\n")
    for name in routers:
        v = report[f"{name}_vs_high"]
        verdict = "ABOVE/LEFT of" if v["router_accuracy_above_baseline_line"] >= -1e-9 else "below"
        lines.append(f"### {name} router")
        lines.append(f"- tokens saved vs always-high: **{v['tokens_saved_pct']:.1f}%** "
                     f"({v['tokens_saved_per_problem']:.0f} tokens/problem)")
        lines.append(f"- accuracy lost vs always-high: **{v['accuracy_lost_vs_high']:+.3f}**")
        lines.append(f"- baseline accuracy at the router's token cost: {v['baseline_accuracy_at_same_tokens']:.3f}; "
                     f"router accuracy: {routers[name]['accuracy']:.3f} "
                     f"(**{v['router_accuracy_above_baseline_line']:+.3f}** vs the baseline line)")
        lines.append(f"- **Verdict: the {name} router sits {verdict} the baseline frontier.**\n")

    # overall verdict
    best = max(routers, key=lambda n: report[f"{n}_vs_high"]["router_accuracy_above_baseline_line"])
    bv = report[f"{best}_vs_high"]
    lines.append("## Verdict\n")
    if bv["router_accuracy_above_baseline_line"] >= -1e-9:
        lines.append(
            f"**Yes.** The {best} router beats the fixed-budget Pareto frontier: at its "
            f"mean cost of {routers[best]['mean_output_tokens']:.0f} output tokens/problem it reaches "
            f"{routers[best]['accuracy']:.3f} accuracy, vs {bv['baseline_accuracy_at_same_tokens']:.3f} "
            f"for the interpolated baseline at the same cost — and it recovers most of the "
            f"always-high accuracy ({high['accuracy']:.3f}) while saving "
            f"{bv['tokens_saved_pct']:.1f}% of the tokens.")
    else:
        lines.append(
            f"**No (negative result).** The {best} router point sits below the baseline "
            f"line. Likely cause: the predictor's budget choices do not align with the "
            f"per-problem accuracy gains — note the baseline gap itself "
            f"(low acc {low['accuracy']:.3f} vs high acc {high['accuracy']:.3f}); if that gap is "
            f"small, there is little token to save without losing accuracy.")
    lines.append("")
    lines.append(f"Oracle ceiling: {oracle['accuracy']:.3f} accuracy at "
                 f"{oracle['mean_output_tokens']:.0f} tokens/problem — the best any predictor "
                 f"using these three budgets could do on this split.")
    (RESULTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
