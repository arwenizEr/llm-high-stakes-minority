"""Agent 2: fixed-budget baseline sweep.

Runs all sampled problems at each of three budgets (low/medium/high), fully
cached. Produces the Pareto baseline that the router is measured against.

Cost guard: estimates spend on the *uncached* calls before running, prints it,
and refuses to start if it exceeds the configurable ceiling. A hard cap on
total API calls is also enforced.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness import CONFIG, MODEL, BUDGETS, run_problem, _cache_path, _prompt_hash

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

BUDGET_ORDER = ["low", "medium", "high"]
HARD_CALL_CAP = 700          # backstop against runaway
# per-budget output-token estimates for the cost guard, calibrated from a probe
# (measured low/med/high mean_out = 144/315/540) plus margin for harder problems
EST_OUTPUT_TOKENS = {"low": 350, "medium": 700, "high": 1200}
EST_INPUT_TOKENS = 150


def load_sample() -> list[dict]:
    with open(ROOT / "data" / "sample.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_cached(problem: dict, budget: str) -> bool:
    phash = _prompt_hash(problem["problem"], budget)
    return _cache_path(problem["id"], budget, phash).exists()


def estimate_cost(jobs: list[tuple[dict, str]]) -> tuple[float, int]:
    """Return (estimated_usd, n_uncached) for the uncached jobs."""
    price = CONFIG["pricing_per_mtok"][MODEL]
    uncached = [(p, b) for (p, b) in jobs if not is_cached(p, b)]
    cost = 0.0
    for _, b in uncached:
        cost += (EST_INPUT_TOKENS / 1e6) * price["input"]
        cost += (EST_OUTPUT_TOKENS[b] / 1e6) * price["output"]
    return cost, len(uncached)


def run_sweep(sample: list[dict]) -> list[dict]:
    jobs = [(p, b) for b in BUDGET_ORDER for p in sample]

    est_cost, n_uncached = estimate_cost(jobs)
    ceiling = CONFIG["cost_ceiling_usd"]
    print(f"sweep: {len(jobs)} jobs ({len(sample)} problems x {len(BUDGET_ORDER)} budgets)")
    print(f"  uncached (will hit API): {n_uncached}")
    print(f"  estimated cost of uncached calls: ${est_cost:.2f} (ceiling ${ceiling:.2f})")
    print(f"  model={MODEL}")

    if n_uncached > HARD_CALL_CAP:
        sys.exit(f"ABORT: {n_uncached} uncached calls exceeds hard cap {HARD_CALL_CAP}")
    if est_cost > ceiling:
        sys.exit(f"ABORT: estimated ${est_cost:.2f} exceeds ceiling ${ceiling:.2f}")

    recs: list[dict] = []
    workers = CONFIG.get("concurrency", 4)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_problem, p, b): (p["id"], b) for (p, b) in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            recs.append(rec)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)}")
    return recs


def summarize(recs: list[dict]) -> dict:
    by_budget: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_budget[r["budget"]].append(r)

    summary = {"model": MODEL, "budgets": {}}
    for b in BUDGET_ORDER:
        rs = by_budget[b]
        n = len(rs)
        acc = sum(r["correct"] for r in rs) / n
        mean_out = sum(r["output_tokens"] for r in rs) / n
        mean_in = sum(r["input_tokens"] for r in rs) / n
        by_level = {}
        lvl_groups: dict[int, list[dict]] = defaultdict(list)
        for r in rs:
            lvl_groups[r["level"]].append(r)
        for lvl in sorted(lvl_groups):
            g = lvl_groups[lvl]
            by_level[lvl] = {
                "accuracy": sum(x["correct"] for x in g) / len(g),
                "mean_output_tokens": sum(x["output_tokens"] for x in g) / len(g),
                "n": len(g),
            }
        summary["budgets"][b] = {
            "n": n,
            "accuracy": acc,
            "mean_output_tokens": mean_out,
            "mean_input_tokens": mean_in,
            "accuracy_by_level": by_level,
        }
    return summary


def plot_pareto(summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [summary["budgets"][b]["mean_output_tokens"] for b in BUDGET_ORDER]
    ys = [summary["budgets"][b]["accuracy"] for b in BUDGET_ORDER]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", color="#444", zorder=2)
    for b, x, y in zip(BUDGET_ORDER, xs, ys):
        ax.annotate(b, (x, y), textcoords="offset points", xytext=(8, -4))
    ax.set_xlabel("mean output tokens / problem")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Fixed-budget baselines ({MODEL})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_pareto.png", dpi=130)
    print(f"  wrote {RESULTS / 'baseline_pareto.png'}")


def main():
    sample = load_sample()
    recs = run_sweep(sample)

    with open(RESULTS / "baseline_runs.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    summary = summarize(recs)
    with open(RESULTS / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BASELINE SUMMARY ===")
    for b in BUDGET_ORDER:
        s = summary["budgets"][b]
        print(f"  {b:<7} acc={s['accuracy']:.3f}  mean_out_tokens={s['mean_output_tokens']:.0f}")

    # savings ceiling signal: how much of the easy set stays correct at low budget
    lvl1_low = summary["budgets"]["low"]["accuracy_by_level"].get(1, {}).get("accuracy")
    lvl1_high = summary["budgets"]["high"]["accuracy_by_level"].get(1, {}).get("accuracy")
    print("\n  SAVINGS SIGNAL (the gap that bounds router savings):")
    if lvl1_low is not None and lvl1_high is not None:
        print(f"    level-1 problems: {lvl1_low*100:.0f}% correct at LOW vs {lvl1_high*100:.0f}% at HIGH")
    for lvl in range(1, 6):
        lo = summary["budgets"]["low"]["accuracy_by_level"].get(lvl, {}).get("accuracy")
        hi = summary["budgets"]["high"]["accuracy_by_level"].get(lvl, {}).get("accuracy")
        lot = summary["budgets"]["low"]["accuracy_by_level"].get(lvl, {}).get("mean_output_tokens")
        hit = summary["budgets"]["high"]["accuracy_by_level"].get(lvl, {}).get("mean_output_tokens")
        if lo is not None:
            print(f"    L{lvl}: low acc={lo:.2f} ({lot:.0f} tok)  high acc={hi:.2f} ({hit:.0f} tok)")

    plot_pareto(summary)
    print("\nbaseline sweep complete.")


if __name__ == "__main__":
    main()
