"""Agent 2: baseline compression sweep, including LLMLingua-2 (the bar).

Strategies: none (rate 0 anchor), random_drop, stopword_drop, llmlingua2.
Sweep each compressing strategy over config rates. Cached; cost-guarded.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness import CONFIG, MODEL, run_qa
from compressors import get_compressor

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RATES = CONFIG["rates"]
COMPRESSING = ["random_drop", "stopword_drop", "llmlingua2"]
RATE0_MEAN_IN = 1583  # measured at rate 0 (smoke)


def load_sample():
    with open(ROOT / "data" / "sample.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def est_input_tokens(rate: float) -> float:
    return RATE0_MEAN_IN * max(0.05, 1 - rate) + 40


def conservative_cost(n_none: int, n_comp_by_rate: dict[float, int]) -> float:
    price = CONFIG["pricing_per_mtok"][MODEL]
    cost = n_none * (RATE0_MEAN_IN / 1e6 * price["input"] + 8 / 1e6 * price["output"])
    for rate, n in n_comp_by_rate.items():
        cost += n * (est_input_tokens(rate) / 1e6 * price["input"] + 8 / 1e6 * price["output"])
    return cost


def precompute_llmlingua(problems):
    """Run LLMLingua-2 compressions serially (cached to disk) before threading."""
    from compressors import llmlingua2
    total = len(problems) * len(RATES)
    print(f"precomputing LLMLingua-2 compressions ({total}) ...")
    done = 0
    for r in RATES:
        for p in problems:
            llmlingua2(p, r)  # cached internally
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  llmlingua ... {done}/{total}")


def run_sweep(problems):
    jobs = [("none", 0.0, p) for p in problems]
    for strat in COMPRESSING:
        for r in RATES:
            for p in problems:
                jobs.append((strat, r, p))

    n_none = len(problems)
    n_comp = defaultdict(int)
    for strat, r, _ in jobs:
        if strat != "none":
            n_comp[r] += 1
    est = conservative_cost(n_none, n_comp)
    ceiling = CONFIG["cost_ceiling_usd"]
    print(f"sweep jobs: {len(jobs)}  (conservative all-uncached cost ${est:.2f}, ceiling ${ceiling:.2f})")
    if est > ceiling:
        sys.exit(f"ABORT: estimated ${est:.2f} exceeds ceiling ${ceiling:.2f}")

    # LLMLingua needs serial precompute (model forward); others are pure-python
    precompute_llmlingua(problems)

    recs = []
    workers = CONFIG.get("concurrency", 4)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(run_qa, p, r, strategy=strat, compress_fn=get_compressor(strat)): (strat, r)
            for (strat, r, p) in jobs
        }
        for fut in as_completed(futs):
            recs.append(fut.result())
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  qa ... {done}/{len(jobs)}")
    return recs


def summarize(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["strategy"], r["rate"])].append(r)
    summary = {}
    for (strat, rate), rs in groups.items():
        n = len(rs)
        summary[f"{strat}@{rate}"] = {
            "strategy": strat, "rate": rate, "n": n,
            "accuracy": sum(x["correct"] for x in rs) / n,
            "mean_f1": sum(x["f1"] for x in rs) / n,
            "mean_input_tokens": sum(x["input_tokens"] for x in rs) / n,
        }
    return summary


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    none_pt = summary["none@0.0"]
    anchor = (none_pt["mean_input_tokens"], none_pt["accuracy"])
    colors = {"random_drop": "#7f7f7f", "stopword_drop": "#1f77b4", "llmlingua2": "#d62728"}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(*anchor, s=120, marker="P", color="#2ca02c", zorder=5, label="none (full context)")
    for strat in COMPRESSING:
        pts = [(summary[f"{strat}@{r}"]["mean_input_tokens"], summary[f"{strat}@{r}"]["accuracy"]) for r in RATES]
        xs = [anchor[0]] + [p[0] for p in pts]
        ys = [anchor[1]] + [p[1] for p in pts]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order], "o-",
                color=colors[strat], label=strat, zorder=3)
    ax.set_xlabel("mean input tokens / question")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Baseline compression Pareto ({MODEL}, HotpotQA)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_pareto.png", dpi=130)
    print(f"  wrote {RESULTS / 'baseline_pareto.png'}")


def main():
    problems = load_sample()
    recs = run_sweep(problems)

    with open(RESULTS / "baseline_runs.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summary = summarize(recs)
    with open(RESULTS / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BASELINE SUMMARY ===")
    print(f"  none@0.0: acc={summary['none@0.0']['accuracy']:.3f} "
          f"mean_in={summary['none@0.0']['mean_input_tokens']:.0f}")
    for strat in COMPRESSING:
        for r in RATES:
            s = summary[f"{strat}@{r}"]
            print(f"  {strat:<14} r={r}: acc={s['accuracy']:.3f} f1={s['mean_f1']:.3f} "
                  f"mean_in={s['mean_input_tokens']:.0f}")
    print("\n  LLMLINGUA-2 (the bar) accuracy by rate:")
    for r in RATES:
        s = summary[f"llmlingua2@{r}"]
        print(f"    rate {r}: acc={s['accuracy']:.3f} @ {s['mean_input_tokens']:.0f} tokens")

    plot(summary)
    print("\nbaseline sweep complete.")


if __name__ == "__main__":
    main()
