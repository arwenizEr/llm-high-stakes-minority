"""Cost calibration: measure real mean output tokens per budget on a small
probe, report actual spend so far, and recommend the largest sample size that
fits a target USD budget. Cheap (~a dozen calls)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from harness import CONFIG, MODEL, run_problem, CACHE_DIR

ROOT = Path(__file__).resolve().parent
PRICE = CONFIG["pricing_per_mtok"][MODEL]
BUDGETS = ["low", "medium", "high"]
PROBE_N = 8
TARGET_USD = 11.0   # leave margin under the $13 cap


def cost_of(rec) -> float:
    return (rec["input_tokens"] / 1e6) * PRICE["input"] + (
        rec["output_tokens"] / 1e6
    ) * PRICE["output"]


def spend_so_far() -> float:
    total = 0.0
    for fp in CACHE_DIR.glob("*.json"):
        try:
            rec = json.loads(fp.read_text())
            total += cost_of(rec)
        except Exception:
            pass
    return total


def main():
    spent = spend_so_far()
    print(f"actual spend banked in cache so far: ${spent:.3f}")

    with open(ROOT / "data" / "sample.jsonl", encoding="utf-8") as f:
        sample = [json.loads(line) for line in f]
    probe = sample[:PROBE_N]

    means = {}
    print(f"\nprobe: {PROBE_N} problems x 3 budgets")
    for b in BUDGETS:
        recs = [run_problem(p, b) for p in probe]
        mo = sum(r["output_tokens"] for r in recs) / len(recs)
        mi = sum(r["input_tokens"] for r in recs) / len(recs)
        acc = sum(r["correct"] for r in recs) / len(recs)
        means[b] = (mi, mo)
        print(f"  {b:<7} mean_in={mi:.0f} mean_out={mo:.0f} acc={acc:.2f}")

    per_problem_all3 = sum(
        (mi / 1e6) * PRICE["input"] + (mo / 1e6) * PRICE["output"]
        for (mi, mo) in means.values()
    )
    print(f"\nmeasured cost per problem (all 3 budgets): ${per_problem_all3:.4f}")

    remaining = TARGET_USD - spent
    # the probe's 8 problems are already cached, so they cost 0 going forward
    n_fit = PROBE_N + int(remaining / per_problem_all3) if per_problem_all3 > 0 else 0
    n_fit = min(n_fit, len(sample))
    proj_full = (len(sample) - PROBE_N) * per_problem_all3
    print(f"remaining budget toward target ${TARGET_USD}: ${remaining:.2f}")
    print(f"projected cost to finish all {len(sample)}: ${proj_full:.2f}")
    print(f"=> RECOMMENDED sample size that fits: {n_fit} problems")


if __name__ == "__main__":
    main()
