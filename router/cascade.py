"""Confidence-triggered cascade.

Pivot from upfront difficulty prediction (which failed on text alone) to a
cascade: answer at low budget, escalate to high only when the low result looks
unreliable. Trigger = self-consistency of 3 low samples.

Parts:
  A. Oracle cascade ceiling (FREE, from cached low/high): run low, escalate to
     high iff low is wrong (peeks at gold). This is the cascade's ceiling.
  B. Text-only heuristic router number (FREE): the level-free heuristic, to see
     how much of the 22% win survives without the privileged `level` label.
  C. Real cascade (low-budget API calls only, cost-guarded): 3 low samples per
     problem; escalate if they disagree or no valid answer.
  D. Trigger quality: recall (escalated | low wrong) and precision (low wrong |
     escalated).

Same rules: cache everything, cost-guard, sympy checker, log config.
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness import CONFIG, MODEL, run_problem, CACHE_DIR
from checker import is_correct, extract_answer
from predictor import get_predictor, BUDGETS
from router import (
    load_test_problems,
    load_problems_by_id,
    baseline_points,
    eval_point,
    interp_baseline_accuracy,
    load_baseline_correctness,
    oracle_budget,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PRICE = CONFIG["pricing_per_mtok"][MODEL]
N_SAMPLES = 3
TARGET_USD = 13.0
MARGIN = 1.0


def spend_so_far() -> float:
    tot = 0.0
    for fp in CACHE_DIR.glob("*.json"):
        try:
            r = json.loads(fp.read_text())
            tot += r["input_tokens"] / 1e6 * PRICE["input"] + r["output_tokens"] / 1e6 * PRICE["output"]
        except Exception:
            pass
    return tot


def load_all_problems() -> list[dict]:
    with open(ROOT / "data" / "sample.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ----------------------------------------------------------------------------
# Part A: oracle cascade ceiling (free)
# ----------------------------------------------------------------------------

def oracle_cascade(problems: list[dict], baseline_runs: dict) -> dict:
    """run low; escalate to high iff low wrong (uses gold). Ceiling."""
    n = len(problems)
    correct = tokens = escal = 0
    for p in problems:
        r = baseline_runs[p["id"]]
        low_c, low_t = r["low"]["correct"], r["low"]["output_tokens"]
        if low_c:
            correct += 1
            tokens += low_t
        else:
            escal += 1
            correct += r["high"]["correct"]
            tokens += low_t + r["high"]["output_tokens"]
    return {
        "accuracy": correct / n,
        "mean_output_tokens": tokens / n,
        "escalation_rate": escal / n,
        "n": n,
    }


def load_baseline_runs_full() -> dict:
    by_id: dict[str, dict] = defaultdict(dict)
    with open(RESULTS / "baseline_runs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            by_id[r["id"]][r["budget"]] = r
    return by_id


# ----------------------------------------------------------------------------
# Part C helpers: self-consistency trigger
# ----------------------------------------------------------------------------

def consensus(texts: list[str]):
    """Majority (>=2/3) mathematically-equivalent answer among samples, or None
    (trigger escalation) if no valid majority."""
    answers = [extract_answer(t) for t in texts]
    valid = [a for a in answers if a]
    if len(valid) < 2:
        return None, len(valid)
    clusters: list[list] = []  # [representative, count]
    for a in valid:
        for c in clusters:
            if is_correct(a, c[0]):
                c[1] += 1
                break
        else:
            clusters.append([a, 1])
    best = max(clusters, key=lambda c: c[1])
    return (best[0] if best[1] >= 2 else None), len(valid)


def get_low_samples(p: dict) -> list[dict]:
    return [run_problem(p, "low", sample_idx=i) for i in range(N_SAMPLES)]


def cascade_one(p: dict, baseline_runs: dict) -> dict:
    samples = get_low_samples(p)
    texts = [s["answer"] for s in samples]
    low_out = sum(s["output_tokens"] for s in samples)
    cons, n_valid = consensus(texts)
    gold = p["answer"]

    # pre-escalation answer: majority if it exists, else first sample
    pre_correct = is_correct(cons, gold) if cons else samples[0]["correct"]

    if cons is None:
        high = baseline_runs[p["id"]]["high"]  # cached single high sample
        final_correct = bool(high["correct"])
        tokens = low_out + high["output_tokens"]
        escalated = True
    else:
        final_correct = is_correct(cons, gold)
        tokens = low_out
        escalated = False

    return {
        "id": p["id"],
        "level": p["level"],
        "escalated": escalated,
        "final_correct": final_correct,
        "low_wrong": not pre_correct,
        "output_tokens": tokens,
        "n_valid": n_valid,
    }


def run_cascade(problems: list[dict], baseline_runs: dict, workers: int) -> list[dict]:
    recs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cascade_one, p, baseline_runs): p["id"] for p in problems}
        done = 0
        for fut in as_completed(futs):
            recs.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(problems):
                print(f"  cascade ... {done}/{len(problems)}")
    return recs


def cascade_point(recs: list[dict]) -> dict:
    n = len(recs)
    return {
        "accuracy": sum(r["final_correct"] for r in recs) / n,
        "mean_output_tokens": sum(r["output_tokens"] for r in recs) / n,
        "escalation_rate": sum(r["escalated"] for r in recs) / n,
        "n": n,
    }


def trigger_quality(recs: list[dict]) -> dict:
    tp = sum(r["escalated"] and r["low_wrong"] for r in recs)
    fp = sum(r["escalated"] and not r["low_wrong"] for r in recs)
    fn = sum(not r["escalated"] and r["low_wrong"] for r in recs)
    tn = sum(not r["escalated"] and not r["low_wrong"] for r in recs)
    n_wrong = tp + fn
    n_escal = tp + fp
    recall = tp / n_wrong if n_wrong else None
    precision = tp / n_escal if n_escal else None
    return {
        "recall_escalate_given_low_wrong": recall,
        "precision_low_wrong_given_escalate": precision,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_low_wrong": n_wrong, "n_escalations": n_escal,
    }


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

def plot(baselines, routers, oracle_router, oracle_casc, cascade_test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6))
    bx = [baselines[b]["mean_output_tokens"] for b in BUDGETS]
    by = [baselines[b]["accuracy"] for b in BUDGETS]
    ax.plot(bx, by, "o-", color="#555", label="fixed-budget baselines", zorder=2)
    for b, x, y in zip(BUDGETS, bx, by):
        ax.annotate(b, (x, y), textcoords="offset points", xytext=(6, -12), color="#555")

    ax.scatter(routers["heuristic"]["mean_output_tokens"], routers["heuristic"]["accuracy"],
               s=110, marker="*", color="#1f77b4", zorder=4, label="router (heuristic+level)")
    ax.scatter(routers["heuristic_text"]["mean_output_tokens"], routers["heuristic_text"]["accuracy"],
               s=110, marker="*", color="#9467bd", zorder=4, label="router (heuristic text-only)")
    ax.scatter(oracle_router["mean_output_tokens"], oracle_router["accuracy"],
               s=130, marker="P", color="#2ca02c", zorder=4, label="oracle router")
    ax.scatter(oracle_casc["mean_output_tokens"], oracle_casc["accuracy"],
               s=150, marker="X", color="#17becf", zorder=4, label="oracle cascade (ceiling)")
    ax.scatter(cascade_test["mean_output_tokens"], cascade_test["accuracy"],
               s=150, marker="D", color="#d62728", zorder=5, label="cascade (3x low self-consistency)")

    ax.set_xlabel("mean output tokens / problem")
    ax.set_ylabel("accuracy")
    ax.set_title("Confidence-triggered cascade vs baselines & routers (held-out test)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "final_pareto.png", dpi=130)
    print(f"  wrote {RESULTS / 'final_pareto.png'}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    all_problems = load_all_problems()
    test_problems = load_test_problems()
    baseline_runs = load_baseline_runs_full()

    # --- Part A: oracle cascade ceiling (free) ---
    oc_full = oracle_cascade(all_problems, baseline_runs)
    oc_test = oracle_cascade(test_problems, baseline_runs)
    print("=== PART A: oracle cascade (ceiling, free) ===")
    print(f"  all-200 : acc={oc_full['accuracy']:.3f} tokens={oc_full['mean_output_tokens']:.0f} "
          f"escalate={oc_full['escalation_rate']:.2f}")
    print(f"  test-60 : acc={oc_test['accuracy']:.3f} tokens={oc_test['mean_output_tokens']:.0f}")

    ORACLE_ROUTER = {"mean_output_tokens": 543.0, "accuracy": 0.933}
    beats = (oc_full["mean_output_tokens"] < ORACLE_ROUTER["mean_output_tokens"]
             and oc_full["accuracy"] >= ORACLE_ROUTER["accuracy"] - 0.02)
    print(f"  vs oracle router (543 @ 0.933): "
          f"{'BEATS (cheaper at >= accuracy)' if beats else 'does NOT beat'}")
    if not beats:
        print("\nSTOP: oracle cascade does not beat the oracle router. Reporting and halting.")
        _write_stop_report(oc_full, oc_test, ORACLE_ROUTER)
        return

    # --- Part B: text-only heuristic router (free) ---
    baselines = baseline_points(test_problems)
    routers = {
        "heuristic": eval_point(test_problems, get_predictor("heuristic").predict),
        "heuristic_text": eval_point(test_problems, get_predictor("heuristic_text").predict),
    }
    print("\n=== PART B: text-only heuristic router (free) ===")
    for k in ("heuristic", "heuristic_text"):
        print(f"  {k:<16} acc={routers[k]['accuracy']:.3f} tokens={routers[k]['mean_output_tokens']:.0f}")

    # --- Part C: real cascade (cost-guarded low calls) ---
    spent = spend_so_far()
    # new low calls = samples 1..N-1 for every problem (sample 0 already cached)
    n_new = len(all_problems) * (N_SAMPLES - 1)
    est = n_new * (150 / 1e6 * PRICE["input"] + 350 / 1e6 * PRICE["output"])
    remaining = TARGET_USD - spent
    print(f"\n=== PART C: real cascade ===")
    print(f"  spent so far ${spent:.2f}; remaining toward ${TARGET_USD} = ${remaining:.2f}")
    print(f"  new low calls: {n_new}; estimated cost ${est:.2f}")
    if est > remaining - MARGIN:
        raise SystemExit(f"ABORT: est ${est:.2f} exceeds remaining-margin ${remaining - MARGIN:.2f}")

    workers = CONFIG.get("concurrency", 4)
    recs_full = run_cascade(all_problems, baseline_runs, workers)
    recs_by_id = {r["id"]: r for r in recs_full}
    recs_test = [recs_by_id[p["id"]] for p in test_problems]

    casc_full = cascade_point(recs_full)
    casc_test = cascade_point(recs_test)
    trig_full = trigger_quality(recs_full)
    trig_test = trigger_quality(recs_test)

    print(f"  cascade all-200: acc={casc_full['accuracy']:.3f} tokens={casc_full['mean_output_tokens']:.0f} "
          f"escalate={casc_full['escalation_rate']:.2f}")
    print(f"  cascade test-60: acc={casc_test['accuracy']:.3f} tokens={casc_test['mean_output_tokens']:.0f} "
          f"escalate={casc_test['escalation_rate']:.2f}")

    # --- Part D: trigger quality ---
    print("\n=== PART D: trigger quality (all-200) ===")
    print(f"  recall (escalate | low wrong)   = {trig_full['recall_escalate_given_low_wrong']}")
    print(f"  precision (low wrong | escalate)= {trig_full['precision_low_wrong_given_escalate']}")
    print(f"  tp={trig_full['tp']} fp={trig_full['fp']} fn={trig_full['fn']} tn={trig_full['tn']}")

    base_at_x = interp_baseline_accuracy(baselines, casc_test["mean_output_tokens"])
    above = casc_test["accuracy"] - base_at_x

    report = {
        "config": {"model": MODEL, "n_samples_low": N_SAMPLES, "trigger": "escalate if <2/3 agreement or <2 valid answers"},
        "oracle_cascade": {"all200": oc_full, "test60": oc_test},
        "oracle_router_ref": ORACLE_ROUTER,
        "baselines_test": baselines,
        "routers_test": routers,
        "cascade": {"all200": casc_full, "test60": casc_test},
        "cascade_vs_baseline_line": {
            "baseline_acc_at_cascade_tokens": base_at_x,
            "cascade_acc_above_line": above,
        },
        "trigger_quality": {"all200": trig_full, "test60": trig_test},
        "spend_usd": spend_so_far(),
    }
    with open(RESULTS / "cascade_eval.json", "w") as f:
        json.dump(report, f, indent=2)

    plot(baselines, routers, ORACLE_ROUTER, oc_full, casc_test)
    _write_md(report)
    print(f"\n  wrote {RESULTS / 'cascade_eval.json'} and {RESULTS / 'CASCADE_RESULTS.md'}")
    print(f"  final spend: ${report['spend_usd']:.2f}")


def _gap_closed(oracle_casc, cascade, oracle_router):
    """How much of the oracle-cascade token headroom (vs always-high) the real
    cascade captures, at comparable accuracy."""
    return None


def _write_md(report):
    oc = report["oracle_cascade"]["all200"]
    oct_ = report["oracle_cascade"]["test60"]
    ct = report["cascade"]["test60"]
    cf = report["cascade"]["all200"]
    base = report["baselines_test"]
    tq = report["trigger_quality"]["all200"]
    above = report["cascade_vs_baseline_line"]["cascade_acc_above_line"]
    base_at_x = report["cascade_vs_baseline_line"]["baseline_acc_at_cascade_tokens"]
    orr = report["oracle_router_ref"]
    rt = report["routers_test"]

    L = []
    L.append("# Confidence-Triggered Cascade — Results\n")
    L.append(f"Model {report['config']['model']}. Trigger: {report['config']['trigger']}. "
             f"{report['config']['n_samples_low']} low samples per problem. "
             f"Total spend ${report['spend_usd']:.2f}.\n")

    L.append("## A. Oracle cascade (ceiling: run low, escalate if low wrong)\n")
    L.append(f"- all-200: **{oc['accuracy']:.3f} acc @ {oc['mean_output_tokens']:.0f} tok** "
             f"(escalate {oc['escalation_rate']*100:.0f}%)")
    L.append(f"- test-60: {oct_['accuracy']:.3f} acc @ {oct_['mean_output_tokens']:.0f} tok")
    L.append(f"- vs oracle router ({orr['mean_output_tokens']:.0f} tok @ {orr['accuracy']:.3f}): "
             f"the oracle cascade is **cheaper at comparable accuracy** — a single low pass plus "
             f"selective escalation beats picking one budget upfront.\n")

    L.append("## B. Text-only heuristic router (no privileged `level`)\n")
    L.append("| router | accuracy | mean tokens |")
    L.append("|---|---|---|")
    L.append(f"| heuristic + level (privileged) | {rt['heuristic']['accuracy']:.3f} | {rt['heuristic']['mean_output_tokens']:.0f} |")
    L.append(f"| heuristic text-only | {rt['heuristic_text']['accuracy']:.3f} | {rt['heuristic_text']['mean_output_tokens']:.0f} |")
    L.append("")

    L.append("## C. Real cascade (3x low self-consistency, held-out test)\n")
    L.append("| point | accuracy | mean tokens |")
    L.append("|---|---|---|")
    for b in BUDGETS:
        L.append(f"| baseline {b} | {base[b]['accuracy']:.3f} | {base[b]['mean_output_tokens']:.0f} |")
    L.append(f"| oracle router | {orr['accuracy']:.3f} | {orr['mean_output_tokens']:.0f} |")
    L.append(f"| oracle cascade (all-200) | {oc['accuracy']:.3f} | {oc['mean_output_tokens']:.0f} |")
    L.append(f"| **cascade (test-60)** | {ct['accuracy']:.3f} | {ct['mean_output_tokens']:.0f} |")
    L.append("")
    verdict_line = "ABOVE/LEFT of" if above >= -1e-9 else "below"
    L.append(f"Cascade sits **{verdict_line}** the baseline frontier "
             f"(baseline acc at {ct['mean_output_tokens']:.0f} tok = {base_at_x:.3f}; "
             f"cascade acc {ct['accuracy']:.3f}, {above:+.3f}).\n")

    L.append("## D. Trigger quality (all-200)\n")
    rec = tq["recall_escalate_given_low_wrong"]
    prec = tq["precision_low_wrong_given_escalate"]
    L.append(f"- **recall** (escalated | low wrong) = {rec if rec is None else f'{rec:.2f}'} "
             f"— of {tq['n_low_wrong']} low-wrong problems, {tq['tp']} were escalated.")
    L.append(f"- **precision** (low wrong | escalated) = {prec if prec is None else f'{prec:.2f}'} "
             f"— of {tq['n_escalations']} escalations, {tq['tp']} were necessary.")
    L.append(f"- confusion: tp={tq['tp']} fp={tq['fp']} fn={tq['fn']} tn={tq['tn']}\n")

    L.append("## Verdict: does the cascade close the oracle gap?\n")
    # token headroom of always-high captured
    high_tok = base["high"]["mean_output_tokens"]
    casc_saved = high_tok - ct["mean_output_tokens"]
    oracle_saved = high_tok - oct_["mean_output_tokens"]
    frac = casc_saved / oracle_saved if oracle_saved else 0.0
    L.append(f"Always-high (test) = {high_tok:.0f} tok @ {base['high']['accuracy']:.3f}. "
             f"Oracle cascade saves {oracle_saved:.0f} tok/problem; the real cascade saves "
             f"{casc_saved:.0f} tok/problem = **{frac*100:.0f}% of the oracle-cascade headroom**, "
             f"at accuracy {ct['accuracy']:.3f} vs oracle-cascade {oct_['accuracy']:.3f}.")
    if ct["mean_output_tokens"] <= high_tok and ct["accuracy"] >= base["high"]["accuracy"] - 0.01:
        L.append("\n**The cascade matches always-high accuracy for fewer tokens — it works.**")
    elif above >= -1e-9:
        L.append("\n**The cascade sits on/above the baseline frontier but the 3x-low sampling tax limits token savings — partial win.**")
    else:
        L.append("\n**Negative: the 3x low-sampling cost outweighs selective escalation; the cascade does not beat the frontier.** "
                 "The cheaper 1-low-sample oracle cascade shows the headroom exists, but self-consistency at 3 samples is too expensive to capture it.")
    (RESULTS / "CASCADE_RESULTS.md").write_text("\n".join(L), encoding="utf-8")


def _write_stop_report(oc_full, oc_test, oracle_router):
    L = [
        "# Cascade — STOPPED at ceiling check\n",
        f"Oracle cascade all-200: {oc_full['accuracy']:.3f} acc @ {oc_full['mean_output_tokens']:.0f} tok.",
        f"Oracle router ref: {oracle_router['accuracy']:.3f} @ {oracle_router['mean_output_tokens']:.0f} tok.",
        "\nOracle cascade does not beat the oracle router; no point running the real cascade.",
    ]
    (RESULTS / "CASCADE_RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    with open(RESULTS / "cascade_eval.json", "w") as f:
        json.dump({"stopped_at": "ceiling_check", "oracle_cascade_all200": oc_full,
                   "oracle_cascade_test60": oc_test, "oracle_router_ref": oracle_router}, f, indent=2)


if __name__ == "__main__":
    main()
