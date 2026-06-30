"""Phase 1 calibration: oracle escalation payoff vs tail-cost concentration C,
across the workload library. Fixed cost model (k=1 cheap + frontier). Anchors
banking77 + MATH reused from cache.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from harness import CONFIG, cost_of, _cache_path
from workloads_lib import LOADERS, run_workload, estimate_new_calls, WorkloadSpec, CHEAP, FRONTIER
from diagnostics import compute

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ROUTER = ROOT.parent / "router"


def oracle_payoff(records):
    fcost = np.array([r["frontier_cost"] for r in records])
    af_cost = fcost.mean()  # always-frontier
    oracle_cost = np.mean([r["cheap_cost"] + (0 if r["cheap_correct"] else r["frontier_cost"])
                           for r in records])
    oracle_acc = np.mean([r["cheap_correct"] or r["frontier_correct"] for r in records])
    cheap_acc = np.mean([r["cheap_correct"] for r in records])
    front_acc = np.mean([r["frontier_correct"] for r in records])
    return {
        "always_frontier_cost": af_cost, "oracle_cost": oracle_cost,
        "payoff": (af_cost - oracle_cost) / af_cost if af_cost else 0.0,
        "oracle_acc": oracle_acc, "cheap_acc": cheap_acc, "frontier_acc": front_acc,
    }


def banking_spec():
    from workload import load_banking77_items, BANK_SYSTEM
    return WorkloadSpec("banking77", load_banking77_items(), BANK_SYSTEM, "classification", 32)


def math_records():
    labels = {json.loads(l)["id"]: json.loads(l) for l in open(ROUTER / "results" / "routing_labels.jsonl", encoding="utf-8")}
    # k=1 cheap = haiku sample 0
    cheap = {}
    for f in glob.glob(str(ROUTER / "cache" / "*claude-haiku-4-5__sc0__*")):
        r = json.loads(open(f).read())
        cheap[r["id"]] = (bool(r["correct"]), cost_of("claude-haiku-4-5", r["input_tokens"], r["output_tokens"]))
    front = {}
    for line in open(ROUTER / "results" / "baseline_runs.jsonl"):
        r = json.loads(line)
        if r["budget"] == "high":
            front[r["id"]] = (bool(r["correct"]), cost_of("claude-sonnet-4-6", r["input_tokens"], r["output_tokens"]))
    recs = []
    for i, l in labels.items():
        if i not in cheap or i not in front:
            continue
        cc, ck = cheap[i]; fc, fk = front[i]
        recs.append({"id": i, "in_test": l["in_test"], "cheap_correct": cc, "frontier_correct": fc,
                     "cheap_cost": ck, "frontier_cost": fk, "signal": 1.0})
    return recs


def main():
    # gate: estimate new spend across the 7 fresh workloads
    specs = []
    print("loading workloads + estimating new calls ...")
    new_calls = 0
    for name, loader in LOADERS.items():
        try:
            spec = loader()
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        nc = estimate_new_calls(spec)
        new_calls += nc
        specs.append(spec)
        print(f"  {name}: {len(spec.items)} items, {nc} new calls")
    # rough cost: classification ~250 in / 10 out; reasoning more. conservative.
    price = CONFIG["pricing_per_mtok"]
    est = new_calls * (300 / 1e6 * (price[CHEAP]["input"] + price[FRONTIER]["input"]) / 2
                       + 200 / 1e6 * (price[CHEAP]["output"] + price[FRONTIER]["output"]) / 2)
    ceil = CONFIG["cost_ceiling_usd"]
    print(f"total new calls ~{new_calls}; conservative est ${est:.2f} (ceiling ${ceil})")
    if est > ceil:
        raise SystemExit(f"ABORT: est ${est:.2f} > ceiling ${ceil}; top up or reduce phase1_n")

    rows = []
    # fresh workloads
    for spec in specs:
        recs = run_workload(spec)
        rows.append(("new", spec.name, recs))
    # anchors (cached, free)
    rows.append(("anchor", "banking77", run_workload(banking_spec())))
    rows.append(("anchor", "math(sonnet-high)", math_records()))

    table = []
    for kind, name, recs in rows:
        diag = compute(recs, name)
        pay = oracle_payoff(recs)
        table.append({
            "workload": name, "kind": kind, "n": len(recs),
            "minority_size": diag["minority_size"], "C": diag["tail_cost_concentration_C"],
            "g": diag["cost_gap_g"], "payoff": pay["payoff"],
            "oracle_acc": pay["oracle_acc"], "cheap_acc": pay["cheap_acc"], "frontier_acc": pay["frontier_acc"],
            "af_cost_mq": pay["always_frontier_cost"] * 1000, "oracle_cost_mq": pay["oracle_cost"] * 1000,
        })
    table.sort(key=lambda r: r["C"])
    (RESULTS / "phase1_calibration.json").write_text(json.dumps(table, indent=2, default=float))

    print("\n=== PHASE 1 CALIBRATION (sorted by C) ===")
    print(f"  {'workload':<18} {'C':>5} {'minority':>9} {'g':>5} {'payoff':>8}  cheap/front acc")
    for r in table:
        print(f"  {r['workload']:<18} {r['C']:>5.2f} {r['minority_size']*100:>7.0f}% {r['g']:>5.2f} "
              f"{r['payoff']:>+7.1%}  {r['cheap_acc']:.2f}/{r['frontier_acc']:.2f}")

    plot(table)
    threshold_analysis(table)


def plot(table):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for r in table:
        col = "#2ca02c" if r["payoff"] > 0 else "#d62728"
        mk = "P" if r["kind"] == "anchor" else "o"
        ax.scatter(r["C"], r["payoff"] * 100, s=130, marker=mk, color=col, zorder=4)
        ax.annotate(r["workload"], (r["C"], r["payoff"] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.axvline(2.0, color="#1f77b4", lw=1, ls=":", label="advisor threshold C=2.0")
    ax.set_xlabel("tail-cost concentration C"); ax.set_ylabel("oracle escalation payoff vs always-frontier (%)")
    ax.set_title("Phase 1: does C predict escalation payoff? (green=GO, red=NO-GO)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "phase1_calibration.png", dpi=130)
    print(f"  wrote {RESULTS / 'phase1_calibration.png'}")


def threshold_analysis(table):
    Cs = [r["C"] for r in table]
    pays = [r["payoff"] for r in table]
    # monotonicity (Spearman-ish: correlation of rank)
    order = sorted(range(len(Cs)), key=lambda i: Cs[i])
    ranked_pay = [pays[i] for i in order]
    decreasing = all(ranked_pay[i] >= ranked_pay[i + 1] - 0.15 for i in range(len(ranked_pay) - 1))
    # crossing band: last GO C and first NO-GO C
    gos = [r["C"] for r in table if r["payoff"] > 0]
    nogos = [r["C"] for r in table if r["payoff"] <= 0]
    band = (max(gos) if gos else None, min(nogos) if nogos else None)
    print("\n=== THRESHOLD ANALYSIS ===")
    print(f"  payoff monotone-decreasing in C (15% tol): {decreasing}")
    print(f"  GO up to C={band[0]}, NO-GO from C={band[1]}  -> crossing band {band}")
    print(f"  pre-registered prediction was GO<~1.5, NO-GO>~2.0")
    (RESULTS / "phase1_threshold.json").write_text(json.dumps(
        {"monotone_decreasing": decreasing, "go_max_C": band[0], "nogo_min_C": band[1]}, indent=2, default=float))


if __name__ == "__main__":
    main()
