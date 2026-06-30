"""Lever sweep + tail-aware Pareto. Strategies: always-cheap, always-frontier,
confidence-gated escalation, oracle. Reports overall AND minority/tail accuracy,
the view incumbents don't show.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diagnostics import minority_mask

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def _eval(test, escalate_flags):
    """escalate_flags: list[bool] aligned with test."""
    M = minority_mask(test)
    n = len(test)
    acc = cost = 0.0
    correct = np.zeros(n, bool)
    for i, r in enumerate(test):
        cost += r["cheap_cost"]  # cheap pass always paid (gated/oracle); see anchors below
        if escalate_flags[i]:
            cost += r["frontier_cost"]
            correct[i] = r["frontier_correct"]
        else:
            correct[i] = r["cheap_correct"]
        acc += correct[i]
    tail_acc = correct[M].mean() if M.sum() else float("nan")
    return {"accuracy": acc / n, "cost_per_q": cost / n, "tail_accuracy": float(tail_acc),
            "escalation_rate": float(np.mean(escalate_flags))}


def sweep(records, workload):
    test = [r for r in records if r["in_test"]]
    n = len(test)
    M = minority_mask(test)

    # anchors that do NOT pay the cheap pass
    always_cheap = {
        "accuracy": np.mean([r["cheap_correct"] for r in test]),
        "cost_per_q": np.mean([r["cheap_cost"] for r in test]),
        "tail_accuracy": 0.0 if M.sum() else float("nan"),
        "escalation_rate": 0.0,
    }
    always_frontier = {
        "accuracy": np.mean([r["frontier_correct"] for r in test]),
        "cost_per_q": np.mean([r["frontier_cost"] for r in test]),
        "tail_accuracy": 1.0 if M.sum() else float("nan"),
        "escalation_rate": 1.0,
    }
    smax = max(r["signal"] for r in test)
    gated = _eval(test, [r["signal"] < smax for r in test])
    oracle = _eval(test, [not r["cheap_correct"] for r in test])

    points = {"always_cheap": always_cheap, "always_frontier": always_frontier,
              "gated_escalation": gated, "oracle": oracle}
    (RESULTS / f"{workload}_levers.json").write_text(json.dumps(points, indent=2, default=float))
    plot(points, workload)
    return points


def plot(points, workload):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    style = {"always_cheap": ("o", "#7f7f7f"), "always_frontier": ("s", "#2ca02c"),
             "gated_escalation": ("*", "#d62728"), "oracle": ("X", "#17becf")}
    for ykey, fname, ylab in [("accuracy", "pareto", "overall accuracy"),
                              ("tail_accuracy", "tail", "TAIL accuracy (high-stakes minority)")]:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for name, pt in points.items():
            m, c = style[name]
            ax.scatter(pt["cost_per_q"] * 1000, pt[ykey], s=160, marker=m, color=c, label=name, zorder=4)
        ax.set_xlabel("cost (milli-USD / query)"); ax.set_ylabel(ylab)
        ax.set_title(f"{workload}: {ylab} vs cost")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
        fig.tight_layout(); fig.savefig(RESULTS / f"{workload}_{fname}.png", dpi=130)
        plt.close(fig)
    print(f"  wrote {workload}_pareto.png + {workload}_tail.png")
