"""Lever Advisor CLI (Agent 5 — reusable tool).

Point it at a workload of normalized per-item records and get the GO/NO-GO on
confidence-gated cost reduction, with tail quality quantified.

Each record is a dict:
  {id, in_test, cheap_correct, frontier_correct, cheap_cost, frontier_cost, signal}
where `signal` is the cheap model's confidence (higher = more confident;
self-consistency agreement or 1 - logprob-entropy).

Usage:
  python advise.py --records my_traffic.json        # your own sampled traffic
  python advise.py --workload banking77             # built-in validation workload
  python advise.py --workload math

The decision rests on TWO numbers (plus a recall sanity check):
  1. minority size       — fraction cheap gets wrong but frontier gets right
  2. tail-cost concentration C = mean frontier cost in that minority / overall
     GO iff C < 2.0 (flat tail; escalation pays) and signal recall is usable
     and the minority is material.

CAVEAT (read this): the confidence signal catches UNCERTAINTY, not CONFIDENT
ERROR. Unanimous-but-wrong items are invisible to it — a hard recall ceiling.
The cost verdict is also price-regime dependent (cheap/frontier gap g).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from diagnostics import compute
from levers import sweep
from recommender import write_report, decide


def main():
    ap = argparse.ArgumentParser(description="Quality-aware lever advisor")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--records", help="path to JSON list of normalized records")
    g.add_argument("--workload", choices=["banking77", "math"], help="built-in workload")
    ap.add_argument("--name", default=None, help="label for outputs")
    args = ap.parse_args()

    if args.records:
        records = json.loads(Path(args.records).read_text())
        name = args.name or Path(args.records).stem
    else:
        from workload import build_records
        records = build_records(args.workload)
        name = args.name or args.workload

    diag = compute(records, name)
    points = sweep(records, name)
    go = write_report(name, diag, points)
    gates_go, gates = decide(diag)

    print(f"\n=== LEVER ADVISOR: {name} ===")
    print(f"  minority size            : {diag['minority_size']*100:.0f}% "
          f"({diag['minority_count']}/{diag['n']})  [{'material' if gates['material'] else 'too small'}]")
    print(f"  escalation concentration C_w: {diag['escalation_concentration_Cw']:.2f} "
          f"vs g-aware threshold {diag['Cw_max_threshold']:.2f}  "
          f"[{'pays' if gates['concentration'] else 'does not pay'}]  (narrative minority C={diag['tail_cost_concentration_C']:.2f})")
    print(f"  confidence-signal recall : {diag['signal_recall_cv']}  [{'usable' if gates['recall'] else 'too low'}]")
    print(f"  cost gap g (cheap->front): {diag['cost_gap_g']:.2f}")
    print(f"\n  VERDICT: {'GO - deploy confidence-gated escalation' if go else 'NO-GO - use a different lever'}")
    print(f"  report: results/{name}_REPORT.md   plots: results/{name}_{{pareto,tail}}.png")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
