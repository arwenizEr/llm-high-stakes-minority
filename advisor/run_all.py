"""Orchestrate Agents 1-4 on both validation workloads and confirm the diagnostic
discriminates: BANKING77 (flat cost) -> GO, MATH (concentrated tail) -> NO-GO."""

from __future__ import annotations

import json
from pathlib import Path

from workload import build_records
from diagnostics import compute
from levers import sweep
from recommender import write_report, decide

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def run(workload):
    print(f"\n========== {workload} ==========")
    records = build_records(workload)
    (RESULTS / f"{workload}_records.json").write_text(json.dumps(records, indent=2, default=float))
    diag = compute(records, workload)
    (RESULTS / f"{workload}_diagnostics.json").write_text(json.dumps(diag, indent=2, default=float))
    print(f"  minority={diag['minority_size']*100:.0f}%  "
          f"C={diag['tail_cost_concentration_C']:.2f} vs C_max={diag['C_max_threshold']:.2f}  "
          f"g={diag['cost_gap_g']:.2f}  signal_recall={diag['signal_recall_cv']}")
    points = sweep(records, workload)
    go = write_report(workload, diag, points)
    print(f"  VERDICT: {'GO' if go else 'NO-GO'}")
    return go


def main():
    results = {}
    for w in ("banking77", "math"):
        results[w] = run(w)
    # NOTE: the original "banking GO, math NO-GO by workload shape" framing was
    # retracted after Phase 1 (see results/CORRECTION.md). Under the corrected
    # g-aware rule, both are NO-GO at the deployed k=3 signal — banking because
    # g~1 (cheap k=3 ties the frontier), math because cost concentrates in the
    # escalated tail. The diagnostic's job is now to report the REGIME and WHY,
    # not to rubber-stamp a shape-based GO.
    print("\n========== HONEST STATUS (corrected g-aware rule, k=3 signal) ==========")
    print(f"  banking77: {'GO' if results['banking77'] else 'NO-GO'} (expect NO-GO: g~1)")
    print(f"  math     : {'GO' if results['math'] else 'NO-GO'} (expect NO-GO: C_w>threshold)")
    print("  Both NO-GO under k=3. GO region needs a cheap+informative signal -> Phase 2.")
    print("  See results/CORRECTION.md.")


if __name__ == "__main__":
    main()
