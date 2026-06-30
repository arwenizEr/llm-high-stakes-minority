# Phase 1 — Pre-Registration (write before running)

**Date:** 2026-06-29. **Models:** cheap = claude-haiku-4-5, frontier = claude-sonnet-4-6
(plain, no extended thinking) for all NEW workloads; MATH reuses the cached sonnet-4-6
effort=high anchor (noted as a distinct frontier-config point). **Prices:** haiku $1/$5,
sonnet $3/$15 per MTok.

## Question
Across workloads spanning the tail-cost-concentration spectrum, where does the
escalation GO/NO-GO line fall, and is it sharp or a gradient?

## Cost model held FIXED (the catch from the advisor build)
The advisor's `g` (price gap) collapsed to ~1.0 under k=3 self-consistency and flipped
verdicts. To isolate workload SHAPE (C) from the signal's cost, Phase 1 uses:
- **k=1 cheap** (one cheap pass; oracle escalation uses ground-truth correctness, so no
  self-consistency is needed to measure the ceiling).
- **oracle escalation** (escalate iff cheap is actually wrong) — removes the signal-recall
  confound entirely. This measures the *economic ceiling* of escalation per workload.

Per workload we measure: minority size r, concentration C, cost gap g, and the realized
oracle payoff vs always-frontier:
`payoff = (always_frontier_cost - oracle_cost) / always_frontier_cost` (>0 = escalation
saves money at >= frontier accuracy; <0 = it does not).

## Pre-registered predictions (what must hold to count)
1. **Monotonicity:** oracle payoff is decreasing in C (higher concentration → less/zero
   saving). If payoff is flat or non-monotonic in C, the diagnostic does NOT generalize —
   a negative for the whole tool.
2. **Sign change:** payoff crosses zero somewhere in C ∈ [1.0, 2.5]. The crossing point is
   the GO/NO-GO threshold. Predicted band: **GO for C < ~1.5, NO-GO for C > ~2.0**, with a
   gray zone between.
3. **Anchors:** banking77 (C≈1.0) has payoff > 0; MATH (C≈2.2) has payoff ≤ 0. (Already
   known; they must reproduce under the k=1 cost model.)

## Stop / abstain rule
If 8+ workloads show no monotone payoff-vs-C relationship, report the negative: "C does
not predict escalation payoff across workloads" and do not proceed to Phase 2/3 on the
assumption that it does.

## Deliverable
`results/phase1_calibration.png` (payoff vs C, one point per workload) +
`results/phase1_calibration.json` + a stated threshold or gray-zone band.
