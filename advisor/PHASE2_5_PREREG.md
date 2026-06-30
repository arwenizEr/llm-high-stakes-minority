# Phase 2.5 — Pre-Registration

**Frozen as of 2026-06-29 (pre-run). Win/fail conditions fixed here; may not move after
results.** Purpose: stress-test the Phase 2 breakthrough before it anchors Phase 3.

## Why this exists
Phase 2 produced the first GO-flip via self-verification catching confident errors (3/7) and
a favorable signal cost (g=4.39). Three weaknesses make it fragile: (1) **n=7** — ±1 case is
±14 points; (2) **possible circularity** — self-verification is the same model, so its win
may be generation/verification asymmetry OR MATH-specific format leakage; (3) **contingent g**
— the GO-flip partly rides on g=4.39, a price-regime term Phase 1 removed from the gate.

## Claim under test
Self-verification adds confident-error recall over self-consistency because generation and
verification are **asymmetric** for the cheap model — not because of format leakage — and this
holds on a confident-error set large enough to estimate, under a stated g.

## Hypotheses (frozen)
- **H1 — Replication at larger n.** Confident-error set ≥ 30 cases (unanimous-but-wrong,
  pooled across the workload spread). PASS if self-verification confident-error recall ≥ 0.30
  with 95% CI lower bound excluding 0; FAIL if CI includes 0 or point estimate < 0.20. The 3/7
  (0.43) is not assumed — it is re-estimated.
- **H2 — Asymmetry, not leakage (the decisive test).** Split the CE set into format-checkable
  (numeric/boxed) vs non-format-checkable (verification requires reasoning). PASS if
  self-verification recall on the non-format-checkable subset is within a stated margin of the
  checkable subset (the win survives where leakage cannot explain it). FAIL if recall collapses
  on the non-format-checkable subset (the crack was format leakage) — blocks the generality
  claim.
- **H3 — Re-anchor the break-even terms.** Recompute combined-signal cost w and g for
  (self-consistency + self-verification) under the current price snapshot; re-evaluate
  `C_w < (1−1/g)/w` across the workload spread (including ≥1 mid-zone C). PASS if the rule still
  predicts realized GO/NO-GO; FAIL if it mispredicts. Every verdict stated as "GO at g=X under
  dated prices."

## Success
Green-light Phase 3 only if H1 + H2 + H3 all PASS. H2 FAIL → leakage-bound, retract the general
claim. H1 FAIL → does not replicate at n; revert to "suggestive, single-digit-n."

## Anti-self-deception
Thresholds frozen as of this timestamp. H2 is the test most likely to kill the result — run it,
don't soften it. Re-derive from cache if a result looks wrong. Report n inline on every recall
figure.
