# Phase 2.6 — Pre-Registration: Cross-Model Disagreement

**Frozen as of 2026-06-29 (pre-run). Thresholds may not move after results.**

## Claim under test
Cross-model disagreement detects confident errors better than single-model self-verification
(recall > 0.30), because two different cheap models' errors are partly **independent** —
while preserving the ~2-call cost budget (g held).

## Models
frontier = claude-sonnet-4-6; cheap A = claude-haiku-4-5; cheap B = a second, different cheap
model. (Run with a LOCAL open-weight model, Qwen2.5-1.5B-Instruct — caveat: weaker than haiku,
not comparably capable; H2 measures whether its errors are independent or merely everywhere.)

## Confident-error set
Pool from the Phase 2.5 set: cheap (haiku) k=3 unanimous (agreement = 1.0) AND cheap-wrong,
across the 4-workload spread (MATH, GSM8K, ARC-Challenge, SQuAD). n ≥ 28.

## Hypotheses (frozen)
- **H1 — CE recall clears the bar.** Cross-model-disagreement confident-error recall ≥ 0.30.
  PASS if ≥ 0.30 and 95% CI lower bound excludes the self-verification incumbent (0.21);
  WEAK PASS if ≥ 0.30 but CI overlaps 0.21; FAIL if < 0.30. The bar is the same 0.30
  self-verification failed; no re-baselining.
- **H2 — Error independence (the mechanism gate).** Of cases A is confidently wrong, fraction
  B is also wrong (error overlap). PASS if overlap materially below 1.0 (models fail on
  different problems) and disagreement tracks the non-overlapping cases. FAIL if overlap ≈ 1.0
  (shared failure) — then a numeric H1 pass is an artifact. (Operationalized: overlap < 0.85.)
- **H3 — Budget preserved.** Compute signal cost w and g for the 2-model signal; re-evaluate
  `C_w < (1−1/g)/w` across the 4 workloads. PASS if g comparable to the 2-call config and the
  rule predicts realized GO/NO-GO; FAIL otherwise.

## Success
- GREEN-LIGHT narrowed Phase 3: H1 + H2 + H3 all PASS.
- CEILING FOUND / consolidate: H1 FAIL or H2 FAIL.
- H3-only FAIL: signal works but doesn't pay at current prices.

## Anti-self-deception
Thresholds frozen here. H2 can kill a numeric H1 pass — run it, report overlap honestly.
Report n inline. Re-derive from cache if a result looks wrong; do not adjust thresholds.
A WEAK PASS is reported as weak.
