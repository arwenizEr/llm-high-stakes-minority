# Phase 3 (Reframed) — Pre-Registration: Self-Gating Competence Detection

**Frozen as of 2026-06-29 (pre-run). Thresholds — including the value below — may not move
after results.**

## Claim under test
A workload where the second cheap model B blanket-disagrees (the MATH failure mode) is
detectable **a priori from cheap B/A signals, without a frontier run** — so routing can
self-gate to GO-domains and abstain on blind-spot domains, matching the realized per-workload
verdicts.

## Models
frontier = claude-sonnet-4-6 (used ONLY to build ground-truth labels, never in the a-priori
predictor — enforced in code); cheap A = claude-haiku-4-5; cheap B = gpt-4o-mini.

## Candidate a-priori signals (frontier-free)
1. Raw A/B blanket-disagreement rate (the MATH tell was 86% vs 9–25% elsewhere).
2. B self-consistency collapse (low self-agreement = guessing).
3. B output-length blowup vs baseline.

## Pre-committed threshold (FROZEN)
Raw A/B blanket-disagreement rate **> 50%** flags a blind-spot (abstain) domain; ≤ 50% flags
GO-eligible. Chosen from the 9–25% vs 86% gap, frozen here before results.

## Hypotheses (frozen)
- **H1 — Competence is a-priori detectable.** The frozen >50% rule correctly classifies the
  original 4 workloads as GO-eligible vs abstain, using B-internal/A-B signals only (no
  frontier). FAIL if it cannot separate them, or only by tuning on these 4 points.
- **H2 — Self-gated routing matches realized verdicts.** Route on disagreement only on
  GO-eligible domains; abstain (pre-committed: always-frontier) on blind-spot domains. PASS if
  this reproduces the realized per-workload GO/NO-GO (GO on competent domains, no
  blanket-escalation disaster on blind-spots).
- **H3 — Not overfit (the honesty gate).** The rule must hold on **≥ 2 held-out workloads**
  not used to set it (new domains). PASS only if it generalizes out-of-sample; FAIL if it
  works in-sample only → suggestive, not proven.

## Success
- SELF-GATING WORKS / green-light narrowed Phase 3: H1 + H2 + **H3** PASS.
- PER-DOMAIN CALIBRATION REQUIRED: H3 FAIL (works in-sample, not out).
- NOT DETECTABLE: H1 FAIL.

## Anti-self-deception
Thresholds frozen here, including the 50% cutoff. H3 is the gate that matters — held-out
domains decide it. The frontier must never leak into the a-priori predictor; verify in code.
Report per-workload, never only pooled. Re-derive from cache if a result looks wrong.
