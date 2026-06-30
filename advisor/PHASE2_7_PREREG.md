# Phase 2.7 — Pre-Registration: Capable Cross-Provider Disagreement

**Frozen as of 2026-06-29 (pre-run). Thresholds may not move after results.**

## Claim under test
A capable cross-provider cheap model breaks the 0.21 ceiling: its errors are independent of
haiku's (right where haiku is confidently wrong, recall > 0.30) AND precise (does not disagree
on most of the majority), so disagreement-gated escalation pays. **Precision is the headline
this time** (the 2.6 lesson): disagreement-recall is NOT the metric; precision (H2) and
genuine-correct recall (H1) are.

## Models
frontier = claude-sonnet-4-6; cheap A = claude-haiku-4-5; cheap B = gpt-4o-mini (OpenAI;
price logged: $0.15/$0.60 per MTok — typically cheaper than haiku, which helps H3).

## Hypotheses (frozen)
- **H1 — Genuine independence beats the plateau.** Of haiku's confident errors (unanimous-
  wrong; reuse Phase 2.5/2.6 set, n ≥ 28), fraction where gpt-4o-mini is actually CORRECT.
  PASS if ≥ 0.30 with 95% CI lower bound excluding 0.21; WEAK PASS if ≥ 0.30 but CI overlaps
  0.21; FAIL if < 0.30.
- **H2 — Precision / everything-flagged guard (DECISIVE; the test 2.6 failed).** Majority-
  disagreement rate (of haiku-right cases, fraction gpt-4o-mini disagrees) and overall
  escalation rate. PASS if majority-disagreement materially lower than minority-disagreement
  AND overall escalation rate < 40%. FAIL if it disagrees on most of the majority too (the 2.6
  artifact recurs). A high H1 with a failed H2 is reported as an artifact, not a win.
- **H3 — Economics.** With gpt-4o-mini's logged price, compute w, g, escalation rate, and net
  cost vs always-frontier across the 4 workloads; re-evaluate `C_w < (1−1/g)/w`. PASS if the
  favorable workloads come out GO and the rule predicts realized verdicts; FAIL if escalation
  cost ≈ always-frontier.

## Success
- CEILING BROKEN / green-light narrowed Phase 3: H1 + H2 + H3 all PASS.
- CEILING CONFIRMED / consolidate: H1 FAIL or H2 FAIL.
- H3-only FAIL: precise + independent but doesn't pay at current prices.

## Anti-self-deception
Thresholds frozen. H2 can kill a numeric H1 pass — run it first. Report precision and
escalation rate alongside every recall figure (never recall alone — that is exactly how 2.6
looked clean). Report per-workload, never only pooled. Re-derive from cache if a result looks
wrong. A WEAK PASS is reported as weak.
