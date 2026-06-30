# Phase 3 (reframed) — Self-Gating Competence Detection

Date 2026-06-29. A=claude-haiku-4-5, B=gpt-4o-mini, frontier=claude-sonnet-4-6 (labels only — predictor is frontier-free). Frozen rule: A/B disagreement > 50% => abstain.

## Competence score (a-priori, no frontier) vs realized routing
| workload | held-out | n | A/B disagree | predicted | realized GO | rule |
|---|---|---|---|---|---|---|
| arc_challenge | no | 120 | 9% | GO-eligible | GO | OK |
| gsm8k | no | 120 | 22% | GO-eligible | GO | OK |
| squad | no | 120 | 25% | GO-eligible | GO | OK |
| math | no | 200 | 86% | abstain | NO-GO | OK |
| boolq | yes | 80 | 6% | GO-eligible | GO | OK |
| csqa | yes | 80 | 16% | GO-eligible | GO | OK |
| aime | yes | 30 | 100% | abstain | NO-GO | OK |

- **H1** (rule separates original 4): **PASS**
- **H2** (self-gated routing matches realized; abstain=always-frontier safe): **PASS**
- **H3** (holds on >=2 held-out domains — the overfit gate): **PASS**

## Verdict

**SELF-GATING WORKS.** B's competence is detectable a priori from the raw A/B disagreement rate alone (no frontier). The frozen >50% rule correctly classifies GO vs blind-spot domains on held-out workloads, and self-gated routing reproduces the per-domain verdicts. The disagreement lever is self-calibrating and shippable.

All verdicts at 2026-06-29 prices.