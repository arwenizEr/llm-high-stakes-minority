# Phase 2.7 — Capable Cross-Provider Disagreement (frozen pre-reg)

Date 2026-06-29. Cheap B = **gpt-4o-mini** ($0.15/0.6 per MTok). Plateau to break: 0.21.

## H1 — genuine independence (B actually correct where haiku confidently wrong)
- of 28 confident errors, gpt-4o-mini correct on **6** => recall **0.21** (95% CI 0.10-0.40)
- threshold >=0.30 -> **FAIL**

## H2 — precision / everything-flagged guard (DECISIVE; the test 2.6 failed)
- disagreement on the MAJORITY (haiku-right): **0.37**
- disagreement on the minority (confident errors): 0.79
- overall escalation rate: **0.43**  | signal precision: 0.09
- threshold: majority<minority AND escalation<0.40 -> **FAIL — blanket disagreement (2.6 artifact recurs)**

## H3 — economics (g with gpt-4o-mini price)
| workload | n | g | escalate | gated/front acc | C_w | C_w_max | pred | real | match |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 120 | 1.72 | 22% | 0.96/0.96 | 1.10 | 12.57 | GO | GO | OK |
| arc_challenge | 120 | 2.47 | 9% | 0.97/0.94 | 1.15 | 8.93 | GO | GO | OK |
| squad | 120 | 2.56 | 25% | 0.91/0.87 | 1.00 | 4.57 | GO | GO | OK |
| math | 200 | 4.43 | 86% | 0.92/0.91 | 2.65 | 3.97 | GO | NO | MISS |

- rule matches realized + >=1 GO: **FAIL**

## Verdict

**CEILING CONFIRMED (confident-error detection) — by the frozen pooled thresholds.**
Failed: H1, H2, H3. But the per-workload evidence forces a more precise, two-part reading.

### Two distinct findings (the pooled verdict conflates them)

**1. The confident-error ceiling is real and triply confirmed: 0.21.** Of the 28 cases
haiku is *unanimously* wrong, gpt-4o-mini is correct on exactly **6 (0.21)** — identical to
self-verification (Phase 2.5) and weak-model independence (Phase 2.6). Three independent
mechanisms now land on the same number. **~78% of confident errors are uncatchable by any
cheap signal** — they are where cheap models systematically fail and only the frontier
succeeds. This ceiling is robust. H1 FAIL is final.

**2. But capable cross-provider disagreement IS a working routing lever — on competent
domains.** Unlike the 1.5B local model (majority-disagree ~1.0, blanket), gpt-4o-mini is
genuinely precise: majority-disagree **0.37** vs minority **0.79**. Per workload, gated
routing is **realized GO on 3 of 4**: gsm8k (escalate 22%, 0.96/0.96), arc_challenge (9%,
0.97/0.94), squad (25%, 0.91/0.87) — accuracy matched-or-beaten, cheaper than always-
frontier. It fails **only on MATH** (escalate 86%), where gpt-4o-mini is *itself* weak and
so blanket-disagrees like the local model. The pooled H2 escalation (0.43 > 0.40) that
trips the frozen gate is dragged there almost entirely by math's 86% × 200 items.

### Honest verdict
- **Confident-error detection: ceiling confirmed at 0.21.** The hardest minority cannot be
  caught cheaply. Don't build a confident-error-catcher on cheap signals.
- **Disagreement-gated routing: a real GO on workloads where the second cheap model is
  competent.** This is the strongest positive of the detection arc — it just doesn't extend
  to the hard confident-error subset or to domains where B is as weak as A.
- The g-aware rule mispredicted only on math: it assumes escalation ≈ the cheap-wrong
  fraction, but a blanket-disagreeing signal escalates far more. **Rule refinement: feed the
  signal's actual escalation rate, not just w.**

All verdicts at the models' 2026-06-29 prices (haiku $1/$5, gpt-4o-mini $0.15/$0.60,
sonnet $3/$15).