# Phase 2.6 — Cross-Model Disagreement (frozen pre-reg)

Date 2026-06-29. Model B = **Qwen/Qwen2.5-1.5B-Instruct** (LOCAL, $0). CAVEAT: B is a 1.5B local model, weaker than haiku; not comparably capable.

## H1 — disagreement CE recall vs the 0.30 bar (incumbent self-verify 0.21)
- CE set n=**28**, disagreement caught **24** => recall **0.86** (95% CI 0.69-0.94)
- threshold recall>=0.30 -> **PASS** (STRONG: CI excludes 0.21)

## H2 — error independence (the mechanism gate)
- of 28 confident errors (haiku unanimous-wrong), B **also wrong** on 0.79 => independence present
- B was RIGHT on 6 of the CE items; disagreement fired on 6 of those
- threshold overlap<0.85 -> **PASS — partial independence**

## H3 — g (B free) + rule check
| workload | n | g | C_w | C_w_max | escalate | gated/front acc | pred | real | match |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 33 | 1.94 | 1.01 | 5.34 | 76% | 0.91/0.91 | GO | NO | MISS |
| arc_challenge | 34 | 2.85 | 1.19 | 4.41 | 85% | 0.94/0.94 | GO | NO | MISS |
| squad | 39 | 2.89 | 1.02 | 2.83 | 36% | 0.79/0.77 | GO | GO | OK |
| math | 42 | 5.76 | 1.87 | 2.17 | 100% | 0.86/0.86 | GO | NO | MISS |

- rule matches realized on all workloads: **FAIL** (verdicts at g under 2026-06-29 prices).

## Honest interpretation — H1's 0.86 is an artifact

The pre-registered H1 *numerically* passes (recall 0.86), but it is a **weak-model
artifact, not a real result**, and three numbers prove it:

- **Indiscriminate disagreement.** B disagrees with haiku on **76–100%** of items
  (math 100%, arc 85%, gsm8k 76%). A 1.5B model is simply wrong-but-different on almost
  everything, so "disagreement" flags almost everything. High recall, near-zero precision.
- **Genuine independence is only 6/28 = 0.21.** B was actually *correct* on just 6 of
  haiku's 28 confident errors (the cases where independent error structure could help);
  disagreement fired on all 6. The real, non-artifactual catch rate is **0.21 — identical
  to self-verification's plateau.** The other 18 "catches" are B being wrong in a different
  way, which carries no usable signal.
- **The economics expose it (H3).** Escalating 76–100% makes the gated router cost ≈
  always-frontier, so it is NO-GO on 3 of 4 workloads despite the rule predicting GO (the
  rule assumes you escalate only the cheap-wrong set; the noisy signal escalates far more).
  Only SQuAD (escalate 36%, where B is least hopeless) realizes GO.

## Verdict

**CONSOLIDATE — the cheap-detection ceiling holds at ~0.21.** Both self-verification
(Phase 2.5: 0.21) and *genuine* cross-model independence (here: 6/28 = 0.21) land at the
same place. Cross-model disagreement does not beat it **with this weak local B** — its
apparent lift is a wrong-everywhere artifact that the economics (H3) correctly reject.

**Caveat (frozen pre-reg, honoured):** the decisive test is a *comparable, capable* hosted
cheap model (e.g. gpt-4o-mini), where errors could be independent without being
everywhere. That requires a second-provider key and is the **unrun stronger test** — the
one experiment that could still move the ceiling. With the instrument available, the
finding is: cheap minority-detection plateaus at ~0.21 confident-error recall.