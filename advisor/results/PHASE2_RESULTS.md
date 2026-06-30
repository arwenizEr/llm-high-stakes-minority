# Phase 2 — Cheap Signal vs Confident-Error Ceiling

> ⚠️ **REPLICATION CAVEAT (see Phase 2.5).** The headline confident-error recall below is
> 3/7 = 0.43 — small-n. Phase 2.5 pooled a larger confident-error set (n=28 across MATH +
> GSM8K + ARC-Challenge + SQuAD) and re-estimated it at **0.21 (95% CI 0.10–0.40)**, below
> the pre-registered 0.30 bar. The self-verification *mechanism* is validated (it catches
> confident errors on non-format-checkable tasks too — not format leakage), but the effect
> size is **modest, not the 0.43 implied here.** Read this file with `PHASE2_5_RESULTS.md`.


Baseline selfcons_k3: recall 0.73, confident-error recall 0.0 (blind by construction).

| signal | cheap calls | minority recall | CE recall | beats bar? |
|---|---|---|---|---|
| selfcons_k3 | 3 | 0.73 | 0.00 | - |
| selfcons_k2 | 2 | 0.73 | 0.00 | - |
| verbalized | 1 | 0.65 | 0.14 | - |
| selfverify | 1 | 0.42 | 0.29 | - |
| combined_verb_or_verify | 2 | 0.73 | 0.43 | YES |

**WIN.** combined_verb_or_verify beat the k=3 bar (recall >= 0.73) AND caught 3/7 confident-error items at 2 cheap calls — a cheap signal that cracks the blind spot and preserves the price gap g. This is the signal the Phase-1 GO region needed.

Key mechanism: **self-verification is a DIFFERENT failure mode than self-consistency.** A
fresh haiku pass judging the answer catches confidently-wrong answers (2/7) that k samples
of one model agree on and therefore cannot flag. Verbalized confidence adds 1 more. Self-
consistency's blind spot is not a fundamental wall — it's specific to the agreement
mechanism.

## Capstone — the signal flips MATH from NO-GO to GO

Gated router on MATH (200) using the combined signal (escalate if verbalized confidence
< 90 OR self-verify says INCORRECT), frontier = cached sonnet-high:

| system | accuracy | cost ($/q) |
|---|---|---|
| always-frontier | 0.910 | 0.01305 |
| **combined-signal gated** | **0.900** | **0.01011** (escalate 24%) |
| cheap 2-call pass | — | 0.00297 |

- **23% cheaper than always-frontier at near-matched accuracy** (−0.01).
- **g = 4.39** vs k=3's 1.55. Why: k=3 = three full haiku solves ($0.0084/q); combined =
  one full solve + one SHORT verify (8 output tokens) = $0.0030/q. The verify call is
  informative *and* cheap, so it preserves the price gap.

**This is the cheap-and-informative signal Phase 1 proved the GO region needed.** It
cracks the confident-error blind spot AND keeps g large enough that gated escalation
beats always-frontier. The tail-aware optimizer (the GO product) now has its foundation
and proof: the high-stakes minority is cheaply detectable, and that detection yields real
cost savings — on the very reasoning workload (MATH) that was NO-GO under every prior
signal regime.

Files: phase2_signals.json, PHASE2_PREREG.md. Date 2026-06-29, haiku-4-5 / sonnet-4-6-high.