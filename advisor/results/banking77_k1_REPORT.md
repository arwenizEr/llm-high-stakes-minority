# Lever Advisor Report — `banking77_k1`

Sample n=100. Signal = self-consistency (no logprobs on Anthropic). Operator default assumed = always-cheap.

## Three diagnostics

1. **Minority size** (cheap wrong, frontier right): **9%** (9/100) — material.
2. **Escalation-set cost concentration C_w** = 1.00 vs g-aware threshold C_w_max = 2.90 (cost gap **g=3.01**, cheap-wrong frac w=0.23). The threshold is `(1 - 1/g)/w` — it MOVES with the signal cost g (Phase 1 finding). C_w below threshold: escalation pays (GO). (Narrative minority concentration C=1.00.)
3. **Confidence-signal recall** on the minority: **0.00** — too low.

## Levers on held-out split (overall acc / TAIL acc / cost)

| strategy | overall acc | tail acc | cost m$/q |
|---|---|---|---|
| always_cheap | 0.875 | 0.000 | 0.63 |
| always_frontier | 0.875 | 1.000 | 1.88 |
| gated_escalation | 0.875 | 0.000 | 0.63 |
| oracle | 0.900 | 1.000 | 0.86 |

## Recommendation

**NO-GO for confidence-gated escalation.** confidence-signal recall (0.0) is below 0.3.

Next-best levers: if traffic is repetitive, **cache** identical/near-identical queries; **trim the prompt** (input-side, lossless) before any model call; or move the default to a **smaller frontier or a mid model** so the expensive tail is cheaper per call. Confidence-gated escalation is not the lever here.

## Caveats

- **Confident-error ceiling:** self-consistency catches *uncertainty*, not confident wrong answers. Unanimous-but-wrong minority items are invisible to the signal — a hard ceiling on recall.
- **Price-regime dependence:** the cost verdict depends on the cheap/frontier price gap (here g=3.01) and on where token cost concentrates. A larger gap or a flatter cost profile can flip a NO-GO to GO.