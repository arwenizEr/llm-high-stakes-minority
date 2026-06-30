# Lever Advisor Report — `math_k1`

Sample n=200. Signal = self-consistency (no logprobs on Anthropic). Operator default assumed = always-cheap.

## Three diagnostics

1. **Minority size** (cheap wrong, frontier right): **14%** (28/200) — material.
2. **Escalation-set cost concentration C_w** = 2.47 vs g-aware threshold C_w_max = 3.75 (cost gap **g=4.70**, cheap-wrong frac w=0.21). The threshold is `(1 - 1/g)/w` — it MOVES with the signal cost g (Phase 1 finding). C_w below threshold: escalation pays (GO). (Narrative minority concentration C=2.01.)
3. **Confidence-signal recall** on the minority: **0.00** — too low.

## Levers on held-out split (overall acc / TAIL acc / cost)

| strategy | overall acc | tail acc | cost m$/q |
|---|---|---|---|
| always_cheap | 0.867 | 0.000 | 2.67 |
| always_frontier | 0.917 | 1.000 | 17.95 |
| gated_escalation | 0.867 | 0.000 | 2.67 |
| oracle | 0.933 | 1.000 | 13.53 |

## Recommendation

**NO-GO for confidence-gated escalation.** confidence-signal recall (0.0) is below 0.3.

Next-best levers: if traffic is repetitive, **cache** identical/near-identical queries; **trim the prompt** (input-side, lossless) before any model call; or move the default to a **smaller frontier or a mid model** so the expensive tail is cheaper per call. Confidence-gated escalation is not the lever here.

## Caveats

- **Confident-error ceiling:** self-consistency catches *uncertainty*, not confident wrong answers. Unanimous-but-wrong minority items are invisible to the signal — a hard ceiling on recall.
- **Price-regime dependence:** the cost verdict depends on the cheap/frontier price gap (here g=4.70) and on where token cost concentrates. A larger gap or a flatter cost profile can flip a NO-GO to GO.