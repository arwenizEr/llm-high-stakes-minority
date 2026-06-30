# Lever Advisor Report — `banking77`

Sample n=100. Signal = self-consistency (no logprobs on Anthropic). Operator default assumed = always-cheap.

## Three diagnostics

1. **Minority size** (cheap wrong, frontier right): **8%** (8/100) — material.
2. **Escalation-set cost concentration C_w** = 1.00 vs g-aware threshold C_w_max = 0.01 (cost gap **g=1.00**, cheap-wrong frac w=0.22). The threshold is `(1 - 1/g)/w` — it MOVES with the signal cost g (Phase 1 finding). C_w above threshold: cost concentrates in the escalated tail; escalation does NOT pay (NO-GO). (Narrative minority concentration C=1.01.)
3. **Confidence-signal recall** on the minority: **0.38** — usable.

## Levers on held-out split (overall acc / TAIL acc / cost)

| strategy | overall acc | tail acc | cost m$/q |
|---|---|---|---|
| always_cheap | 0.800 | 0.000 | 1.87 |
| always_frontier | 0.850 | 1.000 | 1.87 |
| gated_escalation | 0.850 | 0.667 | 2.20 |
| oracle | 0.875 | 1.000 | 2.24 |

## Recommendation

**NO-GO for confidence-gated escalation.** escalation-set cost concentration C_w=1.00 exceeds the g-aware break-even 0.01 — the cheap-wrong items are the token-expensive tail, so escalating them costs about as much as just running the frontier on everything.

Next-best levers: if traffic is repetitive, **cache** identical/near-identical queries; **trim the prompt** (input-side, lossless) before any model call; or move the default to a **smaller frontier or a mid model** so the expensive tail is cheaper per call. Confidence-gated escalation is not the lever here.

## Caveats

- **Confident-error ceiling:** self-consistency catches *uncertainty*, not confident wrong answers. Unanimous-but-wrong minority items are invisible to the signal — a hard ceiling on recall.
- **Price-regime dependence:** the cost verdict depends on the cheap/frontier price gap (here g=1.00) and on where token cost concentrates. A larger gap or a flatter cost profile can flip a NO-GO to GO.