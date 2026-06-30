# Adaptive Reasoning-Budget Router — Results

Held-out test split: **60 problems**. Router evaluation reused cached baseline calls (zero new API spend).

## Pareto points (accuracy vs mean output tokens/problem)

| point | accuracy | mean output tokens |
|---|---|---|
| baseline: low | 0.850 | 499 |
| baseline: medium | 0.900 | 923 |
| baseline: high | 0.917 | 1171 |
| **router: heuristic** | 0.917 | 914 |
| **router: learned** | 0.850 | 499 |
| oracle (post-hoc) | 0.933 | 543 |

## Headline: router vs always-high

### heuristic router
- tokens saved vs always-high: **22.0%** (258 tokens/problem)
- accuracy lost vs always-high: **+0.000**
- baseline accuracy at the router's token cost: 0.899; router accuracy: 0.917 (**+0.018** vs the baseline line)
- **Verdict: the heuristic router sits ABOVE/LEFT of the baseline frontier.**

### learned router
- tokens saved vs always-high: **57.4%** (673 tokens/problem)
- accuracy lost vs always-high: **+0.067**
- baseline accuracy at the router's token cost: 0.850; router accuracy: 0.850 (**-0.000** vs the baseline line)
- **Verdict: the learned router sits below the baseline frontier.**

## Verdict

**Yes.** The heuristic router beats the fixed-budget Pareto frontier: at its mean cost of 914 output tokens/problem it reaches 0.917 accuracy, vs 0.899 for the interpolated baseline at the same cost — and it recovers most of the always-high accuracy (0.917) while saving 22.0% of the tokens.

Oracle ceiling: 0.933 accuracy at 543 tokens/problem — the best any predictor using these three budgets could do on this split.