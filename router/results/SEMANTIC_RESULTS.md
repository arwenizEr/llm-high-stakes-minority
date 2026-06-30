# Semantic-Embedding Difficulty Predictor — Results

Embedding: **all-MiniLM-L6-v2** (384-dim, local, $0). Same train/test split (seed 1234). No new target-model calls.

Test label distribution: {'low': 51, 'medium': 4, 'high': 5} (hard = med+high).

## Classifier metrics (test-60)

| classifier | accuracy | hard-class recall | predicted budget mix |
|---|---|---|---|
| logreg_balanced | 0.683 | **0.11 (1/9)** | {'low': 49, 'medium': 5, 'high': 6} |
| mlp | 0.833 | **0.00 (0/9)** | {'low': 59, 'medium': 0, 'high': 1} |

Hard-class recall is the metric that matters — it is the fraction of the med/high problems (the ones always-low gets wrong) that the predictor routes to a higher budget.

## Router on the held-out test split

| point | accuracy | mean tokens |
|---|---|---|
| baseline low | 0.850 | 499 |
| baseline medium | 0.900 | 923 |
| baseline high | 0.917 | 1171 |
| oracle router | 0.933 | 543 |
| **always-low (the bar)** | 0.850 | 499 |
| router (embed logreg_balanced) | 0.850 | 596 |
| router (embed mlp) | 0.850 | 508 |

## Verdict: label-free win, or collapse?

**Collapse — no label-free win.** Best hard-class recall across classifiers was 0.11: the embedding predictors still fail to identify the med/high problems, so the router cannot recover accuracy above always-low without spending extra tokens on the wrong problems. No embedding router sits above the baseline frontier at always-low's cost.

Same failure mode as the structural predictor: with only ~15% hard labels and a hard class that is not linearly separable in MiniLM space, the classifier either collapses toward low or, when class-balanced, over-budgets indiscriminately. **The difficulty signal is not in cheap semantic embeddings either.**

Oracle router ceiling (with perfect difficulty knowledge): 0.933 @ 543 tokens — the headroom a working predictor would capture. The gap to it is the predictor problem, and cheap text/structural/semantic features have now all failed to close it.