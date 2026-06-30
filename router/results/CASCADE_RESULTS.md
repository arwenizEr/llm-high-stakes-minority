# Confidence-Triggered Cascade — Results

Model `claude-sonnet-4-6`. MATH, 200 problems (seed 1234), 60 held out.
**Total spend $5.40** (the cascade added **$0** — it halted at the free ceiling check).

## Bottom line

The cascade pivot is **refuted at the ceiling**, and the earlier heuristic win is
shown to depend entirely on a privileged label. Two free checks settled it before
any new API calls:

1. A *perfect* (oracle) cascade does not beat the oracle router.
2. The heuristic router's 22% saving disappears when the dataset `level` feature
   is removed.

So the real cascade (Part C) and its trigger study (Part D) were **not run** — by
the task's own stop rule.

---

## A. Oracle cascade ceiling (free; run low, escalate to high iff low is wrong)

| set | accuracy | mean tokens | escalation |
|---|---|---|---|
| oracle cascade (all-200) | 0.930 | 555 | 14% |
| oracle cascade (test-60) | 0.933 | 1131 | — |
| **oracle router (all-200)** | **0.945** | **319** | — |
| oracle router (test-60, ref) | 0.933 | 543 | — |

**The oracle router dominates the oracle cascade on both axes** (all-200: 0.945@319
vs 0.930@555). Why: the cascade *always pays the low pass first*, then the full high
pass on the ~14% it escalates (low+high on hard problems). The router skips that
wasted low pass — it spends a single budget per problem. The low-first overhead is
structural and a better trigger cannot remove it. **Stop rule triggered: cascade
ceiling beats router ceiling is false → halt.**

## B. Does the heuristic win survive without the privileged `level` label?

| router (test-60) | accuracy | mean tokens | budget mix |
|---|---|---|---|
| heuristic **+ level** (privileged) | 0.917 | 914 | low26/med11/high23 |
| heuristic **text-only** | 0.850 | 727 | low56/med4/high0 |
| baseline low (ref) | 0.850 | 499 | — |

**The 22% win does not survive.** Without `level`, the heuristic predicts low for
56/60 problems — accuracy falls to the always-low level (0.850) while spending *more*
tokens than always-low. The entire previous win was carried by the dataset's
difficulty label, which real queries do not have.

## C / D. Real cascade + trigger study — NOT RUN

By Part-A's stop rule. Even ignoring it, the realizable 3x-low self-consistency
cascade is strictly worse than its own oracle ceiling: it pays **3x the low pass on
every problem** (~3x500 = 1500 tok on the test split, already above always-high's
1171) before any escalation. The oracle cascade (1 low pass) already loses to the
router; tripling the low cost cannot recover.

## Verdict: does the cascade close the oracle gap?

**No — and the gap is not the cascade's to close.** The token headroom is real
(oracle router 0.945 @ 319 vs always-high 0.917 @ 1171 — a 73% token cut at *higher*
accuracy), but:

- A cascade cannot capture it: paying low-then-high on escalations is more expensive
  than the router's single-budget choice. The oracle cascade (555 tok) is already
  worse than the oracle router (319 tok).
- The cheap, realistic difficulty signal is the actual missing piece — and **text /
  structural features carry almost none** (text-only heuristic ties always-low;
  learned predictor collapsed to majority in the prior round).

**The real bottleneck is a cheap difficulty predictor, not the escalation policy.**
A *cheaper* confidence signal than 3x self-consistency (a single low pass plus a
validity/format check, or the model's own calibrated confidence) is the only
remaining lever — but it is still capped by the oracle cascade (555 tok), which
loses to the oracle router (319 tok). Within these three discrete budgets, **upfront
routing is the better architecture; the cascade is structurally dominated.**
