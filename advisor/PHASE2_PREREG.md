# Phase 2 — Pre-Registration (write before running)

**Date:** 2026-06-29. **Cheap model:** claude-haiku-4-5. **Frontier label:** cached
sonnet-4-6 effort=high. **Dataset:** MATH 200 (reuse), minority labels from `router/
results/routing_labels.jsonl`. Prices: haiku $1/$5, sonnet $3/$15.

## Question
Can a CHEAP confidence signal (≤2 cheap calls, to preserve the price gap g) beat the
self-consistency baseline on minority recall — and specifically catch the **confident-
error** items (cheap unanimously wrong) that self-consistency is structurally blind to?

## The minority and its blind spot
- minority = cheap-majority-wrong AND frontier-right (26/200).
- **confident-error subset** = minority items where the k=3 cheap samples were UNANIMOUS
  (agreement = 1.0) — confidently wrong (~7/26). Self-consistency cannot flag these by
  construction. Catching them is the whole point.

## Signals tested (all predict `minority`, same labels)
1. `selfcons_k3` — baseline. agreement < 1.0 over 3 cached samples. (recall ~0.73, 0/7 on confident errors).
2. `selfcons_k2` — agreement < 1.0 over 2 cached samples (cheaper baseline).
3. `verbalized` — NEW. 1 haiku call: solve + self-report Confidence 0-100. low conf -> minority.
4. `selfverify` — NEW. 1 haiku call: a fresh haiku judges the cheap majority answer CORRECT/INCORRECT. judged-INCORRECT -> minority. Different mechanism; may catch confident errors.
5. `combined` — verbalized OR selfverify (cheap ensemble, 2 calls).

## Metrics (on the same held-out structure; report all-200 for stability)
- minority recall, precision.
- **confident-error recall** (recall on the ~7 unanimous-wrong subset) — the headline.
- cost: # cheap calls (=> implied g vs frontier; fewer calls = larger g = better economics).

## Pre-registered win condition
A signal WINS if it **catches >= 1 confident-error item (recall_CE > 0)** AND achieves
minority recall **>= 0.73** (the k=3 bar) at **<= 2 cheap calls** (so g stays large enough
to potentially clear the Phase-1 g-aware GO bar). The strongest result: a 1-call signal
(verbalized) meeting both.

## Stop / negative
If NO cheap signal catches any confident-error item (all recall_CE = 0), the confident-
error ceiling stands: self-consistency's blind spot is not cheaply closeable on this task.
Report it plainly — that is the honest fourth-negative completion of the detection arc, and
it tells the advisor that the GO region is unreachable with cheap Anthropic-only signals on
reasoning workloads.
