# Model-Internal Signal — GO / NO-GO Verdict

Signal: **self-consistency** over k=3 cheap-model samples. The Anthropic Messages API
exposes **no token-level logprobs** (no request param, none in the response), so logprob
entropy was not available — self-consistency is the signal. Cheap = `claude-haiku-4-5`,
frontier = `claude-sonnet-4-6` effort=high (cached, no new frontier spend). Cheap spend
this experiment: ~$1.68.

**High-stakes minority** (cheap WRONG, frontier RIGHT): **26/200 = 13%** overall; 4/60 on
the held-out test split.

## (a) Does the internal signal beat the surface bar on minority recall? — **YES (modest)**

Robust, all-200 numbers (test-60 has only 4 minority — too noisy to trust):

| signal | minority recall | precision |
|---|---|---|
| surface embeddings (MiniLM, 5-fold CV) | **0.50** (13/26) | 0.29 |
| internal self-consistency (escalate if not unanimous) | **0.73** (19/26) | 0.43 |

The model-internal signal genuinely beats surface features — **the first positive
detection result in the whole research arc** (the pure-surface difficulty experiments
scored recall 0.00–0.11). It is not dramatic (+0.23), and it has a structural blind spot:
**7 of 26 minority problems are unanimous-but-wrong** — the cheap model is *confidently*
wrong, so self-consistency sees no disagreement to flag. Self-consistency catches
uncertainty, not confident error.

## (b) Does the routed system beat always-frontier on cost at matched accuracy? — **NO**

| system | accuracy | cost (m$/q) | escalation |
|---|---|---|---|
| always-cheap (k=3) | 0.867 | 7.95 | 0% |
| surface router | 0.883 | 14.05 | 10% |
| **always-frontier** | **0.917** | **17.95** | 100% |
| internal router (not-unanimous) | 0.917 | 19.24 | 12% |
| oracle router (escalate iff cheap wrong) | 0.933 | 18.81 | 13% |

**Every router is dominated by always-frontier on the cost axis — including the oracle.**
The internal router reaches frontier accuracy (0.917) but costs *more* (19.24 vs 17.95 m$/q).
Even a perfect detector (oracle) costs more than just always using the frontier.

Why the economics don't flip here:
1. **Price gap is 3×, not the 10–20× the playbook assumed** (haiku $1/$5 vs sonnet $3/$15).
2. **k=3 self-consistency triples the cheap pass** ($7.95 m$/q of pure overhead). The signal
   needs k≥2; k=1 has no agreement signal. This overhead (7.95) slightly exceeds the
   easy-frontier cost it saves by not escalating the easy 87% (~7.1).
3. **The minority is exactly the token-expensive tail.** sonnet-high spends thousands of
   output tokens on hard problems, so escalating the 13% hard minority costs about as much
   as running the frontier on everything (the easy 87% are cheap on the frontier anyway).

This is the same structural domination the budget-cascade round already showed (oracle
cascade 555 > oracle router 319), now on the model lane: paying a cheap pass first, then a
frontier pass on the expensive tail, loses to paying one frontier pass — whenever the price
gap is small and the tail is where the tokens are.

## VERDICT: **NO-GO** (for the tail-aware cost optimizer)

Both gates must pass to GO. (a) passes — the internal signal is real and beats surface. (b)
fails decisively — at a 3× price gap with a token-heavy frontier and k=3 overhead, no
router (not even the oracle) saves cost over always-frontier. **Build the quality-aware
lever advisor, not the tail-aware cost optimizer.**

But note what changed: this is *not* another detection failure. It is the arc's **first
positive signal** — model-internal confidence finds 73% of the high-stakes minority that
every surface approach (and shipping production routers) miss. The NO-GO is purely
economic and setup-specific. The signal would become a cost win under any of: a larger
price gap (≥10×), a cheaper confidence proxy than k=3 self-consistency (k=1 + a single
verbalized-confidence prompt; or logprob entropy on a provider that exposes it), or a
frontier whose per-call cost isn't concentrated in the same hard tail. As a tail-*protection*
signal (catch the minority, accept the cost), it already works.

---
*Files: routing_labels.jsonl, routing_labels_meta.json, signal_comparison.json,
routing_system_eval.json, final_routing_pareto.png.*
