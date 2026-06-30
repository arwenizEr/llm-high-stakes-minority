# Quality-Aware Lever Advisor

Tells operators, **on their own traffic and before deployment**, whether
confidence-gated cost reduction will pay — and where it will silently fail —
by measuring the **tail-cost-concentration** diagnostic and reporting quality
**on the high-stakes minority**, neither of which shipping routers (RouteLLM,
gateways) or observability tools (Braintrust, Galileo) provide.

## The two-number decision

1. **Minority size** — fraction of queries the cheap model gets wrong but the
   frontier gets right. The high-stakes tail. Cost-cutting silently degrades it.
2. **Tail-cost concentration `C`** = mean frontier cost in that minority ÷ mean
   frontier cost overall.
2b. **Decision rule (g-aware, corrected after Phase 1):** escalation pays iff
   `C_w < (1 − 1/g)/w`, where `C_w` = frontier-cost concentration over the
   cheap-wrong set, `g` = frontier/cheap cost gap, `w` = cheap-wrong fraction.
   **The threshold MOVES with the signal cost g** — it is not a fixed `C`. An
   earlier fixed `C < 2.0` was overfit to the k=3 regime and is retracted (see
   `results/CORRECTION.md`).

A confidence-signal **recall** check (does the cheap model's own uncertainty
catch the minority?) gates the GO — you can't escalate what you can't detect.

## What Phase 1 found (and corrected)

Sweeping 8 workloads across the C spectrum under a fixed k=1 cost model
falsified the fixed-C threshold: MATH (C≈2) came out **+27% GO**, not NO-GO. The
GO/NO-GO line is **signal-cost-dependent**, captured by the g-aware rule above.

Under the corrected rule, both built-in workloads are **NO-GO** at the deployed
k=3 self-consistency signal — for opposite, honest reasons that expose the core
tension:

| workload | signal | g | verdict | why |
|---|---|---|---|---|
| BANKING77 | k=3 | 1.00 | NO-GO | g≈1: cheap-k3 ties the frontier; just use frontier |
| MATH | k=3 | 1.55 | NO-GO | cost concentrates in the escalated tail |
| BANKING77 | k=1 | 3.01 | NO-GO | economics pay, but k=1 has **no** confidence signal |
| MATH | k=1 | 4.70 | NO-GO | economics pay, but k=1 has **no** confidence signal |

**The tension:** a cheap signal (k=1) keeps `g` large (economics work) but carries
no self-consistency signal; an informative self-consistency signal (k≥2) inflates
the cheap pass until `g→1` (economics collapse). The GO region needs a signal that
is **both** cheap and informative (verbalized confidence, cross-model disagreement,
or logprobs) — the Phase 2 target.

## Run

```
python advise.py --workload banking77      # or math (built-in validation sets)
python advise.py --records my_traffic.json # your own sampled traffic
```

`my_traffic.json` is a JSON list of normalized records:
```json
{"id": "...", "in_test": true, "cheap_correct": false, "frontier_correct": true,
 "cheap_cost": 0.0008, "frontier_cost": 0.0009, "signal": 0.67}
```
`signal` = cheap-model confidence (self-consistency agreement, or 1 − logprob
entropy where a provider exposes logprobs — Anthropic does not).

## Caveats (load-bearing)

- **Confident-error ceiling.** Self-consistency catches *uncertainty*, not
  *confident* wrong answers. Unanimous-but-wrong items are invisible — a hard
  recall ceiling. ~7/26 of MATH's minority were confidently wrong.
- **Price-regime dependence.** The cost verdict depends on the cheap/frontier
  price gap `g`. With k=3 self-consistency at a 3× gap, the cheap pass can cost
  as much as one frontier call (`g→1`), at which point *always-frontier* may
  dominate gated escalation outright — the advisor reports `g` so you see this.

## Provenance

Built on the harness/checker/cache from a four-experiment research arc that
falsified cheap-feature routing on the output side and cheap-classifier
compression on the input side, and found self-consistency to be the first
usable minority-detection signal. The advisor productizes that finding: it
measures, rather than assumes, whether the lever pays on *your* workload.
