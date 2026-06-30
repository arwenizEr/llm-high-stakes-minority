# Cheap-Tier Competence: A Hard Ceiling on Minority Detection and a Self-Gating Router for LLM Cost Reduction

**A consolidated research program.**
**Date:** June 2026. **Models:** cheap A = `claude-haiku-4-5` ($1/$5), cheap B =
`gpt-4o-mini`, frontier = `claude-sonnet-4-6` ($3/$15). **Total program spend:** ~$13.81,
all runs cached and re-runnable. Prices are a dated snapshot; every cost verdict below is
stated relative to it.

---

## Abstract

We study token reduction for LLM workloads through one lens: the **high-stakes minority** —
queries a cheap model gets wrong but a frontier model gets right. Across eleven experiments
on three levers (output-side reasoning budgets, input-side compression, cross-model
routing), each oracle-bounded and most pre-registered, two co-equal findings emerge. (1)
**A hard ceiling on cheap detection.** The hardest core of the minority — ~78% of the cheap
tier's confident errors — is *uncatchable by any cheap signal*, because the entire cheap
tier shares a blind spot there. Three independent mechanisms (self-verification,
weak-model independence, capable cross-provider independence) all detect exactly 0.21 of
confident errors. (2) **A self-gating router that captures the rest.** Capable cross-provider
disagreement is a working routing lever wherever the second cheap model is competent, and —
the constructive capstone — it detects its own competence *a priori* from the raw
disagreement rate alone, with no frontier call, gating itself on where it pays. The hidden
variable behind both findings is the same: **cheap-tier competence on the workload**, which
determines both why the ceiling exists and where the lever works. We also report the
program's method — pre-registration, oracle bounds, a retracted validation, a downgraded
breakthrough, an artifact caught at a clean-looking 0.86 — as a first-class result, because
the discipline is what makes the negatives trustworthy.

---

## The spine

> **The economics and detectability of the high-stakes minority** — queries a cheap model
> gets wrong but a frontier model gets right. Detecting that minority cheaply is hard; the
> same minority is also the token-expensive tail; and that co-location decides whether cost
> reduction pays.

Every experiment ladders to this one question. The program is its answer, end to end.

---

## The two co-equal findings

**Finding 1 — The cheap-tier shared blind spot (a hard ceiling).** ~78% of the cheap tier's
confident errors are cases where *every* cheap model fails and only the frontier succeeds.
No cheap signal reaches them, because there is no cheap vantage point outside the shared
blind spot. Cheap minority-detection plateaus at **0.21 confident-error recall**, confirmed
three independent ways.

**Finding 2 — The self-gating cross-provider router (a working lever with a known
boundary).** Where a second cheap model is *competent*, its disagreement with the first is
precise and independent, and disagreement-gated routing pays (GO). Where the second model
is itself in the blind spot, it blanket-disagrees and the lever fails. Crucially, the **raw
disagreement rate is self-diagnosing**: it is both the routing signal and its own validity
check, so the router detects a-priori where it is competent and gates itself.

The unifying mechanism: **cheap-tier competence is the hidden variable behind both.** Where
B is competent, disagreement is precise and independent (lever works); where B is in the
blind spot, disagreement is blanket and errors are shared (lever fails, and the ceiling
bites). One variable, seen from two angles.

---

## Methodology (a first-class result)

The findings are only as trustworthy as the process that produced them. The program held to:

- **Oracle bounds on every lever** — a negative means "the headroom is real and here is what
  blocks it," not "we couldn't do it."
- **Pre-registration where it mattered** — win/fail thresholds frozen in writing before
  results, including the threshold *values*; validated out-of-sample, not tuned in-sample.
- **Caching, cost-gating, fixed seeds, logged configs** — every rerun free; every spend
  gated; every result dated to its model + price snapshot.
- **Honest effect sizes** — `n` reported inline on every recall figure; weak passes reported
  as weak.
- **Re-derivation, never threshold-fishing** — when a result looked wrong, the fix came from
  cached data, not a moved goalpost.

This discipline produced, on the record: a **retracted** product-validation claim
(`CORRECTION.md`), a breakthrough **downgraded** from 0.43 to 0.21 against a pre-registered
bar, and an **artifact caught** when a literal recall of 0.86 was shown to be a weak-model
blanket-disagreement and reported as such. Most work in this area reports the 0.86. The
program's worked example of *not* doing that is its own contribution.

---

## Experiment ledger

| # | Experiment | Verdict | Headline |
|---|---|---|---|
| 1 | Reasoning-budget router (output-side) | NEGATIVE | Difficulty latent; cheap predictors collapse to majority (recall ~0). Oracle router 0.945@319tok vs always-high 0.917@1171 — headroom real, unreachable. |
| 2 | Confidence cascade | NEGATIVE (at ceiling) | Oracle cascade 0.930@555 dominated by oracle router 0.945@319 — low-first overhead is structural. |
| 3 | Semantic-embedding difficulty | NEGATIVE | Difficulty not in MiniLM embeddings either; hard-class recall 0.00–0.11. |
| 4 | Model-routing / gating signal | MIXED | Self-consistency detects the minority (recall 0.73, first positive; minority 13%, surface bar 0.50) — but every router incl. oracle dominated on cost (18.8 vs 17.95 m$/q). |
| 5 | Downstream-aware compressor (input-side) | NEGATIVE | Loses to LLMLingua-2 (−0.06 to −0.13); load-bearing-span recall 0.071; labels 90/10. |
| P1 | Advisor calibration | CORRECTION | Fixed-C threshold falsified; GO/NO-GO is signal-cost-dependent → g-aware rule. "Validated by shape" retracted. |
| P2 | Self-verification vs confident-error ceiling | BREAKTHROUGH (tempered) | Self-verification cracks the blind spot (3/7); flipped MATH to GO (0.900 acc, 23% cheaper, g=4.39). |
| P2.5 | Replication + leakage disambiguation | HOLD | H1 replicate FAIL (recall 0.21, n=28, vs 0.30 bar — 0.43 was small-n optimism). H2 asymmetry-not-leakage PASS. H3 g-rule PASS. |
| P2.6 | Weak-model cross-disagreement | CONSOLIDATE | Literal recall 0.86 = artifact (1.5B blanket-disagrees); genuine independence 6/28 = 0.21; economics FAIL. |
| P2.7 | Capable cross-provider (gpt-4o-mini) | TWO-PART | Ceiling confirmed 3rd time (6/28=0.21). But disagreement routing is a real GO lever on competent domains (gsm8k/arc/squad, escalate 9–25%); fails MATH (B weak, blanket 86%). |
| P3 | Self-gating competence detection | CAPSTONE — WORKS | B's competence detectable a-priori from raw A/B disagreement; frozen >50% rule classifies all 7 domains (incl. 3 held-out) perfectly. Frontier walled off in code. |
| — | **Quality-Aware Lever Advisor (product)** | BUILT | Measures minority size, g-aware tail-cost concentration, signal recall on an operator's own traffic; returns GO/NO-GO with **tail** quality. |

---

## The arc

**Part I — Surface features cannot detect the minority (1–3, 5).** The oracle router proves
the output-side headroom is large and real (0.945@319 vs 0.917@1171 — higher accuracy at
73% fewer tokens), yet every cheap predictor of *which* problems need more compute collapsed
to the majority: structural features, then semantic embeddings (hard-recall 0.00–0.11). A
cascade could not rescue it (the oracle cascade is dominated by the oracle router on
low-first overhead). Input-side told the same story: a downstream-aware compressor lost to
LLMLingua-2 because it could not identify load-bearing spans (recall 0.071). Two lanes, two
datasets, two token streams — one failure, at the same near-zero recall.

**Part II — Model-internal signal partly works, and the co-location appears (4).** On the
model-routing lane, the cheap model's *own* self-consistency lifted minority recall to 0.73
— the first positive detection — where surface features sat at 0.50. But it did not pay:
every router, *including the oracle*, was dominated by always-frontier on cost (18.8 vs
17.95 m$/query). The reason is the program's central economic fact: **the high-stakes
minority is the token-expensive tail.** The hard queries and the expensive queries are the
*same queries*, so escalating the minority costs about as much as always paying the frontier.
Savings and risk are co-located; that is why they cannot be cheaply separated.

**Part III — The detection ceiling, triply confirmed (P2 → P2.7).** Self-verification first
appeared to crack the blind spot (3/7) and even flipped MATH to GO — but pre-registered
replication downgraded it to 0.21 (n=28), while confirming the *mechanism* is real
(generation/verification asymmetry survives a leakage control). Weak-model cross-disagreement
produced a clean-looking 0.86 that was an artifact; its genuine independence was again 0.21.
Capable cross-provider independence (gpt-4o-mini) was again 6/28 = 0.21. Three mechanisms,
one number. The interpretation is mechanistic, not coincidental: ~78% of confident errors
are the cheap tier's *shared* blind spot, and no cheap signal has a vantage point outside it.

**Part IV — A working lever, with a precise boundary (P2.7).** Capable cross-provider
disagreement, unlike the weak model's blanket disagreement (majority-disagree 0.37 vs
minority 0.79 — *precise*), is a realized GO on 3 of 4 domains: gsm8k, arc, squad (escalate
9–25%, accuracy matched or beaten, cheaper than always-frontier). It fails only on MATH,
where gpt-4o-mini is itself in the blind spot and blanket-disagrees (86%). The boundary *is*
the finding: routing pays iff the second model is competent on the domain.

**Part V — The lever becomes self-aware (P3).** B's competence is detectable *a priori* from
the raw A/B disagreement rate alone, no frontier call: GO-domains disagree 6–25% (arc,
gsm8k, squad, and held-out boolq, csqa); blind-spot domains 86–100% (math, and held-out
aime). A threshold frozen at >50% *before* results classified all seven domains correctly,
including three held out of calibration. The disagreement rate is self-diagnosing — a
competent B agrees on the easy majority and disagrees only on real errors (low, precise →
route); a blind-spot B is wrong-but-different on everything (blanket → the rate itself says
"don't trust me here"). The router knows when to turn itself on.

---

## The product

The **Quality-Aware Lever Advisor** operationalizes the program. On a sample of an operator's
own traffic plus a quality checker, it reports the three numbers no shipping router or
observability tool provides: **minority size**, **g-aware tail-cost concentration**, and
**confidence-signal recall** — and returns a GO/NO-GO with **tail** quality quantified, not
just average. Validated to discriminate: BANKING77 (flat cost, C=1.01) → GO, recovering 67%
of the silently-dropped tail for +0.33 m$/query; MATH (concentrated tail, C=2.18) → NO-GO,
where escalation costs ~10× and is dominated by always-frontier. The decision rule is the
g-aware break-even `C_w < (1−1/g)/w`, fed the signal's *actual* escalation rate (the P2.7
correction). Every verdict is reported as "GO at g=X under dated prices," never as a
workload-intrinsic property.

---

## Honest boundaries

- **One held-out blind-spot domain.** Blind-spot detection generalized out-of-sample (AIME),
  but the blind-spot class rests on two points (MATH, AIME), both competition math. Whether a
  blind spot of a *different shape* (specialized vocabulary, long-context) also produces
  blanket disagreement is untested. This is the precise edge of Finding 2 — and the one
  sequel experiment, if the program ever resumes.
- **Price-regime contingency.** Every cost verdict and every `g` depends on a dated price
  snapshot; the 3× cheap/frontier gap that shaped the routing economics is not permanent
  (prices fell ~80% in the prior year). Findings refresh on re-run; they are not forever true.
- **Sample sizes.** n≈200 per workload; the confident-error sets are n≈28. The 0.21 ceiling
  is robust across three mechanisms but each estimate is modest-n.
- **Two model families, specific datasets.** The mechanisms plausibly generalize, and the
  *convergence across lanes and providers* is evidence for that, but breadth beyond these
  settings is a hypothesis this program motivates, not one it proves.

---

## Implications

1. **Instrument minority-class recall first.** It predicted every outcome cheaply and early,
   before any Pareto plot; a method with near-zero minority recall collapses to the dumb
   baseline regardless of headline accuracy.
2. **The cheap tier has a shared blind spot.** ~78% of its confident errors are unreachable
   by *any* cheap signal — a hard limit on cheap reliability methods that build no second,
   independent vantage point.
3. **Cross-provider routing works where the second model is competent, and competence is
   self-detectable.** The raw disagreement rate gates the router for free.
4. **Production routers are blind to the variables that decide their value** — tail-cost
   concentration and tail quality — because they optimize and report averages. The advisor
   surfaces both.

---

## Conclusion

The spine question is answered end to end. The hardest core of the high-stakes minority is
*undetectable cheaply* — a hard ceiling at the cheap tier's shared blind spot, proven three
ways. Everything inside that ceiling, where a second cheap model is competent, is captured by
a cross-provider disagreement router that *knows where it is competent* from a signal costing
nothing to compute. One hidden variable — cheap-tier competence — explains both the wall and
the lever. The contribution is two-fold and co-equal: the findings, which tell you where the
walls are; and the method, which is why the walls can be trusted to be real.

---

## Appendix — Artifacts

- Output side: `router/results/{RESULTS,CASCADE_RESULTS,SEMANTIC_RESULTS,VERDICT}.md`
- Input side: `compressor/results/RESULTS.md`
- Product + phases: `advisor/README.md`,
  `advisor/results/{PHASE1_RESULTS,CORRECTION,PHASE2_RESULTS,PHASE2_5_RESULTS,PHASE2_6_RESULTS,PHASE2_7_RESULTS,PHASE3_RESULTS,banking77_REPORT,math_REPORT}.md`
- Pre-registrations: `advisor/{PHASE1_PREREG,PHASE2_PREREG,PHASE2_5_PREREG,PHASE2_6_PREREG,PHASE2_7_PREREG,PHASE3_PREREG}.md`
- All runs cached, cost-gated, fixed-seed, config-logged, dated to a model + price snapshot.
  Entire program under $14.