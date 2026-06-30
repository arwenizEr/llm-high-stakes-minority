# Correction — Advisor decision rule (post Phase 1, 2026-06-29)

## What was wrong

The advisor shipped with a fixed threshold **`C < 2.0`** on the minority's tail-cost
concentration, and an n=2 "validation" (banking77 → GO, MATH → NO-GO). Phase 1 (sweeping
8 workloads across the C spectrum under a fixed **k=1** cost model) falsified it: MATH at
C=2.01 came out **+27% GO**, not NO-GO. The fixed-C threshold was **overfit to the k=3
self-consistency regime** the validation happened to use.

## The corrected rule (g-aware, derived for the real policy)

Oracle escalation (escalate iff cheap is wrong, fraction `w`) beats always-frontier iff

```
cheap_cost + w * (mean frontier cost on cheap-wrong) < frontier_cost
=>  C_w < (1 - 1/g) / w
```

where `g = mean_frontier_cost / mean_cheap_cost` and `C_w` = frontier-cost concentration
over the cheap-wrong set. The threshold **moves with the signal cost g** — it is not a
fixed C. This is the break-even I originally derived and wrongly discarded when k=3 made
g≈1.

## What the corrected rule says (cached, $0 re-run)

| workload | signal | g | C_w | C_w_max | concentration | signal recall | VERDICT |
|---|---|---|---|---|---|---|---|
| banking77 | k=3 | 1.00 | 1.00 | 0.01 | fails | 0.38 | **NO-GO** |
| MATH | k=3 | 1.55 | 2.65 | 1.82 | fails | 0.73 | **NO-GO** |
| banking77 | k=1 | 3.01 | 1.00 | 2.90 | **pays** | — (none) | **NO-GO** |
| MATH | k=1 | 4.70 | 2.47 | 3.75 | **pays** | — (none) | **NO-GO** |

**All four are NO-GO** — and the reasons are different and honest:
- **k=3**: a usable confidence signal exists (self-consistency recall 0.73 on MATH), but
  the k=3 overhead at a 3× price gap drives g→~1, so always-frontier dominates gated
  escalation. Economics fail.
- **k=1**: the economics work (concentration pays, g large), but k=1 provides **no
  self-consistency signal** — there's nothing to gate on. Signal fails.

## The finding this exposes

**There is a tension between signal cost and signal existence.** A cheap signal (k=1)
keeps the price gap g large enough for escalation to pay, but a single sample carries no
self-consistency signal. An informative self-consistency signal needs k≥2, which at a 3×
price gap inflates the cheap pass until g→1 and the economics collapse. **Neither current
regime is GO.**

The GO region requires a signal that is BOTH cheap (≈1 cheap call, preserving g) AND
informative — e.g. k=1 + a single verbalized-confidence prompt, cross-model disagreement
between two *different* cheap models, or logprob entropy on a provider that exposes it.
**That is exactly Phase 2.** Phase 1's correction doesn't just fix a threshold — it
sharpens Phase 2's target: find the cheap-and-informative signal that lands in the GO
region the g-aware rule now defines.

## Status of earlier claims
- "Advisor validated, diagnostic discriminates GO vs NO-GO by workload shape" — **retracted
  as stated.** It was regime-specific to k=3 and a fixed-C threshold. The honest claim is:
  the **g-aware (C_w, g, recall) diagnostic correctly identifies that neither current
  signal regime is GO, and explains why** — which is itself the actionable result, and the
  motivation for Phase 2.
