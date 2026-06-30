# High-Stakes Minority Economics in LLM Workloads — Program Report

**Updated:** 2026-06-29. **Models:** cheap = claude-haiku-4-5 ($1/$5), frontier =
claude-sonnet-4-6 ($3/$15). **Total spend:** ~$12.69. All results cached and re-runnable.

## The spine
> The economics and detectability of the **high-stakes minority** — queries a cheap model
> gets wrong but a frontier model gets right. Detecting it is hard; it's also the
> token-expensive tail; that co-location decides whether cost reduction pays.

Six experiments ladder to this one question. Four negatives, one validated product, one
breakthrough.

---

## Experiment ledger

| # | Experiment | Verdict | Headline | Report |
|---|---|---|---|---|
| 1 | Reasoning-budget router (output-side) | NEGATIVE | Difficulty is latent; cheap difficulty predictors collapse to majority (recall ~0). Oracle router 0.945@319tok unreachable. | `router/results/RESULTS.md` |
| 2 | Confidence cascade | NEGATIVE (at ceiling) | Even an oracle cascade is dominated by the oracle router (low-first overhead). | `router/results/CASCADE_RESULTS.md` |
| 3 | Semantic-embedding difficulty | NEGATIVE | Difficulty signal not in MiniLM embeddings either (hard-recall ~0). | `router/results/SEMANTIC_RESULTS.md` |
| 4 | Model-routing / gating signal | MIXED | Self-consistency detects the minority (recall 0.73) — first positive — but at k=3/3× gap doesn't save cost. | `router/results/VERDICT.md` |
| 5 | Downstream-aware compressor (input-side) | NEGATIVE | Loses to LLMLingua-2; cheap span classifier can't ID load-bearing spans (recall 0.07). | `compressor/results/RESULTS.md` |
| — | **Quality-Aware Lever Advisor (product)** | BUILT | Diagnoses, on an operator's own traffic, whether confidence-gated cost reduction pays. | `advisor/README.md`, `advisor/results/*_REPORT.md` |
| P1 | Advisor calibration | CORRECTION | Fixed-C threshold falsified; GO/NO-GO is signal-cost-dependent → g-aware rule. | `advisor/results/PHASE1_RESULTS.md`, `CORRECTION.md` |
| P2 | Cheap signal vs confident-error ceiling | POSITIVE (then tempered) | Self-verification catches confident errors self-consistency can't; flips MATH to GO at g=4.39. Headline 3/7 was small-n. | `advisor/results/PHASE2_RESULTS.md` |
| P2.5 | Stress-test the breakthrough (pre-reg) | **HOLD** | Mechanism real, NOT leakage (H2 pass on non-format tasks). But CE recall replicates at **0.21 (n=28)**, below the 0.30 bar (H1 fail). g-aware rule holds (H3 pass). Phase 3 on hold. | `advisor/results/PHASE2_5_RESULTS.md` |
| P2.6 | Cross-model disagreement (pre-reg) | CONSOLIDATE | Local weak B disagrees indiscriminately (escalate 76–100%) → H1 0.86 is artifact; genuine independence = 6/28 = **0.21** (same plateau); H3 economics fail. Cheap-detection ceiling ~0.21. Stronger test (comparable hosted model) unrun. | `advisor/results/PHASE2_6_RESULTS.md` |
| P2.7 | Capable cross-provider (gpt-4o-mini, pre-reg) | CEILING CONFIRMED + partial GO | Confident-error independence = **6/28 = 0.21 (3rd confirmation)** — ~78% of confident errors uncatchable by any cheap signal. BUT disagreement-gated routing is realized **GO on 3/4 workloads** (gsm8k/arc/squad, escalate 9–25%, acc matched-or-beaten, cheaper); fails only on math where B is also weak. | `advisor/results/PHASE2_7_RESULTS.md` |
| P3 | Self-gating competence detection (pre-reg) | **SELF-GATING WORKS** | Raw A/B disagreement rate detects B's competence **a priori (no frontier)**: frozen >50% rule perfectly separates GO-domains (6–25%) from blind-spots (86–100%) on all 7 workloads incl **3 held-out** (boolq/csqa GO, aime abstain). H1+H2+H3 all PASS. The disagreement lever is self-calibrating and shippable. | `advisor/results/PHASE3_RESULTS.md` |

---

## The arc in one paragraph

Surface features (structural, lexical, semantic) cannot detect the high-stakes minority —
shown three times, recall 0.00–0.11. A model-internal signal (self-consistency) **can**
(recall 0.73) — the first positive — but at k=3 self-consistency on a 3× price gap the
cheap pass costs as much as the frontier (g→1), so escalation doesn't pay. Phase 1 proved
the GO/NO-GO line isn't a workload-shape constant; it moves with the **signal cost** (g),
captured by the g-aware break-even `C_w < (1−1/g)/w`. Phase 2 then found the missing piece:
**self-verification** — a fresh cheap pass judging the answer — catches the *confident
errors* self-consistency is structurally blind to, at a fraction of the token cost. The
combined cheap signal preserves g (4.39) and **flips MATH from NO-GO to GO**: 0.900
accuracy at 23% lower cost than always-frontier. The minority is cheaply detectable, and
that detection yields real cost savings — the spine, demonstrated end-to-end.

## The product

The **Quality-Aware Lever Advisor** (`advisor/`) measures, on a sampled workload, the three
numbers no shipping router or observability tool reports:
1. **minority size**, 2. **tail-cost concentration** (g-aware), 3. **confidence-signal
recall** — and returns a GO/NO-GO with **tail quality** quantified (not just average).
CLI: `python advisor/advise.py --workload <name>` or `--records my_traffic.json`.

## Honest caveats
- The original "advisor validated, discriminates by shape" claim was **retracted** as
  regime-specific (`advisor/results/CORRECTION.md`). The real decision is the g-aware rule.
- Phase 2's win is on MATH; breadth across workloads (more high-C points, larger n) is
  Phase 1's unfinished extension.
- Self-verification's confident-error recall is 3/7 — a crack, not a closed wall.
- Findings are dated to a model+price snapshot; they refresh on re-run, not forever true.

## Where the program stands
Phases 1–2 of the research roadmap are complete (Phase 1 with a correction, Phase 2 a
clear GO). Next per roadmap: **Phase 3** — predict tail-cost concentration C *a priori*
(from cheap-model output-length distribution) so the advisor can advise without the
frontier run. High-risk, highest-ceiling.

### Per-experiment reports
- Output side: `router/results/{RESULTS,CASCADE_RESULTS,SEMANTIC_RESULTS,VERDICT}.md`
- Input side: `compressor/results/RESULTS.md`
- Product + phases: `advisor/README.md`, `advisor/results/{PHASE1_RESULTS,CORRECTION,PHASE2_RESULTS,banking77_REPORT,math_REPORT}.md`
- Pre-registrations: `advisor/PHASE1_PREREG.md`, `advisor/PHASE2_PREREG.md`
