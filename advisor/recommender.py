"""Decision rule + per-workload REPORT.md. GO for confidence-gated escalation iff
tail-cost concentration below threshold AND signal recall usable AND minority
material; else recommend the next-best lever."""

from __future__ import annotations

from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

MIN_MATERIAL = 0.05
RECALL_USABLE = 0.30


def decide(diag):
    go_conc = diag["concentration_go"]
    rec = diag["signal_recall_cv"] or 0.0
    go_recall = rec >= RECALL_USABLE
    go_material = diag["minority_size"] >= MIN_MATERIAL
    go = go_conc and go_recall and go_material
    return go, {"concentration": go_conc, "recall": go_recall, "material": go_material}


def write_report(workload, diag, points):
    go, gates = decide(diag)
    ac, af = points["always_cheap"], points["always_frontier"]
    gt, orc = points["gated_escalation"], points["oracle"]

    L = []
    L.append(f"# Lever Advisor Report — `{workload}`\n")
    L.append(f"Sample n={diag['n']}. Signal = self-consistency (no logprobs on Anthropic). "
             f"Operator default assumed = always-cheap.\n")

    L.append("## Three diagnostics\n")
    L.append(f"1. **Minority size** (cheap wrong, frontier right): "
             f"**{diag['minority_size']*100:.0f}%** ({diag['minority_count']}/{diag['n']}) "
             f"— {'material' if gates['material'] else 'too small to matter'}.")
    L.append(f"2. **Escalation-set cost concentration C_w** = {diag['escalation_concentration_Cw']:.2f} "
             f"vs g-aware threshold C_w_max = {diag['Cw_max_threshold']:.2f} "
             f"(cost gap **g={diag['cost_gap_g']:.2f}**, cheap-wrong frac w={diag['cheap_wrong_frac_w']:.2f}). "
             f"The threshold is `(1 - 1/g)/w` — it MOVES with the signal cost g (Phase 1 finding). "
             f"{'C_w below threshold: escalation pays (GO)' if gates['concentration'] else ('g<=1 — the cheap pass already costs as much as the frontier, so always-frontier ties or wins; escalation does NOT pay (NO-GO)' if diag['cost_gap_g'] <= 1 else 'C_w above threshold: cost concentrates in the escalated tail; escalation does NOT pay (NO-GO)')}. "
             f"(Narrative minority concentration C={diag['tail_cost_concentration_C']:.2f}.)")
    rc = diag['signal_recall_cv']
    L.append(f"3. **Confidence-signal recall** on the minority: "
             f"**{rc:.2f}**" + (f" — {'usable' if gates['recall'] else 'too low'}." if rc is not None else " — n/a."))
    L.append("")

    L.append("## Levers on held-out split (overall acc / TAIL acc / cost)\n")
    L.append("| strategy | overall acc | tail acc | cost m$/q |")
    L.append("|---|---|---|---|")
    for name, pt in points.items():
        L.append(f"| {name} | {pt['accuracy']:.3f} | {pt['tail_accuracy']:.3f} | {pt['cost_per_q']*1000:.2f} |")
    L.append("")

    L.append("## Recommendation\n")
    if go:
        d_tail = gt["tail_accuracy"] - ac["tail_accuracy"]
        d_cost = (gt["cost_per_q"] - ac["cost_per_q"]) * 1000
        L.append(f"**GO — deploy confidence-gated escalation.** Tail-cost concentration "
                 f"({diag['tail_cost_concentration_C']:.2f}) is below the break-even threshold "
                 f"({diag['C_max_threshold']:.2f}), so escalating the uncertain minority pays. "
                 f"Projected vs the always-cheap default: **tail accuracy "
                 f"{ac['tail_accuracy']:.2f} → {gt['tail_accuracy']:.2f}** (+{d_tail:.2f}) for "
                 f"**+{d_cost:.2f} m$/q** — the high-stakes minority that always-cheap silently "
                 f"drops is recovered at modest cost.")
    else:
        reasons = []
        if not gates["concentration"]:
            if diag["cost_gap_g"] <= 1:
                reasons.append(f"the cost gap g={diag['cost_gap_g']:.2f} is ~1 — the cheap pass "
                               f"(incl. its self-consistency overhead) already costs as much as one "
                               f"frontier call, so always-frontier dominates gated escalation outright; "
                               f"escalation cannot pay regardless of workload shape")
            else:
                reasons.append(f"escalation-set cost concentration C_w={diag['escalation_concentration_Cw']:.2f} "
                               f"exceeds the g-aware break-even {diag['Cw_max_threshold']:.2f} — the cheap-wrong "
                               f"items are the token-expensive tail, so escalating them costs about as much as "
                               f"just running the frontier on everything")
        if not gates["recall"]:
            reasons.append(f"confidence-signal recall ({rc}) is below {RECALL_USABLE}")
        if not gates["material"]:
            reasons.append(f"minority ({diag['minority_size']*100:.0f}%) is too small to matter")
        L.append(f"**NO-GO for confidence-gated escalation.** " + "; ".join(reasons) + ".")
        L.append("")
        L.append("Next-best levers: if traffic is repetitive, **cache** identical/near-identical "
                 "queries; **trim the prompt** (input-side, lossless) before any model call; or move "
                 "the default to a **smaller frontier or a mid model** so the expensive tail is "
                 "cheaper per call. Confidence-gated escalation is not the lever here.")
    L.append("")
    L.append("## Caveats\n")
    L.append("- **Confident-error ceiling:** self-consistency catches *uncertainty*, not confident "
             "wrong answers. Unanimous-but-wrong minority items are invisible to the signal — a hard "
             "ceiling on recall.")
    L.append("- **Price-regime dependence:** the cost verdict depends on the cheap/frontier price gap "
             f"(here g={diag['cost_gap_g']:.2f}) and on where token cost concentrates. A larger gap "
             "or a flatter cost profile can flip a NO-GO to GO.")
    (RESULTS / f"{workload}_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    return go
