"""Phase 2.7 — capable cross-provider disagreement (frozen pre-reg).

Cheap B = gpt-4o-mini (OpenAI). The 2.6 lesson is baked in: disagreement-recall
is NOT the headline; PRECISION (H2) and genuine-correct independence (H1) are.

H1: of haiku's confident errors, fraction gpt-4o-mini is actually CORRECT (>0.30, CI vs 0.21).
H2 (decisive): majority-disagreement rate << minority-disagreement rate, escalation < 40%.
H3: g with gpt-4o-mini price + g-aware rule check.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math as _math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from harness import CONFIG, run_model, cost_of
from checkers import get_checker, classification_check, numeric_check
import workloads_lib as WL

ROOT = Path(__file__).resolve().parent
ROUTER = ROOT.parent / "router"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
CHEAP = CONFIG["cheap_model"]
FRONTIER = CONFIG["frontier_model"]
BMODEL = CONFIG["openai_model"]
BPRICE = CONFIG["pricing_per_mtok"][BMODEL]
K = 3
sys.path.insert(0, str(ROUTER))
from checker import is_correct as math_is_correct

FORMAT_CHECKABLE = {"math": True, "gsm8k": True, "arc_challenge": False, "squad": False}

# import the cached-haiku rebuild from 2.6 (no spend)
from phase2_6 import fresh_items, math_items, checker_for, wilson


# ---------- OpenAI B ----------
_oa = None


def _openai():
    global _oa
    if _oa is None:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            for p in (ROOT / ".env", ROOT.parent / "router" / ".env"):
                if p.exists():
                    for line in p.read_text().splitlines():
                        if line.strip().startswith("OPENAI_API_KEY"):
                            key = line.partition("=")[2].strip().strip('"').strip("'")
        _oa = OpenAI(api_key=key)
    return _oa


def _b_cache(workload, system, prompt):
    h = hashlib.sha256((BMODEL + system + prompt).encode()).hexdigest()[:12]
    return CACHE / f"{workload}__{BMODEL}__{h}.json"


def run_b(workload, item, system, max_tokens, force=False):
    cp = _b_cache(workload, system, item["prompt"])
    if cp.exists() and not force:
        return json.loads(cp.read_text())
    resp = _openai().chat.completions.create(
        model=BMODEL, max_tokens=max_tokens, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": item["prompt"]}])
    ans = (resp.choices[0].message.content or "").strip()
    u = resp.usage
    cost = u.prompt_tokens / 1e6 * BPRICE["input"] + u.completion_tokens / 1e6 * BPRICE["output"]
    rec = {"workload": workload, "id": item["id"], "model": BMODEL, "answer": ans, "cost": cost}
    cp.write_text(json.dumps(rec))
    return rec


def main():
    specs = [WL.load_gsm8k(CONFIG["phase25_n"]), WL.load_arc_challenge(), WL.load_squad(CONFIG["phase25_n"])]
    print("rebuilding haiku-side records from cache (no spend) ...")
    records = []
    for s in specs:
        records += fresh_items(s)
    records += math_items()

    # cost gate for B (gpt-4o-mini on all pooled items)
    n_new = sum(1 for r in records if not _b_cache(r["workload"], r["system"], r["item"]["prompt"]).exists())
    est = n_new * (BPRICE["input"] * 600 / 1e6 + BPRICE["output"] * 150 / 1e6)
    print(f"B=gpt-4o-mini on {len(records)} items ({n_new} new); est ${est:.3f} (ceil ${CONFIG['cost_ceiling_usd']})")
    if est > CONFIG["cost_ceiling_usd"]:
        raise SystemExit("ABORT cost gate")

    done = 0
    for r in records:
        mt = 256 if r["workload"] in ("math", "gsm8k") else 24
        b = run_b(r["workload"], r["item"], r["system"], mt)
        eq = checker_for(r["workload"])
        r["b_answer"] = b["answer"]
        r["b_correct"] = bool(eq(b["answer"], r["gold"]))
        r["b_cost"] = b["cost"]
        r["disagree"] = (not eq(r["maj"], b["answer"])) if r["maj"] else True
        done += 1
        if done % 100 == 0 or done == len(records):
            print(f"  B {done}/{len(records)}")

    arr = lambda f: np.array([f(r) for r in records])
    minority = arr(lambda r: (r["agreement"] >= 1.0) and (not r["cheap_correct"]) and r["frontier_correct"])
    ce = arr(lambda r: (r["agreement"] >= 1.0) and (not r["cheap_correct"]))  # confident error (Phase 2.5/2.6 set)
    haiku_right = arr(lambda r: r["cheap_correct"])
    disagree = arr(lambda r: r["disagree"])
    b_correct = arr(lambda r: r["b_correct"])

    # H1 — genuine independence: of CE, fraction B correct
    ce_idx = np.where(ce)[0]
    h1_caught = int(b_correct[ce].sum())
    h1_recall = h1_caught / ce.sum() if ce.sum() else 0.0
    h1_ci = wilson(h1_caught, int(ce.sum()))

    # H2 — precision / artifact guard
    maj_disagree = disagree[haiku_right].mean() if haiku_right.sum() else 0.0
    min_disagree = disagree[ce].mean() if ce.sum() else 0.0
    escalation = disagree.mean()
    # precision of disagreement as a minority detector
    flagged = disagree.sum()
    prec = (disagree & ce).sum() / flagged if flagged else 0.0

    # H3 — economics per workload
    byw = defaultdict(list)
    for r in records:
        byw[r["workload"]].append(r)
    h3 = []
    for w, rs in byw.items():
        cheap_cost = np.mean([r["cheap1_cost"] + r["b_cost"] for r in rs])  # haiku + B both run
        front_cost = np.mean([r["frontier_cost"] for r in rs])
        g = front_cost / cheap_cost if cheap_cost else float("inf")
        wfrac = np.mean([not r["cheap_correct"] for r in rs])
        gated_cost = np.mean([(r["cheap1_cost"] + r["b_cost"]) + (r["frontier_cost"] if r["disagree"] else 0) for r in rs])
        gated_acc = np.mean([r["frontier_correct"] if r["disagree"] else r["cheap_correct"] for r in rs])
        front_acc = np.mean([r["frontier_correct"] for r in rs])
        realized_go = (gated_cost < front_cost) and (gated_acc >= front_acc - 0.03)
        cw = np.array([not r["cheap_correct"] for r in rs]); fc = np.array([r["frontier_cost"] for r in rs])
        C_w = fc[cw].mean() / front_cost if (cw.sum() and front_cost > 0) else 0.0
        C_w_max = (1 - 1 / g) / wfrac if (g > 1 and wfrac > 0) else (float("inf") if g > 1 else 0.0)
        pred_go = (g > 1) and (C_w < C_w_max)
        h3.append({"workload": w, "n": len(rs), "g": float(g), "C_w": float(C_w), "C_w_max": float(C_w_max),
                   "escalation_rate": float(np.mean([r["disagree"] for r in rs])),
                   "gated_acc": float(gated_acc), "frontier_acc": float(front_acc),
                   "predicted_go": bool(pred_go), "realized_go": bool(realized_go), "match": bool(pred_go == realized_go)})

    report = {"date": CONFIG["price_snapshot_date"], "model_B": BMODEL, "B_price": BPRICE,
              "H1": {"ce_n": int(ce.sum()), "b_correct_on_ce": h1_caught, "recall": h1_recall,
                     "ci95": h1_ci, "plateau": 0.21},
              "H2": {"majority_disagree_rate": float(maj_disagree), "minority_disagree_rate": float(min_disagree),
                     "overall_escalation_rate": float(escalation), "precision": float(prec)},
              "H3": h3}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase2_7_eval.json").write_text(json.dumps(report, indent=2, default=float))
    verdict(report)


def verdict(rep):
    h1, h2, h3 = rep["H1"], rep["H2"], rep["H3"]
    H1_pass = h1["ce_n"] >= 28 and h1["recall"] >= 0.30
    H1_strong = H1_pass and (h1["ci95"][0] or 0) > 0.21
    H2_pass = (h2["majority_disagree_rate"] < h2["minority_disagree_rate"] - 1e-9) and (h2["overall_escalation_rate"] < 0.40)
    H3_pass = all(x["match"] for x in h3) and any(x["realized_go"] for x in h3)

    L = ["# Phase 2.7 — Capable Cross-Provider Disagreement (frozen pre-reg)\n",
         f"Date {rep['date']}. Cheap B = **{rep['model_B']}** "
         f"(${rep['B_price']['input']}/{rep['B_price']['output']} per MTok). Plateau to break: 0.21.\n",
         "## H1 — genuine independence (B actually correct where haiku confidently wrong)",
         f"- of {h1['ce_n']} confident errors, gpt-4o-mini correct on **{h1['b_correct_on_ce']}** "
         f"=> recall **{h1['recall']:.2f}** (95% CI {h1['ci95'][0]:.2f}-{h1['ci95'][1]:.2f})",
         f"- threshold >=0.30 -> **{'PASS' if H1_pass else 'FAIL'}**"
         + (" (STRONG: CI excludes 0.21)" if H1_strong else (" (WEAK: CI overlaps 0.21)" if H1_pass else "")) + "\n",
         "## H2 — precision / everything-flagged guard (DECISIVE; the test 2.6 failed)",
         f"- disagreement on the MAJORITY (haiku-right): **{h2['majority_disagree_rate']:.2f}**",
         f"- disagreement on the minority (confident errors): {h2['minority_disagree_rate']:.2f}",
         f"- overall escalation rate: **{h2['overall_escalation_rate']:.2f}**  | signal precision: {h2['precision']:.2f}",
         f"- threshold: majority<minority AND escalation<0.40 -> "
         f"**{'PASS — precise signal' if H2_pass else 'FAIL — blanket disagreement (2.6 artifact recurs)'}**\n",
         "## H3 — economics (g with gpt-4o-mini price)",
         "| workload | n | g | escalate | gated/front acc | C_w | C_w_max | pred | real | match |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for x in h3:
        L.append(f"| {x['workload']} | {x['n']} | {x['g']:.2f} | {x['escalation_rate']:.0%} | "
                 f"{x['gated_acc']:.2f}/{x['frontier_acc']:.2f} | {x['C_w']:.2f} | {x['C_w_max']:.2f} | "
                 f"{'GO' if x['predicted_go'] else 'NO'} | {'GO' if x['realized_go'] else 'NO'} | {'OK' if x['match'] else 'MISS'} |")
    L.append(f"\n- rule matches realized + >=1 GO: **{'PASS' if H3_pass else 'FAIL'}**\n")
    L.append("## Verdict\n")
    if H1_pass and H2_pass and H3_pass:
        L.append(f"**CEILING BROKEN — {'green-light' if H1_strong else 'weak green-light for'} narrowed Phase 3.** "
                 f"A capable cross-provider cheap model gives independent ({h1['recall']:.2f}) AND precise "
                 f"(majority-disagree {h2['majority_disagree_rate']:.2f}, escalation {h2['overall_escalation_rate']:.2f}) "
                 f"minority detection above the 0.21 plateau, and it pays. "
                 + ("Provably above plateau (CI excludes 0.21)." if H1_strong else "Weak: CI overlaps 0.21."))
    else:
        fails = [h for h, ok in [("H1", H1_pass), ("H2", H2_pass), ("H3", H3_pass)] if not ok]
        L.append(f"**CEILING CONFIRMED — consolidate.** Failed: {', '.join(fails)}. "
                 + ("H2 FAIL: gpt-4o-mini also disagrees on much of the majority -> the 2.6 blanket-flag artifact "
                    "recurs even with a capable model; precision too low to pay. " if not H2_pass else "")
                 + ("H1 FAIL: genuine independence stays at/under the 0.21 plateau -> three mechanisms now confirm it. " if not H1_pass else "")
                 + ("H3 FAIL: precise+independent but doesn't pay at current prices. " if not H3_pass and H1_pass and H2_pass else "")
                 + "Cheap minority-detection ceiling stands at ~0.21.")
    (RESULTS / "PHASE2_7_RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n=== PHASE 2.7 VERDICT ===")
    print(f"  H1 genuine recall {h1['recall']:.2f} (n={h1['ce_n']}, CI {h1['ci95'][0]:.2f}-{h1['ci95'][1]:.2f}) -> {'PASS' if H1_pass else 'FAIL'}{' STRONG' if H1_strong else ''}")
    print(f"  H2 majority-disagree {h2['majority_disagree_rate']:.2f} vs minority {h2['minority_disagree_rate']:.2f}, escalate {h2['overall_escalation_rate']:.2f} -> {'PASS' if H2_pass else 'FAIL'}")
    print(f"  H3 economics -> {'PASS' if H3_pass else 'FAIL'}")
    print(f"  -> {'CEILING BROKEN' if (H1_pass and H2_pass and H3_pass) else 'CEILING CONFIRMED'}")


if __name__ == "__main__":
    main()
