"""Phase 2.5 — stress-test the Phase 2 breakthrough (frozen pre-reg).

H1: replicate self-verification confident-error recall at n>=30 (pooled).
H2: asymmetry vs leakage — recall on non-format-checkable CE vs format-checkable.
H3: re-anchor g and re-check the g-aware rule across the workload spread.
"""

from __future__ import annotations

import glob
import json
import math as _math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from harness import CONFIG, run_model, cost_of
from checkers import get_checker, classification_check, numeric_check
import workloads_lib as WL

ROOT = Path(__file__).resolve().parent
ROUTER = ROOT.parent / "router"
RESULTS = ROOT / "results"
CHEAP = CONFIG["cheap_model"]
FRONTIER = CONFIG["frontier_model"]
K = 3
sys.path.insert(0, str(ROUTER))
from checker import is_correct as math_is_correct, extract_answer as math_extract

VERIFY_SYS = "You check whether a proposed answer to a question is correct. Reply with exactly one word: CORRECT or INCORRECT."

# format_checkable = answer verifiable by structure alone (numeric/boxed)
FORMAT_CHECKABLE = {"math": True, "gsm8k": True, "arc_challenge": False, "squad": False}


def checker_for(name):
    if name in ("math",):
        return math_is_correct
    if name == "gsm8k":
        return numeric_check
    return classification_check


def consensus(answers, eq):
    valid = [a for a in answers if a and a.strip()]
    if not valid:
        return "", 0.0
    clusters = []
    for a in valid:
        for c in clusters:
            if eq(a, c[0]):
                c[1] += 1
                break
        else:
            clusters.append([a, 1])
    best = max(clusters, key=lambda c: c[1])
    return best[0], best[1] / len(answers)


def verify_prompt(item, maj):
    return f"{item['prompt']}\nProposed answer: {maj}\nIs the proposed answer correct?"


def run_fresh(spec):
    """k=3 cheap + frontier + self-verify on a fresh workload."""
    name = spec.name
    eq = checker_for(name)
    check = get_checker(spec.checker_kind)

    def cheap_k(it, k):
        return run_model(name, it, CHEAP, system=spec.system, check_fn=check,
                         k_index=k, temperature=CONFIG["temperature"], max_tokens=spec.max_tokens)

    def front(it):
        return run_model(name, it, FRONTIER, system=spec.system, check_fn=check,
                         k_index=0, max_tokens=spec.max_tokens)

    recs = {}
    with ThreadPoolExecutor(max_workers=CONFIG["concurrency"]) as ex:
        futs = {}
        for it in spec.items:
            for k in range(K):
                futs[ex.submit(cheap_k, it, k)] = (it["id"], "c", k)
            futs[ex.submit(front, it)] = (it["id"], "f", 0)
        res = {}
        for fu in as_completed(futs):
            res[futs[fu]] = fu.result()
    by = {it["id"]: it for it in spec.items}
    # consensus + verify (verify needs majority, so second pass)
    verify_futs = {}
    with ThreadPoolExecutor(max_workers=CONFIG["concurrency"]) as ex:
        majs = {}
        for it in spec.items:
            answers = [res[(it["id"], "c", k)]["answer"] for k in range(K)]
            maj, agr = consensus(answers, eq)
            majs[it["id"]] = (maj, agr, answers)
            vit = {"id": it["id"], "prompt": verify_prompt(it, maj), "gold": "CORRECT"}
            verify_futs[ex.submit(run_model, name, vit, CHEAP, system=VERIFY_SYS,
                                   check_fn=lambda a, g: "correct" in a.lower() and "incorrect" not in a.lower(),
                                   k_index=0, temperature=0.0, max_tokens=8)] = it["id"]
        vres = {verify_futs[f]: f.result() for f in as_completed(verify_futs)}

    out = []
    for it in spec.items:
        maj, agr, answers = majs[it["id"]]
        cheap_correct = bool(eq(maj, it["gold"])) if maj else False
        cheap_cost = sum(res[(it["id"], "c", k)]["cost"] for k in range(K))
        fr = res[(it["id"], "f", 0)]
        v = vres[it["id"]]
        out.append({
            "workload": name, "id": it["id"], "format_checkable": FORMAT_CHECKABLE[name],
            "agreement": agr, "cheap_correct": cheap_correct,
            "verify_incorrect": "incorrect" in v["answer"].lower(),
            "verify_cost": v["cost"], "cheap_cost_k3": cheap_cost,
            "frontier_correct": bool(fr["correct"]), "frontier_cost": fr["cost"],
        })
    return out


def load_math_records():
    L = {json.loads(l)["id"]: json.loads(l) for l in open(ROUTER / "results" / "routing_labels.jsonl", encoding="utf-8")}
    # k3 cheap cost from gating cache; verify from advisor math_selfverify; frontier sonnet-high
    chcost = {}
    for f in glob.glob(str(ROUTER / "cache" / "*claude-haiku-4-5__sc*")):
        r = json.loads(open(f).read())
        chcost.setdefault(r["id"], 0.0)
        chcost[r["id"]] += cost_of(CHEAP, r["input_tokens"], r["output_tokens"])
    verify = {}
    for f in glob.glob(str(ROOT / "cache" / "math_selfverify__*")):
        r = json.loads(open(f).read())
        verify[r["id"]] = ("incorrect" in r["answer"].lower(), r["cost"])
    fcost = {}
    for line in open(ROUTER / "results" / "baseline_runs.jsonl"):
        r = json.loads(line)
        if r["budget"] == "high":
            fcost[r["id"]] = (bool(r["correct"]), cost_of(FRONTIER, r["input_tokens"], r["output_tokens"]))
    out = []
    for i, l in L.items():
        if i not in verify or i not in fcost:
            continue
        vi, vc = verify[i]
        out.append({
            "workload": "math", "id": i, "format_checkable": True,
            "agreement": l["agreement"], "cheap_correct": l["cheap_correct"],
            "verify_incorrect": vi, "verify_cost": vc, "cheap_cost_k3": chcost.get(i, 0.0),
            "frontier_correct": l["frontier_correct"], "frontier_cost": fcost[i][1],
        })
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * _math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    # gate
    specs = [WL.load_gsm8k(CONFIG["phase25_n"]), WL.load_arc_challenge(), WL.load_squad(CONFIG["phase25_n"])]
    # estimate new calls
    def new_calls(spec):
        from harness import _cache_path
        n = 0
        for it in spec.items:
            for k in range(K):
                if not _cache_path(spec.name, CHEAP, spec.system, it["prompt"], k).exists():
                    n += 1
            if not _cache_path(spec.name, FRONTIER, spec.system, it["prompt"], 0).exists():
                n += 1
            n += 1  # verify (approx; key differs)
        return n
    est_calls = sum(new_calls(s) for s in specs)
    est = est_calls * cost_of(CHEAP, 250, 250)  # rough blended
    print(f"phase2.5 fresh workloads: {[s.name for s in specs]}; ~{est_calls} new calls, est ${est:.2f} (ceil ${CONFIG['cost_ceiling_usd']})")
    if est > CONFIG["cost_ceiling_usd"]:
        raise SystemExit("ABORT phase2.5 cost gate")

    records = []
    for s in specs:
        print(f"  running {s.name} ({len(s.items)} items, k=3 + frontier + verify) ...")
        records += run_fresh(s)
    records += load_math_records()

    # ---- confident-error set: cheap unanimous (agreement>=1.0) AND cheap wrong ----
    ce = [r for r in records if r["agreement"] >= 1.0 and not r["cheap_correct"]]
    caught = [r for r in ce if r["verify_incorrect"]]
    rec = len(caught) / len(ce) if ce else 0.0
    ci = wilson(len(caught), len(ce))

    # H2 split
    ce_fmt = [r for r in ce if r["format_checkable"]]
    ce_non = [r for r in ce if not r["format_checkable"]]
    def grp(g):
        c = sum(r["verify_incorrect"] for r in g)
        return c, len(g), (c / len(g) if g else None), wilson(c, len(g))
    f_c, f_n, f_r, f_ci = grp(ce_fmt)
    n_c, n_n, n_r, n_ci = grp(ce_non)

    # ---- H3: re-anchor g + rule check per workload (combined = selfcons k3 + verify) ----
    from collections import defaultdict
    byw = defaultdict(list)
    for r in records:
        byw[r["workload"]].append(r)
    h3 = []
    for w, rs in byw.items():
        n = len(rs)
        cheap_cost = np.mean([r["cheap_cost_k3"] + r["verify_cost"] for r in rs])  # combined signal cost
        front_cost = np.mean([r["frontier_cost"] for r in rs])
        g = front_cost / cheap_cost if cheap_cost else float("inf")
        wfrac = np.mean([not r["cheap_correct"] for r in rs])
        # realized oracle payoff (escalate iff cheap wrong) vs always-frontier, combined cheap pass
        oracle_cost = np.mean([(r["cheap_cost_k3"] + r["verify_cost"]) + (0 if r["cheap_correct"] else r["frontier_cost"]) for r in rs])
        realized_go = oracle_cost < front_cost
        cheap_wrong = np.array([not r["cheap_correct"] for r in rs])
        fc = np.array([r["frontier_cost"] for r in rs])
        C_w = fc[cheap_wrong].mean() / front_cost if (cheap_wrong.sum() and front_cost > 0) else 0.0
        C_w_max = (1 - 1 / g) / wfrac if (g > 1 and wfrac > 0) else (float("inf") if g > 1 else 0.0)
        pred_go = (g > 1) and (C_w < C_w_max)
        h3.append({"workload": w, "n": n, "g": float(g), "w": float(wfrac), "C_w": float(C_w),
                   "C_w_max": float(C_w_max), "predicted_go": bool(pred_go), "realized_go": bool(realized_go),
                   "match": bool(pred_go == realized_go)})

    report = {
        "date": "2026-06-29", "models": {"cheap": CHEAP, "frontier": FRONTIER},
        "H1": {"ce_n": len(ce), "caught": len(caught), "recall": rec, "ci95": ci},
        "H2": {"format": {"caught": f_c, "n": f_n, "recall": f_r, "ci95": f_ci},
               "non_format": {"caught": n_c, "n": n_n, "recall": n_r, "ci95": n_ci}},
        "H3": h3,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase2_5_eval.json").write_text(json.dumps(report, indent=2, default=float))
    verdict(report)


def verdict(rep):
    h1 = rep["H1"]; h2 = rep["H2"]; h3 = rep["H3"]
    H1_pass = h1["ce_n"] >= 30 and h1["recall"] >= 0.30 and (h1["ci95"][0] or 0) > 0
    nf = h2["non_format"]
    H2_pass = (nf["n"] >= 1 and (nf["recall"] or 0) > 0 and (nf["ci95"][0] or 0) > 0
               and (nf["recall"] or 0) >= (h2["format"]["recall"] or 0) - 0.15)
    H3_pass = all(x["match"] for x in h3) and any(1.2 < x["C_w"] < 2.0 or 1.2 < x["g"] for x in h3)

    L = ["# Phase 2.5 — Stress Test Results (frozen pre-reg)\n",
         f"Date 2026-06-29. cheap={rep['models']['cheap']}, frontier={rep['models']['frontier']}.\n",
         "## H1 — Replication at larger n",
         f"- confident-error set n = **{h1['ce_n']}**, self-verify caught **{h1['caught']}** "
         f"=> recall **{h1['recall']:.2f}** (95% CI {h1['ci95'][0]:.2f}–{h1['ci95'][1]:.2f})",
         f"- threshold: n>=30, recall>=0.30, CI lower>0 -> **{'PASS' if H1_pass else 'FAIL'}**\n",
         "## H2 — Asymmetry vs leakage (decisive)",
         f"- format-checkable CE: recall {h2['format']['recall']} ({h2['format']['caught']}/{h2['format']['n']})",
         f"- **non-format-checkable CE: recall {nf['recall']} ({nf['caught']}/{nf['n']}), "
         f"CI {None if nf['ci95'][0] is None else round(nf['ci95'][0],2)}–{None if nf['ci95'][1] is None else round(nf['ci95'][1],2)}**",
         f"- threshold: non-format recall>0, CI lower>0, within 0.15 of format -> **{'PASS — asymmetry real' if H2_pass else 'FAIL — leakage-bound'}**\n",
         "## H3 — Re-anchored g + rule check (per workload, combined = k3 selfcons + verify)",
         "| workload | n | g | C_w | C_w_max | predicted | realized | match |",
         "|---|---|---|---|---|---|---|---|"]
    for x in h3:
        L.append(f"| {x['workload']} | {x['n']} | {x['g']:.2f} | {x['C_w']:.2f} | "
                 f"{x['C_w_max']:.2f} | {'GO' if x['predicted_go'] else 'NO'} | "
                 f"{'GO' if x['realized_go'] else 'NO'} | {'OK' if x['match'] else 'MISS'} |")
    L.append(f"\n- rule predicts realized GO/NO-GO on all workloads: **{'PASS' if H3_pass else 'FAIL'}** "
             f"(every verdict stated at its g under 2026-06-29 prices).\n")
    overall = H1_pass and H2_pass and H3_pass
    L.append("## Phase-3 go/no-go\n")
    if overall:
        L.append("**GREEN-LIGHT Phase 3.** H1+H2+H3 all PASS: the self-verification win replicates at "
                 "n>=30, survives where format leakage cannot explain it, and the g-aware rule holds "
                 "across the spread. The a-priori-C theory may be built on this foundation.")
    else:
        fails = [h for h, ok in [("H1", H1_pass), ("H2", H2_pass), ("H3", H3_pass)] if not ok]
        L.append(f"**HOLD — do not advance to Phase 3 as-is.** Failed: {', '.join(fails)}. "
                 + ("H2 FAIL => the verification win is leakage-bound; retract the general claim and "
                    "reframe Phase 3 to format-checkable workloads. " if not H2_pass else "")
                 + ("H1 FAIL => the breakthrough does not replicate at n; revert to suggestive. " if not H1_pass else "")
                 + ("H3 FAIL => the break-even formula mispredicts; revise before Phase 3. " if not H3_pass else ""))
    (RESULTS / "PHASE2_5_RESULTS.md").write_text("\n".join(L), encoding="utf-8")

    print("\n=== PHASE 2.5 VERDICT ===")
    print(f"  H1 recall {h1['recall']:.2f} (n={h1['ce_n']}, CI {h1['ci95'][0]:.2f}-{h1['ci95'][1]:.2f}) -> {'PASS' if H1_pass else 'FAIL'}")
    print(f"  H2 non-format recall {nf['recall']} ({nf['caught']}/{nf['n']}) -> {'PASS' if H2_pass else 'FAIL'}")
    print(f"  H3 rule match all workloads -> {'PASS' if H3_pass else 'FAIL'}")
    print(f"  PHASE 3: {'GREEN-LIGHT' if overall else 'HOLD'}")


if __name__ == "__main__":
    main()
