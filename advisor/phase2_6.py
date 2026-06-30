"""Phase 2.6 — cross-model disagreement (frozen pre-reg).

Second cheap model = LOCAL open-weight Qwen2.5-1.5B-Instruct ($0). CAVEAT: weaker
than haiku, NOT comparably capable — so on reasoning tasks its errors may be
"wrong everywhere" rather than independent. H2 measures exactly that.

H1: disagreement CE recall >= 0.30, CI excludes the 0.21 self-verify incumbent.
H2: error overlap on the CE set (independence test) — the mechanism gate.
H3: g with the 2-model signal (B free) + g-aware rule check.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math as _math
import sys
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
K = 3
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_TAG = "qwen2.5-1.5b-local"
sys.path.insert(0, str(ROUTER))
from checker import is_correct as math_is_correct

FORMAT_CHECKABLE = {"math": True, "gsm8k": True, "arc_challenge": False, "squad": False}


def checker_for(name):
    if name == "math":
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


# ---------- local model ----------
_lm = None
_tok = None


def _load_local():
    global _lm, _tok
    if _lm is None:
        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        _lm = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, torch_dtype="auto").eval()
    return _lm, _tok


def _local_cache(workload, system, prompt):
    h = hashlib.sha256((LOCAL_TAG + system + prompt).encode()).hexdigest()[:12]
    return CACHE / f"{workload}__{LOCAL_TAG}__{h}.json"


def run_local(workload, item, system, max_tokens, force=False):
    cp = _local_cache(workload, system, item["prompt"])
    if cp.exists() and not force:
        return json.loads(cp.read_text())
    import torch
    model, tok = _load_local()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": item["prompt"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    rec = {"workload": workload, "id": item["id"], "model": LOCAL_TAG, "answer": ans}
    cp.write_text(json.dumps(rec))
    return rec


# ---------- haiku-side per-item (cached) ----------

def fresh_items(spec):
    """rebuild haiku majority + agreement + frontier from cache (no new spend)."""
    eq = checker_for(spec.name)
    check = get_checker(spec.checker_kind)
    out = []
    for it in spec.items:
        answers = [run_model(spec.name, it, CHEAP, system=spec.system, check_fn=check,
                             k_index=k, temperature=CONFIG["temperature"], max_tokens=spec.max_tokens)["answer"]
                   for k in range(K)]
        maj, agr = consensus(answers, eq)
        fr = run_model(spec.name, it, FRONTIER, system=spec.system, check_fn=check, k_index=0, max_tokens=spec.max_tokens)
        # cheap k=1 cost (single haiku) for g; B is free
        c1 = run_model(spec.name, it, CHEAP, system=spec.system, check_fn=check, k_index=0,
                       temperature=CONFIG["temperature"], max_tokens=spec.max_tokens)
        out.append({"workload": spec.name, "id": it["id"], "item": it, "system": spec.system,
                    "maj": maj, "agreement": agr, "cheap_correct": bool(eq(maj, it["gold"])) if maj else False,
                    "gold": it["gold"], "format_checkable": FORMAT_CHECKABLE[spec.name],
                    "frontier_correct": bool(fr["correct"]), "frontier_cost": fr["cost"],
                    "cheap1_cost": c1["cost"], "max_tokens": spec.max_tokens})
    return out


def math_items():
    L = {json.loads(l)["id"]: json.loads(l) for l in open(ROUTER / "results" / "routing_labels.jsonl", encoding="utf-8")}
    sample = {json.loads(l)["id"]: json.loads(l) for l in open(ROUTER / "data" / "sample.jsonl", encoding="utf-8")}
    # haiku majority answer from gating cache; cost of one sample
    samp = {}
    for f in glob.glob(str(ROUTER / "cache" / "*claude-haiku-4-5__sc*")):
        r = json.loads(open(f).read()); samp.setdefault(r["id"], []).append(r)
    fcost = {}
    for line in open(ROUTER / "results" / "baseline_runs.jsonl"):
        r = json.loads(line)
        if r["budget"] == "high":
            fcost[r["id"]] = cost_of(FRONTIER, r["input_tokens"], r["output_tokens"])
    out = []
    for i, l in L.items():
        if i not in samp or i not in fcost:
            continue
        anss = [s["answer"] for s in samp[i][:3]]
        maj, agr = consensus(anss, math_is_correct)
        c1 = cost_of(CHEAP, samp[i][0]["input_tokens"], samp[i][0]["output_tokens"])
        p = sample[i]
        out.append({"workload": "math", "id": i, "item": {"id": i, "prompt": p["problem"], "gold": p["answer"]},
                    "system": "Solve the problem. Give the final answer as \\boxed{...}.", "maj": maj,
                    "agreement": l["agreement"], "cheap_correct": l["cheap_correct"], "gold": p["answer"],
                    "format_checkable": True, "frontier_correct": l["frontier_correct"],
                    "frontier_cost": fcost[i], "cheap1_cost": c1, "max_tokens": 256})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * _math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    specs = [WL.load_gsm8k(CONFIG["phase25_n"]), WL.load_arc_challenge(), WL.load_squad(CONFIG["phase25_n"])]
    print("rebuilding haiku-side records from cache (no spend) ...")
    records = []
    for s in specs:
        records += fresh_items(s)
    records += math_items()

    # Bound local-B to what the metrics need (CPU-feasible): all CE items + a
    # capped sample per workload for H3. B is weak — cap tokens at 64.
    from collections import defaultdict
    ce_ids = {r["id"] for r in records if r["agreement"] >= 1.0 and not r["cheap_correct"]}
    per_w = defaultdict(int)
    b_targets = []
    for r in records:
        if r["id"] in ce_ids:
            b_targets.append(r)
        elif per_w[r["workload"]] < 30:
            per_w[r["workload"]] += 1
            b_targets.append(r)
    tset = {r["id"] for r in b_targets}
    print(f"  {len(records)} items; running local B on {len(b_targets)} (CE + <=30/workload), tokens<=64 ...")

    eqs = {r["id"]: checker_for(r["workload"]) for r in records}
    done = 0
    for r in records:
        if r["id"] not in tset:
            r["b_answer"] = None; r["b_correct"] = None; r["disagree"] = None
            continue
        b = run_local(r["workload"], r["item"], r["system"], 64)
        eq = eqs[r["id"]]
        r["b_answer"] = b["answer"]
        r["b_correct"] = bool(eq(b["answer"], r["gold"]))
        r["disagree"] = not eq(r["maj"], b["answer"]) if r["maj"] else True
        done += 1
        if done % 30 == 0 or done == len(b_targets):
            print(f"    local B {done}/{len(b_targets)}")
    records = [r for r in records if r["disagree"] is not None]  # B-covered only

    # confident-error set: haiku unanimous AND haiku wrong (same as Phase 2.5, n~28)
    ce = [r for r in records if r["agreement"] >= 1.0 and not r["cheap_correct"]]
    caught = [r for r in ce if r["disagree"]]
    rec = len(caught) / len(ce) if ce else 0.0
    ci = wilson(len(caught), len(ce))

    # H2: error overlap on CE (B also wrong) + independence
    b_also_wrong = [r for r in ce if not r["b_correct"]]
    overlap = len(b_also_wrong) / len(ce) if ce else None
    # of CE where B is RIGHT (independent), disagreement should fire (B differs)
    b_right = [r for r in ce if r["b_correct"]]
    indep_caught = sum(r["disagree"] for r in b_right)

    # H3: per-workload g (cheap = 1 haiku; B free) + realized payoff with disagreement signal
    from collections import defaultdict
    byw = defaultdict(list)
    for r in records:
        byw[r["workload"]].append(r)
    h3 = []
    for w, rs in byw.items():
        n = len(rs)
        cheap_cost = np.mean([r["cheap1_cost"] for r in rs])  # B is free
        front_cost = np.mean([r["frontier_cost"] for r in rs])
        g = front_cost / cheap_cost if cheap_cost else float("inf")
        wfrac = np.mean([not r["cheap_correct"] for r in rs])
        # gated by disagreement: escalate iff disagree
        gated_cost = np.mean([r["cheap1_cost"] + (r["frontier_cost"] if r["disagree"] else 0) for r in rs])
        gated_acc = np.mean([r["frontier_correct"] if r["disagree"] else r["cheap_correct"] for r in rs])
        front_acc = np.mean([r["frontier_correct"] for r in rs])
        realized_go = gated_cost < front_cost and gated_acc >= front_acc - 0.03
        cw = np.array([not r["cheap_correct"] for r in rs]); fc = np.array([r["frontier_cost"] for r in rs])
        C_w = fc[cw].mean() / front_cost if (cw.sum() and front_cost > 0) else 0.0
        C_w_max = (1 - 1 / g) / wfrac if (g > 1 and wfrac > 0) else (float("inf") if g > 1 else 0.0)
        pred_go = (g > 1) and (C_w < C_w_max)
        h3.append({"workload": w, "n": n, "g": float(g), "C_w": float(C_w), "C_w_max": float(C_w_max),
                   "escalation_rate": float(np.mean([r["disagree"] for r in rs])),
                   "gated_acc": float(gated_acc), "frontier_acc": float(front_acc),
                   "predicted_go": bool(pred_go), "realized_go": bool(realized_go), "match": bool(pred_go == realized_go)})

    report = {"date": "2026-06-29", "model_B": LOCAL_MODEL, "caveat": "B is a 1.5B local model, weaker than haiku; not comparably capable",
              "H1": {"ce_n": len(ce), "caught": len(caught), "recall": rec, "ci95": ci, "incumbent_selfverify": 0.21},
              "H2": {"error_overlap_B_also_wrong": overlap, "ce_n": len(ce),
                     "b_right_on_ce": len(b_right), "of_those_disagreement_fired": indep_caught},
              "H3": h3}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase2_6_eval.json").write_text(json.dumps(report, indent=2, default=float))
    verdict(report)


def verdict(rep):
    h1 = rep["H1"]; h2 = rep["H2"]; h3 = rep["H3"]
    H1_pass = h1["ce_n"] >= 28 and h1["recall"] >= 0.30
    H1_strong = H1_pass and (h1["ci95"][0] or 0) > 0.21
    ov = h2["error_overlap_B_also_wrong"]
    H2_pass = (ov is not None and ov < 0.85)  # materially below 1.0 = some independence
    H3_pass = all(x["match"] for x in h3)

    L = ["# Phase 2.6 — Cross-Model Disagreement (frozen pre-reg)\n",
         f"Date 2026-06-29. Model B = **{rep['model_B']}** (LOCAL, $0). "
         f"CAVEAT: {rep['caveat']}.\n",
         "## H1 — disagreement CE recall vs the 0.30 bar (incumbent self-verify 0.21)",
         f"- CE set n=**{h1['ce_n']}**, disagreement caught **{h1['caught']}** => recall **{h1['recall']:.2f}** "
         f"(95% CI {h1['ci95'][0]:.2f}-{h1['ci95'][1]:.2f})",
         f"- threshold recall>=0.30 -> **{'PASS' if H1_pass else 'FAIL'}**"
         + (" (STRONG: CI excludes 0.21)" if H1_strong else (" (WEAK: CI overlaps 0.21)" if H1_pass else "")) + "\n",
         "## H2 — error independence (the mechanism gate)",
         f"- of {h2['ce_n']} confident errors (haiku unanimous-wrong), B **also wrong** on "
         f"{h2['error_overlap_B_also_wrong']:.2f} => independence "
         f"{'present' if H2_pass else 'absent (shared failure)'}",
         f"- B was RIGHT on {h2['b_right_on_ce']} of the CE items; disagreement fired on "
         f"{h2['of_those_disagreement_fired']} of those",
         f"- threshold overlap<0.85 -> **{'PASS — partial independence' if H2_pass else 'FAIL — shared failure modes'}**\n",
         "## H3 — g (B free) + rule check",
         "| workload | n | g | C_w | C_w_max | escalate | gated/front acc | pred | real | match |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for x in h3:
        L.append(f"| {x['workload']} | {x['n']} | {x['g']:.2f} | {x['C_w']:.2f} | {x['C_w_max']:.2f} | "
                 f"{x['escalation_rate']:.0%} | {x['gated_acc']:.2f}/{x['frontier_acc']:.2f} | "
                 f"{'GO' if x['predicted_go'] else 'NO'} | {'GO' if x['realized_go'] else 'NO'} | {'OK' if x['match'] else 'MISS'} |")
    L.append(f"\n- rule matches realized on all workloads: **{'PASS' if H3_pass else 'FAIL'}** (verdicts at g under 2026-06-29 prices).\n")
    L.append("## Verdict\n")
    if H1_pass and H2_pass and H3_pass:
        L.append(f"**{'GREEN-LIGHT narrowed Phase 3' if H1_strong else 'WEAK GREEN-LIGHT'}.** Cross-model "
                 f"disagreement clears the 0.30 bar self-verification failed, with partial error "
                 f"independence and preserved economics. " + ("Provably better than self-verify (CI excludes 0.21)."
                 if H1_strong else "Clears the bar but not provably better than self-verify (CI overlap) — report as weak.")
                 + " NOTE: B is a weak local model; a comparable hosted model could differ.")
    else:
        fails = [h for h, ok in [("H1", H1_pass), ("H2", H2_pass), ("H3", H3_pass)] if not ok]
        L.append(f"**CONSOLIDATE — ceiling found.** Failed: {', '.join(fails)}. "
                 + ("H2 FAIL: the local model shares haiku's failures (or is just wrong everywhere) — no "
                    "independence to exploit; disagreement can't help. " if not H2_pass else "")
                 + ("H1 FAIL: disagreement recall stays at/under the cheap-signal plateau. " if not H1_pass else "")
                 + "Cheap minority-detection plateaus ~0.21-0.30; map this as the stated ceiling. "
                 + "(Caveat: tested with a weak local B; a comparable hosted cheap model is the unrun stronger test.)")
    (RESULTS / "PHASE2_6_RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n=== PHASE 2.6 VERDICT ===")
    print(f"  H1 recall {h1['recall']:.2f} (n={h1['ce_n']}, CI {h1['ci95'][0]:.2f}-{h1['ci95'][1]:.2f}) -> {'PASS' if H1_pass else 'FAIL'}{' STRONG' if H1_strong else ''}")
    print(f"  H2 error overlap {ov:.2f} -> {'PASS' if H2_pass else 'FAIL'}")
    print(f"  H3 rule match -> {'PASS' if H3_pass else 'FAIL'}")
    print(f"  -> {'GREEN-LIGHT' if (H1_pass and H2_pass and H3_pass) else 'CONSOLIDATE'}")


if __name__ == "__main__":
    main()
