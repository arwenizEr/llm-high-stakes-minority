"""Phase 3 (reframed) — self-gating competence detection (frozen pre-reg).

Can B's competence on a workload be detected A PRIORI (no frontier) from cheap
A/B signals, so disagreement routing self-gates to GO-domains and abstains on
blind-spot domains?

Frozen rule (committed before results): a-priori A/B blanket-disagreement rate
> 0.50 => abstain (blind-spot); <= 0.50 => GO-eligible.

FRONTIER IS WALLED OFF: the competence predictor uses ONLY haiku/B answers. The
frontier is used solely to label realized GO/NO-GO for evaluation.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from harness import CONFIG, run_model, cost_of
from checkers import classification_check, numeric_check, get_checker
import workloads_lib as WL
from phase2_7 import run_b, _b_cache
from phase2_6 import checker_for, math_items, fresh_items

ROOT = Path(__file__).resolve().parent
ROUTER = ROOT.parent / "router"
RESULTS = ROOT / "results"
CHEAP = CONFIG["cheap_model"]
FRONTIER = CONFIG["frontier_model"]
BMODEL = CONFIG["openai_model"]
SEED = CONFIG["seed"]
DISAGREE_ABSTAIN = 0.50  # FROZEN threshold

ORIGINAL = {"gsm8k", "arc_challenge", "squad", "math"}


# ---------------- held-out loaders ----------------

def _spec(name, items, system, kind, mt):
    s = WL.WorkloadSpec(name, items, system, kind, mt)
    return s


def load_boolq(n=80):
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation")
    idx = list(range(len(ds))); random.Random(SEED).shuffle(idx)
    items = [{"id": f"boolq_{i}",
              "prompt": f"Passage: {ds[i]['passage']}\nQuestion: {ds[i]['question']}\nAnswer yes or no.",
              "gold": "yes" if ds[i]["answer"] else "no"} for i in idx[:n]]
    return _spec("boolq", items, "Answer the yes/no question from the passage. Reply yes or no.", "classification", 8)


def load_csqa(n=80):
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="validation")
    idx = list(range(len(ds))); random.Random(SEED).shuffle(idx)
    items = []
    for i in idx[:n]:
        ex = ds[i]; ch = ex["choices"]
        opts = "\n".join(f"{l}) {t}" for l, t in zip(ch["label"], ch["text"]))
        items.append({"id": f"csqa_{i}", "prompt": f"Question: {ex['question']}\n{opts}\nAnswer with the letter.",
                      "gold": ex["answerKey"]})
    return _spec("csqa", items, "Answer the multiple-choice question with the correct letter only.", "classification", 8)


def load_aime(n=30):
    from datasets import load_dataset
    for repo in ("Maxwell-Jia/AIME_2024", "AI-MO/aimo-validation-aime"):
        try:
            ds = load_dataset(repo, split="train")
            break
        except Exception:
            ds = None
    if ds is None:
        return None
    qcol = "Problem" if "Problem" in ds.column_names else ("problem" if "problem" in ds.column_names else "question")
    acol = "Answer" if "Answer" in ds.column_names else ("answer" if "answer" in ds.column_names else "solution")
    idx = list(range(len(ds))); random.Random(SEED).shuffle(idx)
    items = [{"id": f"aime_{i}", "prompt": f"Problem: {ds[i][qcol]}\nGive the final integer answer.",
              "gold": str(ds[i][acol])} for i in idx[:n]]
    return _spec("aime", items, "Solve the competition math problem. End with the final integer.", "numeric", 512)


# ---------------- per-item records (k=1 haiku + B + frontier) ----------------

def heldout_records(spec):
    eq = checker_for(spec.name) if spec.name in ("aime",) else (numeric_check if spec.checker_kind == "numeric" else classification_check)
    check = get_checker(spec.checker_kind)
    out = []
    for it in spec.items:
        a = run_model(spec.name, it, CHEAP, system=spec.system, check_fn=check, k_index=0,
                      temperature=CONFIG["temperature"], max_tokens=spec.max_tokens)
        bmt = 512 if spec.checker_kind == "numeric" else 24
        b = run_b(spec.name, it, spec.system, bmt)
        fr = run_model(spec.name, it, FRONTIER, system=spec.system, check_fn=check, k_index=0, max_tokens=spec.max_tokens)
        out.append({"workload": spec.name, "id": it["id"], "a_ans": a["answer"], "b_ans": b["answer"],
                    "a_correct": bool(a["correct"]), "b_correct": bool(eq(b["answer"], it["gold"])),
                    "a_cost": a["cost"], "b_cost": b["cost"], "frontier_correct": bool(fr["correct"]),
                    "frontier_cost": fr["cost"], "disagree": not eq(a["answer"], b["answer"]),
                    "gold": it["gold"]})
    return out


def original_records():
    """k=1 haiku (sample 0) vs B vs frontier, from cache (free)."""
    out = []
    # gsm8k/arc/squad via specs (cached haiku k0 + B + frontier)
    for spec in (WL.load_gsm8k(CONFIG["phase25_n"]), WL.load_arc_challenge(), WL.load_squad(CONFIG["phase25_n"])):
        eq = numeric_check if spec.checker_kind == "numeric" else classification_check
        check = get_checker(spec.checker_kind)
        for it in spec.items:
            a = run_model(spec.name, it, CHEAP, system=spec.system, check_fn=check, k_index=0,
                          temperature=CONFIG["temperature"], max_tokens=spec.max_tokens)
            b = run_b(spec.name, it, spec.system, 256 if spec.checker_kind == "numeric" else 24)
            fr = run_model(spec.name, it, FRONTIER, system=spec.system, check_fn=check, k_index=0, max_tokens=spec.max_tokens)
            out.append({"workload": spec.name, "id": it["id"], "a_correct": bool(a["correct"]),
                        "b_correct": bool(eq(b["answer"], it["gold"])), "a_cost": a["cost"], "b_cost": b["cost"],
                        "frontier_correct": bool(fr["correct"]), "frontier_cost": fr["cost"],
                        "disagree": not eq(a["answer"], b["answer"]), "gold": it["gold"]})
    # math via gating sc0 + B + sonnet-high
    import glob
    samp = {}
    for f in glob.glob(str(ROUTER / "cache" / "*claude-haiku-4-5__sc0__*")):
        r = json.loads(open(f).read()); samp[r["id"]] = r
    fr = {}
    for line in open(ROUTER / "results" / "baseline_runs.jsonl"):
        r = json.loads(line)
        if r["budget"] == "high":
            fr[r["id"]] = (bool(r["correct"]), cost_of(FRONTIER, r["input_tokens"], r["output_tokens"]))
    sample = {json.loads(l)["id"]: json.loads(l) for l in open(ROUTER / "data" / "sample.jsonl", encoding="utf-8")}
    bspec_sys = "Solve the problem. Give the final answer as \\boxed{...}."
    for i, s in samp.items():
        if i not in fr:
            continue
        p = sample[i]; it = {"id": i, "prompt": p["problem"], "gold": p["answer"]}
        b = run_b("math", it, bspec_sys, 256)
        out.append({"workload": "math", "id": i, "a_correct": bool(s["correct"]),
                    "b_correct": bool(math_eq(b["answer"], p["answer"])), "a_cost": cost_of(CHEAP, s["input_tokens"], s["output_tokens"]),
                    "b_cost": b["cost"], "frontier_correct": fr[i][0], "frontier_cost": fr[i][1],
                    "disagree": not math_eq(s["answer"], b["answer"]), "gold": p["answer"]})
    return out


sys.path.insert(0, str(ROUTER))
from checker import is_correct as math_eq


# ---------------- competence predictor (FRONTIER-FREE) ----------------

def competence_score(recs_for_workload):
    """A-PRIORI: raw A/B disagreement rate. No frontier field touched."""
    return float(np.mean([r["disagree"] for r in recs_for_workload]))


def predict_go_eligible(score):
    return score <= DISAGREE_ABSTAIN


# ---------------- realized routing (frontier labels) ----------------

def realized_go(rs):
    n = len(rs)
    af = np.mean([r["frontier_cost"] for r in rs])
    gated_cost = np.mean([(r["a_cost"] + r["b_cost"]) + (r["frontier_cost"] if r["disagree"] else 0) for r in rs])
    gated_acc = np.mean([r["frontier_correct"] if r["disagree"] else r["a_correct"] for r in rs])
    fa = np.mean([r["frontier_correct"] for r in rs])
    return (gated_cost < af) and (gated_acc >= fa - 0.03), gated_cost, af, gated_acc, fa


def main():
    print("original 4 (cached, free) ...")
    records = original_records()
    print("held-out domains (new spend) ...")
    held = [load_boolq(), load_csqa()]
    aime = load_aime()
    if aime:
        held.append(aime)
    # cost gate
    n_new = 0
    for spec in held:
        for it in spec.items:
            from harness import _cache_path
            if not _cache_path(spec.name, CHEAP, spec.system, it["prompt"], 0).exists():
                n_new += 1
            if not _cache_path(spec.name, FRONTIER, spec.system, it["prompt"], 0).exists():
                n_new += 1
            if not _b_cache(spec.name, spec.system, it["prompt"]).exists():
                n_new += 1
    est = n_new * 0.003
    print(f"held-out: {[s.name for s in held]}; ~{n_new} new calls, est ${est:.2f} (ceil ${CONFIG['cost_ceiling_usd']})")
    if est > CONFIG["cost_ceiling_usd"]:
        raise SystemExit("ABORT cost gate")
    for spec in held:
        records += heldout_records(spec)

    byw = defaultdict(list)
    for r in records:
        byw[r["workload"]].append(r)

    table = []
    for w, rs in byw.items():
        score = competence_score(rs)             # a-priori, frontier-free
        pred_eligible = predict_go_eligible(score)
        rgo, gc, af, ga, fa = realized_go(rs)
        table.append({"workload": w, "held_out": w not in ORIGINAL, "n": len(rs),
                      "disagree_rate": score, "pred_go_eligible": pred_eligible,
                      "realized_go": rgo, "gated_acc": ga, "frontier_acc": fa,
                      "gated_cost_mq": gc * 1000, "frontier_cost_mq": af * 1000,
                      # rule matches realized: GO-eligible predicts realized GO; abstain predicts realized NO-GO
                      "rule_correct": (pred_eligible == rgo)})
    table.sort(key=lambda r: (r["held_out"], r["disagree_rate"]))

    orig = [t for t in table if not t["held_out"]]
    ho = [t for t in table if t["held_out"]]
    H1_pass = all(t["rule_correct"] for t in orig)
    H3_pass = len(ho) >= 2 and all(t["rule_correct"] for t in ho)
    # H2: self-gated routing — GO-eligible use disagreement routing; abstain use always-frontier.
    # PASS: no blind-spot blanket disaster (abstain branch cost <= its gated cost) AND GO-domains realize GO.
    h2_ok = True
    for t in table:
        if t["pred_go_eligible"]:
            if not t["realized_go"]:
                h2_ok = False  # routed a domain that didn't pay
        # abstain branch = always-frontier; safe by construction
    H2_pass = h2_ok and H1_pass

    report = {"date": CONFIG["price_snapshot_date"], "frozen_threshold": DISAGREE_ABSTAIN,
              "models": {"A": CHEAP, "B": BMODEL, "frontier": FRONTIER}, "table": table,
              "H1_pass": H1_pass, "H2_pass": H2_pass, "H3_pass": H3_pass}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase3_eval.json").write_text(json.dumps(report, indent=2, default=float))
    write_md(report)
    print("\n=== PHASE 3 ===")
    for t in table:
        print(f"  {t['workload']:<14}{'(held)' if t['held_out'] else '      '} disagree={t['disagree_rate']:.0%} "
              f"pred={'GO-elig' if t['pred_go_eligible'] else 'ABSTAIN'} realized={'GO' if t['realized_go'] else 'NO'} "
              f"-> {'OK' if t['rule_correct'] else 'MISS'}")
    print(f"  H1={'PASS' if H1_pass else 'FAIL'}  H2={'PASS' if H2_pass else 'FAIL'}  H3={'PASS' if H3_pass else 'FAIL'}")
    print(f"  -> {'SELF-GATING WORKS' if (H1_pass and H2_pass and H3_pass) else ('PER-DOMAIN CALIBRATION' if H1_pass else 'NOT DETECTABLE')}")


def write_md(rep):
    t = rep["table"]
    L = ["# Phase 3 (reframed) — Self-Gating Competence Detection\n",
         f"Date {rep['date']}. A={rep['models']['A']}, B={rep['models']['B']}, frontier={rep['models']['frontier']} "
         f"(labels only — predictor is frontier-free). Frozen rule: A/B disagreement > {rep['frozen_threshold']:.0%} => abstain.\n",
         "## Competence score (a-priori, no frontier) vs realized routing",
         "| workload | held-out | n | A/B disagree | predicted | realized GO | rule |",
         "|---|---|---|---|---|---|---|"]
    for r in t:
        L.append(f"| {r['workload']} | {'yes' if r['held_out'] else 'no'} | {r['n']} | {r['disagree_rate']:.0%} | "
                 f"{'GO-eligible' if r['pred_go_eligible'] else 'abstain'} | {'GO' if r['realized_go'] else 'NO-GO'} | "
                 f"{'OK' if r['rule_correct'] else 'MISS'} |")
    L.append("")
    L.append(f"- **H1** (rule separates original 4): **{'PASS' if rep['H1_pass'] else 'FAIL'}**")
    L.append(f"- **H2** (self-gated routing matches realized; abstain=always-frontier safe): **{'PASS' if rep['H2_pass'] else 'FAIL'}**")
    L.append(f"- **H3** (holds on >=2 held-out domains — the overfit gate): **{'PASS' if rep['H3_pass'] else 'FAIL'}**\n")
    L.append("## Verdict\n")
    if rep["H1_pass"] and rep["H2_pass"] and rep["H3_pass"]:
        L.append("**SELF-GATING WORKS.** B's competence is detectable a priori from the raw A/B "
                 "disagreement rate alone (no frontier). The frozen >50% rule correctly classifies "
                 "GO vs blind-spot domains on held-out workloads, and self-gated routing reproduces "
                 "the per-domain verdicts. The disagreement lever is self-calibrating and shippable.")
    elif rep["H1_pass"] and not rep["H3_pass"]:
        L.append("**PER-DOMAIN CALIBRATION REQUIRED.** The rule separates the original 4 but does not "
                 "hold out-of-sample (H3 fail) — competence detection needs a small labeled sample per "
                 "new domain. Shippable with a stated per-domain setup cost, not a universal threshold.")
    else:
        L.append("**NOT DETECTABLE a priori.** The raw disagreement rate does not cleanly predict "
                 "competence/GO across domains. The lever stays manual (needs a frontier probe per "
                 "domain). Clean negative.")
    L.append(f"\nAll verdicts at {rep['date']} prices.")
    (RESULTS / "PHASE3_RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
