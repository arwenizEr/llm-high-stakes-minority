"""Phase 2: can a CHEAP confidence signal beat self-consistency on minority
recall AND catch the confident-error blind spot? Signals: selfcons k3/k2 (cached),
verbalized confidence (new), self-verification (new), combined.
"""

from __future__ import annotations

import glob
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from harness import CONFIG, run_model, cost_of
from checkers import get_checker

ROOT = Path(__file__).resolve().parent
ROUTER = ROOT.parent / "router"
RESULTS = ROOT / "results"
CHEAP = "claude-haiku-4-5"
MATH_CHECK = get_checker("math")

VERB_SYS = ("Solve the math problem. Give the final answer as \\boxed{...}. Then on a new "
            "line write exactly 'Confidence: N' where N is an integer 0-100 = your probability "
            "the answer is correct.")
VERIFY_SYS = "You check a student's math answer. Reply with exactly one word: CORRECT or INCORRECT."


def load_math():
    return [json.loads(l) for l in open(ROUTER / "data" / "sample.jsonl", encoding="utf-8")]


def load_labels():
    return {json.loads(l)["id"]: json.loads(l)
            for l in open(ROUTER / "results" / "routing_labels.jsonl", encoding="utf-8")}


def load_gating_samples():
    """id -> {sidx: correct?} and answers, from cached gating haiku samples."""
    ans = {}
    for f in glob.glob(str(ROUTER / "cache" / "*claude-haiku-4-5__sc*")):
        r = json.loads(open(f).read())
        m = re.search(r"__sc(\d)__", f)
        if m:
            ans.setdefault(r["id"], {})[int(m.group(1))] = r["answer"]
    return ans


def _consensus_agreement(answers, is_corr):
    valid = [a for a in (extract_ans(x) for x in answers) if a]
    if not valid:
        return 0.0
    clusters = []
    for a in valid:
        for c in clusters:
            if is_corr(a, c[0]):
                c[1] += 1; break
        else:
            clusters.append([a, 1])
    return max(c[1] for c in clusters) / len(answers)


import sys
sys.path.insert(0, str(ROUTER))
from checker import is_correct as math_is_correct, extract_answer as extract_ans


def parse_conf(text):
    m = re.findall(r"[Cc]onfidence[:=]\s*(\d{1,3})", text)
    if m:
        return max(0, min(100, int(m[-1])))
    return 50  # no confidence stated -> neutral


def run_new_signals(math, labels):
    check = MATH_CHECK
    def verb(p):
        return run_model("math_verbalized", {"id": p["id"], "prompt": p["problem"], "gold": p["answer"]},
                         CHEAP, system=VERB_SYS, check_fn=check, k_index=0, temperature=0.0, max_tokens=1024)
    def verify(p):
        maj = labels[p["id"]].get("cheap_majority_answer") or ""
        prompt = f"Problem: {p['problem']}\nStudent's final answer: {maj}\nIs the student's answer correct?"
        return run_model("math_selfverify", {"id": p["id"], "prompt": prompt, "gold": "CORRECT"},
                         CHEAP, system=VERIFY_SYS, check_fn=lambda a, g: "correct" in a.lower() and "incorrect" not in a.lower(),
                         k_index=0, temperature=0.0, max_tokens=8)
    new = sum(1 for p in math) * 2
    est = new * cost_of(CHEAP, 200, 300)
    print(f"phase2 new calls ~{new}; est ${est:.2f} (ceil ${CONFIG['cost_ceiling_usd']})")
    if est > CONFIG["cost_ceiling_usd"]:
        raise SystemExit("ABORT phase2 cost gate")
    verb_res, verify_res = {}, {}
    with ThreadPoolExecutor(max_workers=CONFIG["concurrency"]) as ex:
        vf = {ex.submit(verb, p): p["id"] for p in math}
        sf = {ex.submit(verify, p): p["id"] for p in math}
        for f in as_completed(vf): r = f.result(); verb_res[vf[f]] = r
        for f in as_completed(sf): r = f.result(); verify_res[sf[f]] = r
    return verb_res, verify_res


def evaluate(scores, minority, ce_subset, flag_count=None, binary_flags=None):
    n = len(minority)
    if binary_flags is not None:
        flagged = binary_flags
    else:
        # flag the top `flag_count` by score (matched escalation budget)
        order = np.argsort(-scores)
        flagged = np.zeros(n, bool); flagged[order[:flag_count]] = True
    tp = int((flagged & minority).sum())
    rec = tp / minority.sum() if minority.sum() else 0
    prec = tp / flagged.sum() if flagged.sum() else 0
    ce_tp = int((flagged & ce_subset).sum())
    ce_rec = ce_tp / ce_subset.sum() if ce_subset.sum() else None
    return {"flag_count": int(flagged.sum()), "recall_minority": rec, "precision": prec,
            "recall_confident_error": ce_rec, "ce_caught": ce_tp, "ce_total": int(ce_subset.sum())}


def main():
    math = load_math()
    labels = load_labels()
    gsamp = load_gating_samples()
    ids = [p["id"] for p in math]

    minority = np.array([labels[i]["minority"] for i in ids])
    agree_k3 = np.array([labels[i]["agreement"] for i in ids])
    ce_subset = minority & (agree_k3 >= 1.0)  # confident-error: unanimous AND wrong-minority
    print(f"minority={minority.sum()}  confident-error subset (unanimous wrong)={ce_subset.sum()}")

    # k2 agreement from cached samples 0,1
    agree_k2 = np.array([_consensus_agreement([gsamp[i].get(0, ""), gsamp[i].get(1, "")], math_is_correct)
                         for i in ids])

    verb_res, verify_res = run_new_signals(math, labels)
    conf = np.array([parse_conf(verb_res[i]["answer"]) for i in ids])
    verify_incorrect = np.array(["incorrect" in verify_res[i]["answer"].lower() for i in ids])

    # k3 flag count = baseline escalation budget (agreement<1.0)
    k3_flagcount = int((agree_k3 < 1.0).sum())

    rows = {}
    rows["selfcons_k3"] = evaluate(1 - agree_k3, minority, ce_subset, flag_count=k3_flagcount)
    rows["selfcons_k2"] = evaluate(1 - agree_k2, minority, ce_subset, flag_count=k3_flagcount)
    rows["verbalized"] = evaluate(100 - conf, minority, ce_subset, flag_count=k3_flagcount)
    rows["selfverify"] = evaluate(None, minority, ce_subset, binary_flags=verify_incorrect)
    # combined: verbalized (matched-K) OR selfverify
    vb_order = np.argsort(-(100 - conf)); vb_flag = np.zeros(len(ids), bool); vb_flag[vb_order[:k3_flagcount]] = True
    rows["combined_verb_or_verify"] = evaluate(None, minority, ce_subset, binary_flags=(vb_flag | verify_incorrect))

    calls = {"selfcons_k3": 3, "selfcons_k2": 2, "verbalized": 1, "selfverify": 1, "combined_verb_or_verify": 2}
    for k in rows:
        rows[k]["cheap_calls"] = calls[k]

    (RESULTS / "phase2_signals.json").write_text(json.dumps(
        {"minority": int(minority.sum()), "confident_error_subset": int(ce_subset.sum()), "signals": rows},
        indent=2, default=float))

    print("\n=== PHASE 2: SIGNAL COMPARISON (MATH, all-200) ===")
    print(f"  {'signal':<24} {'calls':>5} {'flags':>5} {'recall':>7} {'prec':>6} {'CE-recall':>10}")
    for k, r in rows.items():
        cer = "n/a" if r["recall_confident_error"] is None else f"{r['recall_confident_error']:.2f} ({r['ce_caught']}/{r['ce_total']})"
        print(f"  {k:<24} {r['cheap_calls']:>5} {r['flag_count']:>5} {r['recall_minority']:>7.2f} {r['precision']:>6.2f} {cer:>10}")
    verdict(rows)


def verdict(rows):
    base = rows["selfcons_k3"]["recall_minority"]
    L = ["# Phase 2 — Cheap Signal vs Confident-Error Ceiling\n",
         f"Baseline selfcons_k3: recall {base:.2f}, confident-error recall "
         f"{rows['selfcons_k3']['recall_confident_error']} (blind by construction).\n",
         "| signal | cheap calls | minority recall | CE recall | beats bar? |", "|---|---|---|---|---|"]
    winners = []
    for k, r in rows.items():
        ce = r["recall_confident_error"]
        beats = (r["recall_minority"] >= base - 1e-9) and (ce is not None and ce > 0) and r["cheap_calls"] <= 2
        if beats and k != "selfcons_k3":
            winners.append(k)
        cer = "n/a" if ce is None else f"{ce:.2f}"
        L.append(f"| {k} | {r['cheap_calls']} | {r['recall_minority']:.2f} | {cer} | {'YES' if beats and k!='selfcons_k3' else '-'} |")
    L.append("")
    any_ce = any((rows[k]['recall_confident_error'] or 0) > 0 for k in rows if k != 'selfcons_k3')
    if winners:
        L.append(f"**WIN.** {', '.join(winners)} beat the k=3 bar (recall >= {base:.2f}) AND caught "
                 f">=1 confident-error item at <=2 cheap calls — a cheap signal that cracks the blind "
                 f"spot and preserves the price gap g. This is the signal the Phase-1 GO region needed.")
    elif any_ce:
        L.append("**PARTIAL.** A cheap signal caught confident-error items the k=3 baseline cannot, "
                 "but did not clear the full recall bar at <=2 calls. The blind spot is cheaply "
                 "*dentable*, not yet *closeable*.")
    else:
        L.append("**NEGATIVE.** No cheap Anthropic-only signal caught any confident-error item. The "
                 "confident-error ceiling stands; self-consistency's blind spot is not cheaply "
                 "closeable on this reasoning task. The GO region remains unreachable with cheap "
                 "signals here.")
    (RESULTS / "PHASE2_RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n  verdict: {'WIN '+str(winners) if winners else ('PARTIAL' if any_ce else 'NEGATIVE')}  -> PHASE2_RESULTS.md")


if __name__ == "__main__":
    main()
