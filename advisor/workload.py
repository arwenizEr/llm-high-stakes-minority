"""Workload ingestion -> normalized per-item records:
  {id, in_test, cheap_correct, frontier_correct, cheap_cost, frontier_cost, signal}
where signal = self-consistency agreement (lower = less confident).

BANKING77 (flat-cost classification) runs fresh via the harness (cost-gated).
MATH (concentrated-tail) is ingested from the existing router/gating caches —
zero new spend.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness import CONFIG, run_model, cost_of, _cache_path
from checkers import get_checker, classification_check

ROOT = Path(__file__).resolve().parent
ROUTER = ROOT.parent / "router"
K = CONFIG["k"]
SEED = CONFIG["seed"]
CHEAP = CONFIG["cheap_model"]
FRONTIER = CONFIG["frontier_model"]


# ---------------- BANKING77 ----------------

BANK_SYSTEM = "You are an intent classifier for banking customer queries. Reply with ONLY the single most appropriate intent label from the provided list, nothing else."


def _bank_prompt(text, labels):
    return f"Intent labels:\n{', '.join(labels)}\n\nQuery: {text}\nIntent label:"


def load_banking77_items():
    from datasets import load_dataset
    ds = load_dataset("mteb/banking77", split="test")  # parquet mirror, has label_text
    names = sorted(set(ds["label_text"]))
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    items = []
    for i in idx[: CONFIG["banking77_n"]]:
        ex = ds[i]
        items.append({
            "id": f"bank_{i}",
            "prompt": _bank_prompt(ex["text"], names),
            "gold": ex["label_text"],
        })
    return items


def _bank_consensus(answers):
    norm = [classification_check, ]  # not used; cluster by exact-normalized label
    from checkers import _norm
    labs = [_norm(a.replace("_", " ")) for a in answers]
    c = Counter(labs)
    top, cnt = c.most_common(1)[0]
    return top, cnt / len(answers)


def run_banking77():
    items = load_banking77_items()
    check = classification_check
    # cost gate
    n = len(items)
    new_cheap = sum(not _cache_path("banking77", CHEAP, BANK_SYSTEM, it["prompt"], k).exists()
                    for it in items for k in range(K))
    new_front = sum(not _cache_path("banking77", FRONTIER, BANK_SYSTEM, it["prompt"], 0).exists()
                    for it in items)
    est = new_cheap * cost_of(CHEAP, 300, 8) + new_front * cost_of(FRONTIER, 300, 8)
    ceil = CONFIG["cost_ceiling_usd"]
    print(f"banking77: {n} items; new cheap={new_cheap} front={new_front}; est ${est:.3f} (ceil ${ceil})")
    if est > ceil:
        raise SystemExit(f"ABORT banking77 est ${est:.3f} > ceil ${ceil}")

    def cheap_samples(it):
        return [run_model("banking77", it, CHEAP, system=BANK_SYSTEM, check_fn=check,
                          k_index=k, temperature=CONFIG["temperature"], max_tokens=32) for k in range(K)]

    def front_run(it):
        return run_model("banking77", it, FRONTIER, system=BANK_SYSTEM, check_fn=check,
                         k_index=0, max_tokens=32)

    records = []
    with ThreadPoolExecutor(max_workers=CONFIG["concurrency"]) as ex:
        cf = {ex.submit(cheap_samples, it): it for it in items}
        ff = {ex.submit(front_run, it): it for it in items}
        cheap_res = {cf[f]["id"]: f.result() for f in as_completed(cf)}
        front_res = {ff[f]["id"]: f.result() for f in as_completed(ff)}

    rng = random.Random(SEED)
    test_ids = set(rng.sample([it["id"] for it in items], int(n * 0.4)))
    for it in items:
        cs = cheap_res[it["id"]]
        maj, agree = _bank_consensus([s["answer"] for s in cs])
        cheap_correct = check(maj, it["gold"])
        fr = front_res[it["id"]]
        records.append({
            "id": it["id"], "in_test": it["id"] in test_ids,
            "cheap_correct": bool(cheap_correct), "frontier_correct": bool(fr["correct"]),
            "cheap_cost": sum(s["cost"] for s in cs), "frontier_cost": fr["cost"],
            "signal": agree,
        })
    return records


# ---------------- MATH (from cache) ----------------

def load_math_records():
    labels = [json.loads(l) for l in open(ROUTER / "results" / "routing_labels.jsonl", encoding="utf-8")]
    # frontier (sonnet high) tokens from baseline_runs
    ftok = {}
    for line in open(ROUTER / "results" / "baseline_runs.jsonl"):
        r = json.loads(line)
        if r["budget"] == "high":
            ftok[r["id"]] = cost_of("claude-sonnet-4-6", r["input_tokens"], r["output_tokens"])
    # cheap (haiku k=3) cost from gating caches
    import glob
    chcost = {}
    for f in glob.glob(str(ROUTER / "cache" / "*claude-haiku-4-5__sc*")):
        r = json.loads(open(f).read())
        chcost.setdefault(r["id"], 0.0)
        chcost[r["id"]] += cost_of("claude-haiku-4-5", r["input_tokens"], r["output_tokens"])
    records = []
    for l in labels:
        records.append({
            "id": l["id"], "in_test": l["in_test"],
            "cheap_correct": l["cheap_correct"], "frontier_correct": l["frontier_correct"],
            "cheap_cost": chcost.get(l["id"], 0.0), "frontier_cost": ftok.get(l["id"], 0.0),
            "signal": l["agreement"],
        })
    return records


def build_records(workload):
    if workload == "banking77":
        return run_banking77()
    if workload == "math":
        return load_math_records()
    raise ValueError(workload)
