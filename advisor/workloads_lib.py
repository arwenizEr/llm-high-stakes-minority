"""Phase 1 workload library: 7 fresh workloads spanning the C spectrum, run with
a FIXED cost model (k=1 cheap haiku + plain sonnet frontier). Plus banking77 and
MATH reused from cache as anchors.

Each loader returns a WorkloadSpec; run_workload produces normalized records
(cheap k=1, frontier 1) for the diagnostics engine.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from harness import CONFIG, run_model, cost_of, _cache_path
from checkers import get_checker

ROOT = Path(__file__).resolve().parent
SEED = CONFIG["seed"]
CHEAP = CONFIG["cheap_model"]
FRONTIER = CONFIG["frontier_model"]
N = CONFIG.get("phase1_n", 50)


@dataclass
class WorkloadSpec:
    name: str
    items: list  # {id, prompt, gold}
    system: str
    checker_kind: str
    max_tokens: int = 32


def _cls_prompt(text, labels):
    return f"Labels: {', '.join(labels)}\nText: {text}\nLabel:"


def _sample(seq, n):
    idx = list(range(len(seq)))
    random.Random(SEED).shuffle(idx)
    return idx[:n]


def load_ag_news():
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split="test")
    names = ds.features["label"].names
    items = [{"id": f"agnews_{i}", "prompt": _cls_prompt(ds[i]["text"], names),
              "gold": names[ds[i]["label"]]} for i in _sample(ds, N)]
    return WorkloadSpec("ag_news", items, "Classify the news topic. Reply with only the label.", "classification", 12)


def load_emotion():
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion", split="test")
    names = ds.features["label"].names
    items = [{"id": f"emotion_{i}", "prompt": _cls_prompt(ds[i]["text"], names),
              "gold": names[ds[i]["label"]]} for i in _sample(ds, N)]
    return WorkloadSpec("emotion", items, "Classify the emotion. Reply with only the label.", "classification", 12)


def load_sst2():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    names = ds.features["label"].names
    items = [{"id": f"sst2_{i}", "prompt": _cls_prompt(ds[i]["sentence"], names),
              "gold": names[ds[i]["label"]]} for i in _sample(ds, N)]
    return WorkloadSpec("sst2", items, "Classify sentiment. Reply with only the label.", "classification", 12)


def load_trec():
    from datasets import load_dataset
    ds = load_dataset("CogComp/trec", split="test")
    col = "coarse_label" if "coarse_label" in ds.features else "label-coarse"
    names = ds.features[col].names
    items = [{"id": f"trec_{i}", "prompt": _cls_prompt(ds[i]["text"], names),
              "gold": names[ds[i][col]]} for i in _sample(ds, N)]
    return WorkloadSpec("trec", items, "Classify the question type. Reply with only the label.", "classification", 12)


def load_squad(n=N):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    items = [{"id": f"squad_{i}",
              "prompt": f"Context: {ds[i]['context']}\nQuestion: {ds[i]['question']}\nShort answer:",
              "gold": ds[i]["answers"]["text"][0]} for i in _sample(ds, n)]
    return WorkloadSpec("squad", items, "Answer with the shortest exact span from the context.", "classification", 24)


def load_gsm8k(n=N):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    def gold(a): return a.split("####")[-1].strip()
    items = [{"id": f"gsm8k_{i}", "prompt": f"Question: {ds[i]['question']}\nGive the final numeric answer.",
              "gold": gold(ds[i]["answer"])} for i in _sample(ds, n)]
    return WorkloadSpec("gsm8k", items, "Solve the math word problem. End with the final number.", "numeric", 512)


def _load_arc(cfg, name, n):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", cfg, split="test")
    items = []
    for i in _sample(ds, n):
        ex = ds[i]
        ch = ex["choices"]
        opts = "\n".join(f"{l}) {t}" for l, t in zip(ch["label"], ch["text"]))
        items.append({"id": f"{name}_{i}", "prompt": f"Question: {ex['question']}\n{opts}\nAnswer with the letter.",
                      "gold": ex["answerKey"]})
    return WorkloadSpec(name, items, "Answer the multiple-choice question with the correct letter only.", "classification", 12)


def load_arc():
    return _load_arc("ARC-Easy", "arc_easy", N)


def load_arc_challenge():
    return _load_arc("ARC-Challenge", "arc_challenge", CONFIG.get("phase25_n", 150))


LOADERS = {
    "ag_news": load_ag_news, "emotion": load_emotion, "sst2": load_sst2,
    "trec": load_trec, "squad": load_squad, "gsm8k": load_gsm8k, "arc_easy": load_arc,
}


def estimate_new_calls(spec: WorkloadSpec):
    """count uncached cheap(k=1)+frontier calls for the gate."""
    new = 0
    for it in spec.items:
        if not _cache_path(spec.name, CHEAP, spec.system, it["prompt"], 0).exists():
            new += 1
        if not _cache_path(spec.name, FRONTIER, spec.system, it["prompt"], 0).exists():
            new += 1
    return new


def run_workload(spec: WorkloadSpec):
    """Records with FIXED cost model: cheap k=1 + frontier 1."""
    check = get_checker(spec.checker_kind)
    rng = random.Random(SEED + 1)
    test_ids = set(rng.sample([it["id"] for it in spec.items], int(len(spec.items) * 0.4)))

    def cheap(it):
        return run_model(spec.name, it, CHEAP, system=spec.system, check_fn=check,
                         k_index=0, temperature=CONFIG["temperature"], max_tokens=spec.max_tokens)

    def front(it):
        return run_model(spec.name, it, FRONTIER, system=spec.system, check_fn=check,
                         k_index=0, max_tokens=spec.max_tokens)

    recs = []
    with ThreadPoolExecutor(max_workers=CONFIG["concurrency"]) as ex:
        cf = {ex.submit(cheap, it): it["id"] for it in spec.items}
        ff = {ex.submit(front, it): it["id"] for it in spec.items}
        cr = {cf[f]: f.result() for f in as_completed(cf)}
        fr = {ff[f]: f.result() for f in as_completed(ff)}
    for it in spec.items:
        c, f = cr[it["id"]], fr[it["id"]]
        recs.append({
            "id": it["id"], "in_test": it["id"] in test_ids,
            "cheap_correct": bool(c["correct"]), "frontier_correct": bool(f["correct"]),
            "cheap_cost": c["cost"], "frontier_cost": f["cost"],
            "signal": 1.0,  # k=1: no self-consistency signal in Phase 1 (oracle escalation used)
        })
    return recs
