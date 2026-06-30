"""Agent 3: downstream perturbation labeling (the cost-sensitive step).

For each span (sentence) in a question's context: remove it, re-ask, and check
whether the answer survived. Label droppable (survived) or load-bearing (flipped
a previously-correct answer).

HARD COST GATE: before any calls, estimate n_spans x per-call cost and refuse if
over `label_ceiling_usd`. If over, coarsen sentences -> paragraphs, then reduce
question count, until it fits.

Reuses cached full-context answers from the baseline sweep (none@0.0) for free.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness import CONFIG, MODEL, run_qa, full_context
from checker import is_correct, normalize

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PRICE = CONFIG["pricing_per_mtok"][MODEL]
PER_CALL = 1583 / 1e6 * PRICE["input"] + 8 / 1e6 * PRICE["output"]  # ~full-context call
LABEL_CEIL = CONFIG["label_ceiling_usd"]
N_TRAIN = CONFIG["label_train_n"]


def load_sample():
    with open(ROOT / "data" / "sample.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_split(sample):
    import random
    ids = [p["id"] for p in sample]
    random.Random(CONFIG["seed"]).shuffle(ids)
    train_ids = ids[:N_TRAIN]
    test_ids = ids[N_TRAIN:N_TRAIN + 60]
    split = {"seed": CONFIG["seed"], "label_train_ids": train_ids, "test_ids": test_ids}
    (RESULTS / "compressor_split.json").write_text(json.dumps(split, indent=2))
    return split


def q_overlap(span_text: str, question: str) -> float:
    s = set(normalize(span_text).split())
    q = set(normalize(question).split())
    if not s or not q:
        return 0.0
    return len(s & q) / len(q)


def enumerate_spans(problem, granularity: str):
    """Yield (key, span_text, feature_dict). key identifies the span for removal."""
    paras = problem["paragraphs"]
    if granularity == "paragraph":
        total = len(paras)
        for pi, para in enumerate(paras):
            text = para["title"] + ". " + " ".join(para["sentences"])
            yield ((pi, None), text, {"para_idx": pi, "sent_idx": -1,
                    "position": pi / max(1, total - 1) if total > 1 else 0.0,
                    "n_words": len(text.split()),
                    "q_overlap": q_overlap(text, problem["question"])})
    else:
        flat = [(pi, si) for pi, para in enumerate(paras) for si in range(len(para["sentences"]))]
        total = len(flat)
        for k, (pi, si) in enumerate(flat):
            text = paras[pi]["sentences"][si]
            yield ((pi, si), text, {"para_idx": pi, "sent_idx": si,
                    "position": k / max(1, total - 1) if total > 1 else 0.0,
                    "n_words": len(text.split()),
                    "q_overlap": q_overlap(text, problem["question"])})


def context_without(problem, drop_key):
    """Rebuild context_text with one span removed."""
    pi_drop, si_drop = drop_key
    parts = []
    for pi, para in enumerate(problem["paragraphs"]):
        if pi == pi_drop and si_drop is None:
            continue  # drop whole paragraph
        sents = [s for si, s in enumerate(para["sentences"]) if not (pi == pi_drop and si == si_drop)]
        parts.append(para["title"] + ". " + " ".join(sents))
    return "\n\n".join(parts)


def choose_granularity(full_correct_problems):
    """Return (granularity, problems_subset) that fits the cost gate."""
    for gran in ("sentence", "paragraph"):
        problems = full_correct_problems
        n_spans = sum(sum(1 for _ in enumerate_spans(p, gran)) for p in problems)
        est = n_spans * PER_CALL
        print(f"  gate[{gran}]: {len(problems)} q, {n_spans} spans, est ${est:.2f} (ceiling ${LABEL_CEIL})")
        if est <= LABEL_CEIL:
            return gran, problems
    # paragraph still over -> shrink question count to fit at paragraph granularity
    gran = "paragraph"
    problems = list(full_correct_problems)
    while problems:
        n_spans = sum(sum(1 for _ in enumerate_spans(p, gran)) for p in problems)
        if n_spans * PER_CALL <= LABEL_CEIL:
            break
        problems = problems[:-1]
    print(f"  gate: shrank to {len(problems)} q @ paragraph granularity")
    return gran, problems


def label_one(problem, gran):
    gold = problem["answer"]
    out = []
    for key, text, feats in enumerate_spans(problem, gran):
        rec = run_qa(problem, 0.0, strategy="loo",
                     compress_fn=lambda p, r, k=key: context_without(p, k))
        removed_correct = bool(rec["correct"])
        out.append({
            "qid": problem["id"],
            "drop_key": list(key),
            "span_text": text,
            **feats,
            "removed_correct": removed_correct,
            "label": "droppable" if removed_correct else "load_bearing",
        })
    return out


def main():
    sample = load_sample()
    by_id = {p["id"]: p for p in sample}
    split = make_split(sample)
    train = [by_id[i] for i in split["label_train_ids"]]

    # full-context correctness (cached from baseline none@0.0)
    full_correct = []
    for p in train:
        rec = run_qa(p, 0.0)  # cached
        if rec["correct"]:
            full_correct.append(p)
    print(f"label-train: {len(train)} questions, {len(full_correct)} answerable at full context")

    print("HARD COST GATE:")
    gran, problems = choose_granularity(full_correct)
    n_spans = sum(sum(1 for _ in enumerate_spans(p, gran)) for p in problems)
    est = n_spans * PER_CALL
    print(f"  RUNNING: {gran} granularity, {len(problems)} q, {n_spans} spans, est ${est:.2f}")
    if est > LABEL_CEIL:
        raise SystemExit(f"ABORT: ${est:.2f} > ceiling ${LABEL_CEIL}")

    all_labels = []
    workers = CONFIG.get("concurrency", 4)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(label_one, p, gran): p["id"] for p in problems}
        for fut in as_completed(futs):
            all_labels.extend(fut.result())
            done += 1
            if done % 5 == 0 or done == len(problems):
                print(f"  labeled {done}/{len(problems)} questions")

    with open(RESULTS / "span_labels.jsonl", "w", encoding="utf-8") as f:
        for r in all_labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(all_labels)
    n_drop = sum(r["label"] == "droppable" for r in all_labels)
    n_load = n - n_drop
    meta = {
        "granularity": gran, "n_questions": len(problems), "n_spans": n,
        "droppable": n_drop, "load_bearing": n_load,
        "droppable_frac": n_drop / n if n else 0,
        "est_cost_usd": est, "per_call_usd": PER_CALL,
    }
    (RESULTS / "span_labels_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nlabeled {n} spans: droppable={n_drop} ({n_drop/n*100:.0f}%) "
          f"load_bearing={n_load} ({n_load/n*100:.0f}%)")
    print("class balance is the key check vs the 85% collapse from the difficulty experiment.")


if __name__ == "__main__":
    main()
