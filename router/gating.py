"""Model-Internal Signal gating experiment (Agent 1).

Reuses the router harness/checker/cache/sample/split. Tests whether a cheap
model's own confidence (self-consistency over k samples) identifies the
high-stakes minority: problems where the cheap model is WRONG but the cached
frontier (sonnet-high) is RIGHT.

LOGPROBS: the Anthropic Messages API does not expose token-level logprobs
(no `logprobs` request param, none in the response). Signal source is therefore
SELF-CONSISTENCY, not logprob entropy. Documented and decided here.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import harness as H
from checker import is_correct, extract_answer

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = H.CACHE_DIR

CHEAP_MODEL = "claude-haiku-4-5"
K = 3
TEMPERATURE = 0.8
MAX_TOKENS = 1024
PRICE = {"claude-haiku-4-5": {"input": 1.0, "output": 5.0}}
REMAINING_USD = 3.21
GATE_USD = 3.0


def _phash(prompt: str) -> str:
    h = hashlib.sha256()
    h.update(CHEAP_MODEL.encode()); h.update(H.SYSTEM_PROMPT.encode()); h.update(prompt.encode())
    return h.hexdigest()[:12]


def _cache_path(pid, sidx, phash):
    safe = str(pid).replace("/", "_").replace("\\", "_")
    return CACHE / f"{safe}__{CHEAP_MODEL}__sc{sidx}__{phash}.json"


def run_cheap(problem, sidx, *, force=False):
    prompt = problem["problem"]
    phash = _phash(prompt)
    cpath = _cache_path(problem["id"], sidx, phash)
    if cpath.exists() and not force:
        r = json.loads(cpath.read_text()); r["cached"] = True; return r
    client = H._get_client()
    resp = client.messages.create(
        model=CHEAP_MODEL, max_tokens=MAX_TOKENS, system=H.SYSTEM_PROMPT,
        temperature=TEMPERATURE, messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    rec = {
        "id": problem["id"], "sidx": sidx, "model": CHEAP_MODEL,
        "answer": text, "gold": problem["answer"],
        "correct": is_correct(text, problem["answer"]),
        "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
    }
    tmp = cpath.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, cpath)
    rec["cached"] = False
    return rec


def load_sample():
    with open(ROOT / "data" / "sample.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_frontier_correct():
    """sonnet-high correctness per problem from cached baseline runs."""
    d = {}
    with open(RESULTS / "baseline_runs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["budget"] == "high":
                d[r["id"]] = bool(r["correct"])
    return d


def load_test_ids():
    return set(json.loads((RESULTS / "split.json").read_text())["test_ids"])


def consensus(answers):
    """Self-consistency: majority math-equiv answer + agreement fraction."""
    exts = [extract_answer(a) for a in answers]
    valid = [a for a in exts if a]
    if not valid:
        return None, 0.0
    clusters = []  # [rep, count]
    for a in valid:
        for c in clusters:
            if is_correct(a, c[0]):
                c[1] += 1; break
        else:
            clusters.append([a, 1])
    best = max(clusters, key=lambda c: c[1])
    return best[0], best[1] / len(answers)


def calibrate(sample):
    print("calibration: 5 problems x1 cheap sample ...")
    outs = []
    for p in sample[:5]:
        r = run_cheap(p, 0)
        outs.append(r["output_tokens"])
    mo = sum(outs) / len(outs)
    per_call = 150 / 1e6 * PRICE[CHEAP_MODEL]["input"] + mo / 1e6 * PRICE[CHEAP_MODEL]["output"]
    print(f"  mean cheap output tokens ~{mo:.0f}, per-call ~${per_call:.4f}")
    return per_call


def main():
    sample = load_sample()
    frontier = load_frontier_correct()
    test_ids = load_test_ids()

    per_call = calibrate(sample)
    # already cached: 5 sidx0; need k*n total minus cached
    n = len(sample)
    total_calls = K * n
    cached = sum(_cache_path(p["id"], s, _phash(p["problem"])).exists() for p in sample for s in range(K))
    est = (total_calls - cached) * per_call
    print(f"plan: k={K} x n={n} = {total_calls} cheap calls; {cached} cached; "
          f"est new ${est:.2f} (gate ${GATE_USD}, remaining ${REMAINING_USD})")
    if est > GATE_USD:
        raise SystemExit(f"ABORT: est ${est:.2f} > gate ${GATE_USD}; reduce n or k")

    jobs = [(p, s) for p in sample for s in range(K)]
    recs = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=H.CONFIG.get("concurrency", 4)) as ex:
        futs = {ex.submit(run_cheap, p, s): (p["id"], s) for (p, s) in jobs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result(); recs[r["id"]][r["sidx"]] = r
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  cheap ... {done}/{len(jobs)}")

    labels = []
    cells = Counter()  # (cheap_correct, frontier_correct)
    for p in sample:
        samples = [recs[p["id"]][s] for s in range(K)]
        answers = [s["answer"] for s in samples]
        maj, agreement = consensus(answers)
        cheap_correct = is_correct(maj, p["answer"]) if maj else False
        front_correct = frontier.get(p["id"], False)
        minority = (not cheap_correct) and front_correct
        cells[(cheap_correct, front_correct)] += 1
        labels.append({
            "id": p["id"], "in_test": p["id"] in test_ids,
            "cheap_correct": cheap_correct, "frontier_correct": front_correct,
            "minority": minority, "agreement": agreement,
            "per_sample_correct": [s["correct"] for s in samples],
            "cheap_majority_answer": maj,
            "mean_cheap_output_tokens": sum(s["output_tokens"] for s in samples) / K,
        })

    with open(RESULTS / "routing_labels.jsonl", "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_min = sum(l["minority"] for l in labels)
    meta = {
        "cheap_model": CHEAP_MODEL, "frontier": "sonnet-4-6 effort=high (cached)",
        "k": K, "temperature": TEMPERATURE, "signal": "self_consistency (no logprobs on Anthropic API)",
        "n": n, "minority_n": n_min, "minority_frac": n_min / n,
        "cell_both_right": cells[(True, True)], "cell_both_wrong": cells[(False, False)],
        "cell_minority_cheapwrong_frontierright": cells[(False, True)],
        "cell_cheapright_frontierwrong": cells[(True, False)],
    }
    (RESULTS / "routing_labels_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== CHEAP vs FRONTIER (2x2) ===")
    print(f"                 frontier_right  frontier_wrong")
    print(f"  cheap_right     {cells[(True,True)]:>10}     {cells[(True,False)]:>10}")
    print(f"  cheap_wrong     {cells[(False,True)]:>10}     {cells[(False,False)]:>10}")
    print(f"\n  HIGH-STAKES MINORITY (cheap wrong, frontier right): "
          f"{n_min}/{n} = {n_min/n*100:.0f}%")
    print(f"  signal = self-consistency over k={K} (Anthropic API has NO logprobs)")


if __name__ == "__main__":
    main()
