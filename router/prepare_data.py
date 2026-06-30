"""Load the MATH dataset (Hendrycks et al.) and build a fixed, stratified
200-problem subset with a logged seed.

The gold answer for MATH lives inside the `solution` field as the last
\\boxed{...}. We extract it and keep per-problem `level` (1-5) and `type`.

Output: data/sample.jsonl  (one JSON object per line: id, problem, answer,
level, type, subject) and data/sample_meta.json (the run config).
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from checker import extract_boxed

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

with open(ROOT / "config.json") as f:
    CONFIG = json.load(f)

SEED = CONFIG["seed"]
SAMPLE_SIZE = CONFIG["sample_size"]
DATASET = CONFIG["dataset"]

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _level_int(level_field: str) -> int | None:
    m = re.search(r"(\d+)", str(level_field))
    return int(m.group(1)) if m else None


def load_all() -> list[dict]:
    """Load the MATH test split across all subjects into a flat list."""
    rows = []
    for subj in SUBJECTS:
        try:
            ds = load_dataset(DATASET, subj, split="test")
        except Exception as e:
            print(f"  warn: could not load subject {subj}: {e}")
            continue
        for i, ex in enumerate(ds):
            gold = extract_boxed(ex.get("solution", "")) if ex.get("solution") else None
            lvl = _level_int(ex.get("level", ""))
            if gold is None or lvl is None:
                continue
            rows.append(
                {
                    "id": f"{subj}_{i}",
                    "problem": ex["problem"],
                    "answer": gold.strip(),
                    "level": lvl,
                    "type": ex.get("type", subj),
                    "subject": subj,
                }
            )
        print(f"  {subj}: loaded {sum(1 for r in rows if r['subject'] == subj)} usable")
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n problems stratified evenly across difficulty levels 1-5."""
    by_level: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)

    levels = sorted(by_level)
    per_level = n // len(levels)
    rng = random.Random(seed)
    sample: list[dict] = []
    for lvl in levels:
        pool = by_level[lvl]
        rng.shuffle(pool)
        take = pool[:per_level]
        sample.extend(take)

    # top up to exactly n from the remaining pool if rounding left a deficit
    if len(sample) < n:
        chosen_ids = {r["id"] for r in sample}
        remainder = [r for r in rows if r["id"] not in chosen_ids]
        rng.shuffle(remainder)
        sample.extend(remainder[: n - len(sample)])

    rng.shuffle(sample)
    return sample


def main():
    print(f"loading {DATASET} ...")
    rows = load_all()
    print(f"total usable problems: {len(rows)}")

    sample = stratified_sample(rows, SAMPLE_SIZE, SEED)

    out = DATA / "sample.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = defaultdict(int)
    for r in sample:
        dist[r["level"]] += 1

    meta = {
        "dataset": DATASET,
        "seed": SEED,
        "sample_size": len(sample),
        "level_distribution": dict(sorted(dist.items())),
        "subjects": SUBJECTS,
    }
    with open(DATA / "sample_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {len(sample)} problems to {out}")
    print(f"level distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
