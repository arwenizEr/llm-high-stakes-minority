"""Load HotpotQA (distractor) and build a fixed 200-question sample with logged
seed. Each question keeps its full structured context (10 paragraphs: title +
sentences) so the labeler can split at sentence/paragraph granularity.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

with open(ROOT / "config.json") as f:
    CONFIG = json.load(f)

SEED = CONFIG["seed"]
N = CONFIG["sample_size"]


def context_text(paragraphs: list[dict]) -> str:
    parts = []
    for p in paragraphs:
        parts.append(p["title"] + ". " + " ".join(p["sentences"]))
    return "\n\n".join(parts)


def main():
    print(f"loading {CONFIG['dataset']} [{CONFIG['dataset_config']}] {CONFIG['dataset_split']} ...")
    ds = load_dataset(CONFIG["dataset"], CONFIG["dataset_config"], split=CONFIG["dataset_split"])
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)

    rows = []
    for i in idx[:N]:
        ex = ds[i]
        ctx = ex["context"]
        paragraphs = [
            {"title": t, "sentences": list(s)}
            for t, s in zip(ctx["title"], ctx["sentences"])
        ]
        ctext = context_text(paragraphs)
        rows.append(
            {
                "id": ex.get("id", f"hotpot_{i}"),
                "question": ex["question"],
                "answer": ex["answer"],
                "type": ex.get("type"),
                "level": ex.get("level"),
                "paragraphs": paragraphs,
                "supporting_facts": ex.get("supporting_facts"),
                "context_text": ctext,
                "n_words": len(ctext.split()),
                "n_paragraphs": len(paragraphs),
            }
        )

    out = DATA / "sample.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import statistics
    wc = [r["n_words"] for r in rows]
    meta = {
        "dataset": CONFIG["dataset"], "config": CONFIG["dataset_config"],
        "split": CONFIG["dataset_split"], "seed": SEED, "n": len(rows),
        "mean_context_words": statistics.mean(wc),
        "median_context_words": statistics.median(wc),
        "mean_paragraphs": statistics.mean(r["n_paragraphs"] for r in rows),
    }
    with open(DATA / "sample_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {len(rows)} questions to {out}")
    print(f"mean context words: {meta['mean_context_words']:.0f}  "
          f"median: {meta['median_context_words']:.0f}  mean paragraphs: {meta['mean_paragraphs']:.1f}")


if __name__ == "__main__":
    main()
