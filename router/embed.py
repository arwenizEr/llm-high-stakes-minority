"""Embed the 200 problem statements with a cheap local embedding model and
cache to disk. No API cost (runs locally), so the cost-guard is trivial but
logged for parity with the rest of the pipeline.

Model: sentence-transformers all-MiniLM-L6-v2 (384-dim). Cached to
data/embeddings_minilm.npz keyed by problem id.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_PATH = DATA / "embeddings_minilm.npz"


def load_sample() -> list[dict]:
    with open(DATA / "sample.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_embeddings(force: bool = False) -> tuple[list[str], np.ndarray]:
    sample = load_sample()
    ids = [p["id"] for p in sample]

    if EMB_PATH.exists() and not force:
        d = np.load(EMB_PATH, allow_pickle=True)
        cached_ids = list(d["ids"])
        if cached_ids == ids:
            print(f"embeddings cache hit: {EMB_PATH} ({d['X'].shape})")
            return cached_ids, d["X"]
        print("cache id-mismatch; rebuilding")

    # cost guard (local embedding: $0)
    n = len(sample)
    est_usd = 0.0
    print(f"embedding {n} problems with {EMB_MODEL} (local, est cost ${est_usd:.2f})")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)
    texts = [p["problem"] for p in sample]
    X = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    X = np.asarray(X, dtype=np.float32)

    DATA.mkdir(exist_ok=True)
    np.savez(EMB_PATH, ids=np.array(ids, dtype=object), X=X)
    print(f"wrote {EMB_PATH} {X.shape}")

    # log config sidecar
    with open(DATA / "embeddings_meta.json", "w") as f:
        json.dump(
            {"model": EMB_MODEL, "dim": int(X.shape[1]), "n": n, "normalized": True, "cost_usd": est_usd},
            f, indent=2,
        )
    return ids, X


def load_embeddings() -> dict[str, np.ndarray]:
    ids, X = build_embeddings()
    return {i: X[k] for k, i in enumerate(ids)}


if __name__ == "__main__":
    build_embeddings()
