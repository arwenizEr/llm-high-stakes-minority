"""Compression strategies: compress(problem, rate) -> context_text.

`rate` = fraction of tokens to DROP (0 = keep everything).

- none          : full context (rate ignored)
- random_drop   : drop a seeded-random `rate` fraction of whitespace tokens
- stopword_drop : drop stopwords/punctuation first, then seeded content tokens
- llmlingua2    : off-the-shelf LLMLingua-2 (intrinsic perplexity compressor) — THE BAR

random/stopword are deterministic given (problem_id, rate) so the harness cache
key (which hashes the compressed prompt) is stable. LLMLingua-2 output is cached
to disk (it is a model forward pass; do not recompute).
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LL_CACHE = ROOT / "data" / "llmlingua_cache"
LL_CACHE.mkdir(parents=True, exist_ok=True)

STOPWORDS = set(
    "a an the and or but if then else of to in on at by for with from as is are was "
    "were be been being this that these those it its he she they them his her their "
    "i you we my your our me us do does did have has had not no nor so than too very "
    "can will just about into over under again further once here there all any both "
    "each few more most other some such only own same s t up out off down".split()
)
_PUNCT = set(".,;:!?\"'()[]{}`-—–…")


def _is_stopword(tok: str) -> bool:
    core = tok.strip("".join(_PUNCT)).lower()
    if core == "" or all(c in _PUNCT for c in tok):
        return True
    return core in STOPWORDS


# ---------------------------------------------------------------------------

def none(problem: dict, rate: float = 0.0) -> str:
    return problem["context_text"]


def random_drop(problem: dict, rate: float) -> str:
    toks = problem["context_text"].split()
    n_drop = int(round(len(toks) * rate))
    if n_drop <= 0:
        return problem["context_text"]
    rng = random.Random(f"{problem['id']}|{rate}|rand")
    drop = set(rng.sample(range(len(toks)), min(n_drop, len(toks))))
    return " ".join(t for i, t in enumerate(toks) if i not in drop)


def stopword_drop(problem: dict, rate: float) -> str:
    toks = problem["context_text"].split()
    n_drop = int(round(len(toks) * rate))
    if n_drop <= 0:
        return problem["context_text"]
    sw_idx = [i for i, t in enumerate(toks) if _is_stopword(t)]
    rng = random.Random(f"{problem['id']}|{rate}|stop")
    if len(sw_idx) >= n_drop:
        drop = set(rng.sample(sw_idx, n_drop))
    else:
        drop = set(sw_idx)
        rest = [i for i in range(len(toks)) if i not in drop]
        extra = n_drop - len(drop)
        drop |= set(rng.sample(rest, min(extra, len(rest))))
    return " ".join(t for i, t in enumerate(toks) if i not in drop)


# ---- LLMLingua-2 (the bar) -------------------------------------------------

_ll_compressor = None
_ll_lock = threading.Lock()
LL_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def _get_ll():
    global _ll_compressor
    if _ll_compressor is None:
        from llmlingua import PromptCompressor
        _ll_compressor = PromptCompressor(model_name=LL_MODEL, use_llmlingua2=True, device_map="cpu")
    return _ll_compressor


def _ll_cache_path(pid: str, rate: float) -> Path:
    safe = str(pid).replace("/", "_").replace("\\", "_")
    return LL_CACHE / f"{safe}__r{rate}.json"


def llmlingua2(problem: dict, rate: float) -> str:
    cpath = _ll_cache_path(problem["id"], rate)
    if cpath.exists():
        return json.loads(cpath.read_text())["compressed"]
    keep = max(0.05, 1.0 - rate)  # llmlingua `rate` = fraction KEPT
    with _ll_lock:
        comp = _get_ll()
        out = comp.compress_prompt(problem["context_text"], rate=keep, force_tokens=["\n", ".", "?", ","])
    text = out["compressed_prompt"]
    cpath.write_text(json.dumps({"compressed": text, "keep_rate": keep,
                                 "origin_tokens": out.get("origin_tokens"),
                                 "compressed_tokens": out.get("compressed_tokens")}))
    return text


REGISTRY = {
    "none": none,
    "random_drop": random_drop,
    "stopword_drop": stopword_drop,
    "llmlingua2": llmlingua2,
}


def get_compressor(name: str):
    return REGISTRY[name]
