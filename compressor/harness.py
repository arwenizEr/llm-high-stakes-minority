"""QA compression harness: run_qa(problem, rate, ...) -> {answer, correct, f1,
input_tokens, output_tokens}.

The compression step is pluggable: pass a `compress_fn(problem, rate) -> str`
that returns the (possibly compressed) context text. At rate 0 the default
returns the full context. Baselines and the downstream-aware compressor share
this one code path.

Cache keyed by (problem_id, strategy, rate, prompt_hash). A second identical
call makes zero network requests. Token counts come from the API usage fields.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import anthropic

from checker import is_correct, f1

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

with open(ROOT / "config.json") as f:
    CONFIG = json.load(f)

MODEL = CONFIG["model"]
MAX_TOKENS = CONFIG["max_tokens"]

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. Reply with the shortest "
    "exact answer span (a few words, a name, a number, or yes/no). No explanation."
)

_client: anthropic.Anthropic | None = None


def _load_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # reuse the router project's gitignored key file
    for env_path in (ROOT / ".env", ROOT.parent / "router" / ".env"):
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY"):
                    return line.partition("=")[2].strip().strip('"').strip("'")
    return None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = _load_key()
        _client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    return _client


def full_context(problem: dict, rate: float = 0.0) -> str:
    return problem["context_text"]


def build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def _hash(text: str) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode()); h.update(b"\x00")
    h.update(SYSTEM_PROMPT.encode()); h.update(b"\x00")
    h.update(text.encode())
    return h.hexdigest()[:16]


def _cache_path(pid: str, strategy: str, rate: float, phash: str) -> Path:
    safe = str(pid).replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}__{strategy}__r{rate}__{phash}.json"


def run_qa(problem: dict, rate: float, *, strategy: str = "none",
           compress_fn=None, force: bool = False) -> dict:
    """Answer `problem` with context compressed at `rate` via `compress_fn`."""
    compress_fn = compress_fn or full_context
    context = compress_fn(problem, rate)
    prompt = build_prompt(context, problem["question"])
    phash = _hash(prompt)
    pid = problem["id"]
    cpath = _cache_path(pid, strategy, rate, phash)

    if cpath.exists() and not force:
        rec = json.loads(cpath.read_text())
        rec["cached"] = True
        return rec

    client = _get_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    gold = problem["answer"]
    rec = {
        "id": pid,
        "strategy": strategy,
        "rate": rate,
        "answer": answer,
        "gold": gold,
        "correct": is_correct(answer, gold),
        "f1": f1(answer, gold),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "model": MODEL,
    }
    tmp = cpath.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, cpath)
    rec["cached"] = False
    return rec


def _smoke():
    sample_path = ROOT / "data" / "sample.jsonl"
    if not sample_path.exists():
        print("no sample; run prepare_data.py first")
        return
    with open(sample_path, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f][:5]
    print(f"smoke: 5 questions @ rate 0 (full context), model={MODEL}\n")
    tin = tout = corr = 0
    for p in problems:
        rec = run_qa(p, 0.0)
        tin += rec["input_tokens"]; tout += rec["output_tokens"]; corr += rec["correct"]
        print(f"  id={rec['id'][:16]:<16} correct={rec['correct']!s:<5} f1={rec['f1']:.2f} "
              f"in={rec['input_tokens']:>5} out={rec['output_tokens']:>3} cached={rec['cached']} "
              f"ans={rec['answer'][:30]!r}")
    print(f"\n  total correct={corr}/5  input_tokens={tin}  mean_in={tin/5:.0f}  output_tokens={tout}")


if __name__ == "__main__":
    _smoke()
