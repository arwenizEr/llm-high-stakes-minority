"""Core harness: run_problem(problem, budget) -> {answer, correct, tokens}.

Budget mechanism
----------------
The project spec assumed a `budget_tokens` thinking parameter. That parameter
is REMOVED on every current Claude model (Opus 4.8 / Sonnet 4.6 return HTTP 400
for `thinking={"type":"enabled","budget_tokens":N}`). The current reasoning-
budget knob is `output_config.effort` combined with adaptive thinking, so the
three budget levels map to:

    low    -> output_config={"effort": "low"}   + thinking adaptive
    medium -> output_config={"effort": "medium"} + thinking adaptive
    high   -> output_config={"effort": "high"}  + thinking adaptive

`effort` controls thinking depth AND overall token spend, which is exactly the
"reasoning budget" this experiment varies. Reasoning tokens are billed as
output, so `usage.output_tokens` already includes them — that is the x-axis
metric, taken from the API usage field (never estimated from string length).

Caching
-------
Every call is cached to disk keyed by (problem_id, budget, prompt_hash). A cache
hit makes zero API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import anthropic

from checker import is_correct

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

with open(ROOT / "config.json") as f:
    CONFIG = json.load(f)

MODEL = CONFIG["model"]
MAX_TOKENS = CONFIG["max_tokens"]
BUDGETS = CONFIG["budgets"]

SYSTEM_PROMPT = (
    "You are a careful mathematician solving a competition problem. "
    "Show your reasoning, then give the final answer on its own line as "
    r"\boxed{ANSWER}."
)

_client: anthropic.Anthropic | None = None


def _load_key() -> str | None:
    """Resolve the API key: env var first, then a gitignored router/.env file
    (line: ANTHROPIC_API_KEY=sk-ant-...). Keeps the key off the command line
    and out of chat."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY"):
                _, _, val = line.partition("=")
                return val.strip().strip('"').strip("'")
    return None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = _load_key()
        _client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    return _client


def _prompt_hash(problem_text: str, budget: str) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode())
    h.update(b"\x00")
    h.update(SYSTEM_PROMPT.encode())
    h.update(b"\x00")
    h.update(json.dumps(BUDGETS[budget], sort_keys=True).encode())
    h.update(b"\x00")
    h.update(problem_text.encode())
    return h.hexdigest()[:16]


def _cache_path(problem_id: str, budget: str, phash: str, sample_idx: int = 0) -> Path:
    safe_id = str(problem_id).replace("/", "_").replace("\\", "_")
    suffix = "" if sample_idx == 0 else f"__s{sample_idx}"
    return CACHE_DIR / f"{safe_id}__{budget}__{phash}{suffix}.json"


def run_problem(problem: dict, budget: str, *, force: bool = False, sample_idx: int = 0) -> dict:
    """Solve one problem at the given budget.

    `problem` is a dict with keys: id, problem, answer (gold), level, type.
    Returns {answer, correct, input_tokens, output_tokens, cached}.
    """
    if budget not in BUDGETS:
        raise ValueError(f"unknown budget {budget!r}; expected one of {list(BUDGETS)}")

    problem_text = problem["problem"]
    gold = problem["answer"]
    pid = problem["id"]
    phash = _prompt_hash(problem_text, budget)
    cpath = _cache_path(pid, budget, phash, sample_idx)

    if cpath.exists() and not force:
        with open(cpath) as f:
            rec = json.load(f)
        rec["cached"] = True
        return rec

    client = _get_client()
    effort = BUDGETS[budget]["effort"]

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": problem_text}],
    )

    # Concatenate visible text blocks (thinking blocks carry no text by default).
    answer_text = "".join(b.text for b in resp.content if b.type == "text")

    rec = {
        "id": pid,
        "budget": budget,
        "level": problem.get("level"),
        "type": problem.get("type"),
        "answer": answer_text,
        "gold": gold,
        "correct": is_correct(answer_text, gold),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "stop_reason": resp.stop_reason,
        "model": MODEL,
        "effort": effort,
    }

    tmp = cpath.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, cpath)

    rec["cached"] = False
    return rec


def _smoke():
    """5-problem smoke run at one budget, printed with token counts."""
    sample_path = ROOT / "data" / "sample.jsonl"
    if not sample_path.exists():
        print(f"no sample at {sample_path}; run prepare_data.py first")
        return
    with open(sample_path) as f:
        problems = [json.loads(line) for line in f][:5]

    budget = "low"
    print(f"smoke run: {len(problems)} problems @ budget={budget}, model={MODEL}\n")
    tot_in = tot_out = correct = 0
    for p in problems:
        rec = run_problem(p, budget)
        tot_in += rec["input_tokens"]
        tot_out += rec["output_tokens"]
        correct += rec["correct"]
        print(
            f"  id={rec['id']:<14} lvl={rec['level']} correct={rec['correct']!s:<5} "
            f"in={rec['input_tokens']:>5} out={rec['output_tokens']:>6} "
            f"cached={rec['cached']}"
        )
    print(
        f"\n  total: correct={correct}/{len(problems)}  "
        f"input_tokens={tot_in}  output_tokens={tot_out}"
    )


if __name__ == "__main__":
    _smoke()
