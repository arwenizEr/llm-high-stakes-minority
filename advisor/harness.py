"""Generalized, workload-agnostic harness. Runs any registered model on any
workload item, records answer/correctness/tokens/cost, caches by
(workload, model, input_hash, k_index). Self-consistency via k_index.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

with open(ROOT / "config.json") as f:
    CONFIG = json.load(f)

PRICING = CONFIG["pricing_per_mtok"]

_client: anthropic.Anthropic | None = None


def _load_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for p in (ROOT / ".env", ROOT.parent / "router" / ".env", ROOT.parent / "compressor" / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY"):
                    return line.partition("=")[2].strip().strip('"').strip("'")
    return None


def _client_():
    global _client
    if _client is None:
        k = _load_key()
        _client = anthropic.Anthropic(api_key=k) if k else anthropic.Anthropic()
    return _client


def cost_of(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING[model]
    return in_tok / 1e6 * p["input"] + out_tok / 1e6 * p["output"]


def _cache_path(workload, model, system, prompt, k_index):
    h = hashlib.sha256()
    h.update(model.encode()); h.update(system.encode()); h.update(prompt.encode())
    ph = h.hexdigest()[:12]
    return CACHE / f"{workload}__{model}__{ph}__k{k_index}.json"


def run_model(workload, item, model, *, system, check_fn, k_index=0,
              temperature=None, max_tokens=512, force=False):
    prompt = item["prompt"]
    cpath = _cache_path(workload, model, system, prompt, k_index)
    if cpath.exists() and not force:
        r = json.loads(cpath.read_text()); r["cached"] = True; return r
    kw = {}
    if temperature is not None:
        kw["temperature"] = temperature
    resp = _client_().messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}], **kw)
    ans = "".join(b.text for b in resp.content if b.type == "text").strip()
    intok, outtok = resp.usage.input_tokens, resp.usage.output_tokens
    rec = {
        "workload": workload, "id": item["id"], "model": model, "k_index": k_index,
        "answer": ans, "gold": item["gold"], "correct": bool(check_fn(ans, item["gold"])),
        "input_tokens": intok, "output_tokens": outtok, "cost": cost_of(model, intok, outtok),
    }
    tmp = cpath.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, cpath)
    rec["cached"] = False
    return rec
