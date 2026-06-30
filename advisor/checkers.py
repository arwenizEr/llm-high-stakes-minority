"""Pluggable quality checkers per workload. classification: exact-match on the
intent label (tolerant of the model echoing extra words). math/qa reuse the
research checkers when needed (math is consumed pre-checked from cache, so its
import is lazy)."""

from __future__ import annotations

import re
import string

_PUNCT = str.maketrans("", "", string.punctuation)


def _norm(s: str) -> str:
    return " ".join(str(s).lower().translate(_PUNCT).split())


def classification_check(predicted: str, gold: str) -> bool:
    """True if the gold intent label appears as a token-run in the prediction.
    Label tokens joined with spaces or underscores both accepted."""
    p = _norm(predicted.replace("_", " "))
    g = _norm(str(gold).replace("_", " "))
    if not g:
        return False
    if p == g:
        return True
    pt, gt = p.split(), g.split()
    for i in range(len(pt) - len(gt) + 1):
        if pt[i:i + len(gt)] == gt:
            return True
    return False


def numeric_check(predicted: str, gold: str) -> bool:
    """Compare the last number in the prediction to the gold number."""
    def last_num(s):
        nums = re.findall(r"-?\d[\d,]*\.?\d*", str(s).replace(",", ""))
        return float(nums[-1]) if nums else None
    p, g = last_num(predicted), last_num(gold)
    if p is None or g is None:
        return False
    return abs(p - g) < 1e-6


def get_checker(kind: str):
    if kind == "classification":
        return classification_check
    if kind == "numeric":
        return numeric_check
    if kind == "math":
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "router"))
        from checker import is_correct
        return is_correct
    raise ValueError(kind)
