"""QA answer checker: SQuAD-style normalized Exact Match + token F1.

Replaces the sympy math-equivalence checker from the router project. Normalizes
by lowercasing, stripping punctuation and articles (a/an/the), and collapsing
whitespace, then compares.

is_correct: normalized EM, OR the normalized gold appears as a token-aligned
substring of the normalized prediction (so a generative model that answers
"The answer is Paris." still counts against gold "Paris"). f1: token-overlap F1.
"""

from __future__ import annotations

import re
import string


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


def _tokens(s: str) -> list[str]:
    return normalize(s).split()


def f1(predicted: str, gold: str) -> float:
    pt = _tokens(predicted)
    gt = _tokens(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    # multiset overlap
    from collections import Counter
    common = Counter(pt) & Counter(gt)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pt)
    recall = n_same / len(gt)
    return 2 * precision * recall / (precision + recall)


def _contains_aligned(pred_norm: str, gold_norm: str) -> bool:
    """True if gold appears as a whole-token run inside pred."""
    if not gold_norm:
        return False
    p = pred_norm.split()
    g = gold_norm.split()
    if len(g) > len(p):
        return False
    for i in range(len(p) - len(g) + 1):
        if p[i : i + len(g)] == g:
            return True
    return False


def is_correct(predicted: str, gold: str) -> bool:
    pn = normalize(predicted)
    gn = normalize(gold)
    if pn == gn:
        return True
    # generative tolerance: gold span contained in the answer
    if gn and _contains_aligned(pn, gn):
        return True
    # yes/no questions
    if gn in ("yes", "no") and pn.split()[:1] == [gn]:
        return True
    return False


def _test():
    cases = [
        # (predicted, gold, expect_correct)
        ("Paris", "Paris", True),
        ("The answer is Paris.", "Paris", True),
        ("paris", "Paris", True),
        ("A red apple", "red apple", True),            # article stripped
        ("Barack Obama", "Obama", True),               # gold contained
        ("Obama", "Barack Obama", False),              # gold longer, not contained
        ("1,000", "1000", True),                       # comma stripped by normalization
        ("yes", "yes", True),
        ("No, it isn't", "no", True),
        ("New York City", "New York", True),           # gold contained
        ("London", "Paris", False),
    ]
    passed = 0
    for i, (p, g, exp) in enumerate(cases):
        got = is_correct(p, g)
        ok = got == exp
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] case {i}: is_correct({p!r},{g!r})={got} exp={exp}  f1={f1(p,g):.2f}")
    print(f"\n{passed}/{len(cases)} QA checker tests passed")
    return passed == len(cases)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _test() else 1)
