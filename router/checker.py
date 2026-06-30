"""Math-equivalence answer checker for the MATH dataset.

Parses the final boxed/last answer from a model completion and checks
mathematical equivalence against the gold answer using sympy. Handles
fractions, radicals, simplification, and common LaTeX forms — never string
equality.
"""

from __future__ import annotations

import re
import signal
from contextlib import contextmanager

import sympy
from sympy.parsing.latex import parse_latex


# ----------------------------------------------------------------------------
# Answer extraction
# ----------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    r"""Return the contents of the LAST \boxed{...} in `text`, or None.

    Brace-matched so nested braces (e.g. \boxed{\frac{1}{2}}) are handled.
    """
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = idx + len(r"\boxed")
    # skip optional whitespace
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text):
        return None
    if text[i] == "{":
        # brace-matched extraction
        depth = 0
        start = i + 1
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
            i += 1
        return None
    # \boxed X  (rare, single token)
    return text[i:].split()[0] if text[i:].strip() else None


def extract_answer(text: str) -> str | None:
    r"""Pull the final answer string from a model completion.

    Order of preference: last \boxed{}, then "answer is X", then last
    non-empty line. Returns the raw answer substring (still LaTeX-y).
    """
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()

    m = re.findall(r"(?:final answer|answer)\s*(?:is|:|=)\s*\$?([^\n.$]+)", text, re.I)
    if m:
        return m[-1].strip()

    # fall back to last non-empty line
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip()
    return None


# ----------------------------------------------------------------------------
# Normalisation + equivalence
# ----------------------------------------------------------------------------

_REPLACEMENTS = [
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\!", ""),
    (r"\\,", ""),
    (r"\\;", ""),
    (r"\\ ", " "),
    (r"\\\\", ""),
    (r"\\%", ""),
    (r"\\$", ""),
    (r"\$", ""),
    (r"\\text\s*\{[^}]*\}", ""),
    (r"\\mathrm\s*\{([^}]*)\}", r"\1"),
    (r"\\mbox\s*\{[^}]*\}", ""),
    (r"\\!", ""),
    (r"^\\\$", ""),
]


def _clean(s: str) -> str:
    s = s.strip()
    # strip surrounding $...$
    s = s.strip("$").strip()
    # remove trailing punctuation / units commonly trailing answers
    s = s.rstrip(".")
    # degrees / dollar signs / percent
    s = s.replace(r"^\circ", "").replace(r"\circ", "")
    s = s.replace(",", "")  # thousands separators (sympy chokes on 1,000)
    for pat, rep in _REPLACEMENTS:
        s = re.sub(pat, rep, s)
    s = s.strip()
    # normalise \dfrac/\tfrac -> \frac
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    # brace-less single-token args: \sqrt2 -> \sqrt{2}, \frac12 -> \frac{1}{2}
    s = re.sub(r"\\sqrt\s*([0-9A-Za-z])", r"\\sqrt{\1}", s)
    s = re.sub(r"\\frac\s*([0-9A-Za-z])\s*([0-9A-Za-z])", r"\\frac{\1}{\2}", s)
    return s


@contextmanager
def _time_limit(seconds: float):
    """Best-effort timeout. signal.SIGALRM is POSIX-only; on Windows this is a
    no-op and we rely on sympy returning promptly."""
    if hasattr(signal, "SIGALRM"):
        def handler(signum, frame):
            raise TimeoutError
        old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    else:
        yield


def _to_expr(s: str):
    """Best-effort parse of a cleaned answer string into a sympy expression."""
    s = s.strip()
    if s == "":
        return None
    # try latex first if it looks latex-y (lark backend — pure-python, no antlr)
    candidates = []
    if "\\" in s or "^" in s or "frac" in s or "sqrt" in s:
        try:
            candidates.append(parse_latex(s, backend="lark"))
        except Exception:
            pass
    # plain sympify with ^ -> ** translation
    plain = s.replace("^", "**")
    try:
        candidates.append(sympy.sympify(plain))
    except Exception:
        pass
    try:
        candidates.append(sympy.sympify(s))
    except Exception:
        pass
    return candidates[0] if candidates else None


def is_correct(predicted: str | None, gold: str | None) -> bool:
    """Return True iff `predicted` is mathematically equivalent to `gold`.

    `predicted` may be a full model completion (the boxed answer is extracted)
    or a bare answer string. `gold` is the dataset's answer (bare or boxed).
    """
    if predicted is None or gold is None:
        return False

    pred_ans = extract_answer(predicted)
    gold_ans = extract_answer(gold) or gold
    if pred_ans is None:
        return False

    p = _clean(pred_ans)
    g = _clean(gold_ans)

    # fast path: exact string match after cleaning
    if p == g:
        return True

    # numeric / symbolic equivalence
    try:
        with _time_limit(5):
            pe = _to_expr(p)
            ge = _to_expr(g)
            if pe is None or ge is None:
                return False
            diff = sympy.simplify(pe - ge)
            if diff == 0:
                return True
            # numerical fallback for irrational forms sympy won't simplify
            try:
                return bool(abs(float(pe.evalf() - ge.evalf())) < 1e-9)
            except Exception:
                return False
    except (TimeoutError, Exception):
        return False


# ----------------------------------------------------------------------------
# Unit tests (run: python checker.py)
# ----------------------------------------------------------------------------

def _test():
    cases = [
        # (predicted, gold, expected)
        (r"\boxed{\frac{1}{2}}", "0.5", True),
        ("1/2", "0.5", True),
        (r"\frac{1}{2}", r"\frac{1}{2}", True),
        (r"2\sqrt{2}", r"\sqrt{8}", True),
        (r"\boxed{2\sqrt2}", r"\sqrt8", True),
        (r"The answer is \boxed{42}.", "42", True),
        (r"\boxed{x^2 + 2x + 1}", r"(x+1)^2", True),
        (r"\boxed{\frac{3}{6}}", r"\frac{1}{2}", True),
        (r"\boxed{0.25}", r"\frac{1}{4}", True),
        (r"\boxed{1}", "2", False),
        (r"\boxed{\frac{1}{3}}", "0.5", False),
        (r"\boxed{3}", r"\boxed{3}", True),
        (r"\boxed{1000}", "1,000", True),
    ]
    passed = 0
    for i, (pred, gold, exp) in enumerate(cases):
        got = is_correct(pred, gold)
        ok = got == exp
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] case {i}: is_correct({pred!r}, {gold!r}) = {got} (expected {exp})")
    print(f"\n{passed}/{len(cases)} checker tests passed")
    return passed == len(cases)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _test() else 1)
