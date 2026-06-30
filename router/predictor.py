"""Agent 3: difficulty predictor.

Decides a per-query reasoning budget (low/medium/high) WITHOUT calling the model
and WITHOUT seeing the answer.

Two predictors behind one interface (`predict(problem) -> budget`):
  * HeuristicPredictor  — hand-tuned rules over cheap structural signals.
  * LearnedPredictor    — a classifier trained on baseline-derived labels,
                          using only text/structural features of the problem.

Training labels come from the baseline sweep: a problem's label is the MINIMUM
budget at which it was answered correctly (low < medium < high). Problems unsolved
even at high are labelled 'high' for training (conservative); the post-hoc oracle
treats them as 'low' since extra budget is wasted — these costs are asymmetric and
are discussed in the metrics output.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BUDGETS = ["low", "medium", "high"]
BUDGET_IDX = {b: i for i, b in enumerate(BUDGETS)}


# ----------------------------------------------------------------------------
# Labels from baseline data
# ----------------------------------------------------------------------------

def load_baseline_correctness() -> dict[str, dict[str, bool]]:
    """problem_id -> {budget: correct_bool}."""
    path = RESULTS / "baseline_runs.jsonl"
    by_id: dict[str, dict[str, bool]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_id[r["id"]][r["budget"]] = bool(r["correct"])
    return by_id


def min_budget_label(correct_by_budget: dict[str, bool]) -> str:
    """Minimum budget that solved the problem; 'high' if never solved."""
    for b in BUDGETS:
        if correct_by_budget.get(b):
            return b
    return "high"


def oracle_budget(correct_by_budget: dict[str, bool]) -> str:
    """Post-hoc cheapest budget that is correct; 'low' if never solved
    (spending more is pure waste)."""
    for b in BUDGETS:
        if correct_by_budget.get(b):
            return b
    return "low"


# ----------------------------------------------------------------------------
# Features (no API, no answer)
# ----------------------------------------------------------------------------

_OP_TOKENS = [
    r"\\frac", r"\\sqrt", r"\\sum", r"\\int", r"\\prod", r"\\lim",
    r"\\binom", r"\\log", r"\\sin", r"\\cos", r"\\tan", r"\\pi",
    r"\\cdot", r"\\times", r"\\div", r"\\pm", r"\\equiv", r"\\le", r"\\ge",
    "=", "+", "-", "*", "/", "^", "<", ">",
]

_MULTISTEP_CUES = [
    "first", "then", "next", "finally", "after", "subject to", "given that",
    "for all", "for each", "prove", "show that", "in terms of", "how many ways",
    "maximum", "minimum", "least", "greatest", "remainder when",
]


def structural_features(problem_text: str) -> dict[str, float]:
    t = problem_text
    low = t.lower()
    n_ops = sum(t.count(op.replace("\\\\", "\\")) for op in _OP_TOKENS)
    n_distinct_ops = sum(1 for op in _OP_TOKENS if op.replace("\\\\", "\\") in t)
    n_backslash = t.count("\\")
    n_dollar = t.count("$")
    n_cues = sum(low.count(c) for c in _MULTISTEP_CUES)
    n_digits = sum(c.isdigit() for c in t)
    return {
        "char_len": len(t),
        "word_count": len(t.split()),
        "n_ops": n_ops,
        "n_distinct_ops": n_distinct_ops,
        "n_backslash": n_backslash,
        "n_dollar": n_dollar,
        "n_multistep_cues": n_cues,
        "n_digits": n_digits,
        "has_prove": float("prove" in low or "show that" in low),
    }


FEATURE_KEYS = list(structural_features("x").keys())


# ----------------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------------

class HeuristicPredictor:
    """Rule-based. Uses cheap structural signals; optionally the dataset `level`
    if present on the problem (realistic queries may not carry one)."""

    name = "heuristic"

    def __init__(self, use_level: bool = True):
        self.use_level = use_level

    def predict(self, problem: dict) -> str:
        if self.use_level and problem.get("level") is not None:
            lvl = int(problem["level"])
            if lvl <= 2:
                return "low"
            if lvl == 3:
                return "medium"
            return "high"
        # text-only fallback: combine length + operator richness + cues
        f = structural_features(problem["problem"])
        score = (
            (f["char_len"] > 200)
            + (f["n_distinct_ops"] > 6)
            + (f["n_multistep_cues"] > 1)
            + (f["has_prove"] > 0)
            + (f["n_backslash"] > 8)
        )
        if score <= 1:
            return "low"
        if score == 2:
            return "medium"
        return "high"


class LearnedPredictor:
    """Gradient-boosted trees over structural + TF-IDF text features. No level,
    no answer — problem text only."""

    name = "learned"

    def __init__(self, model=None, vectorizer=None, feat_keys=None):
        self.model = model
        self.vectorizer = vectorizer
        self.feat_keys = feat_keys or FEATURE_KEYS

    def _vectorize(self, problems: list[dict]):
        struct = np.array(
            [[structural_features(p["problem"])[k] for k in self.feat_keys] for p in problems],
            dtype=float,
        )
        tfidf = self.vectorizer.transform([p["problem"] for p in problems]).toarray()
        return np.hstack([struct, tfidf])

    def predict(self, problem: dict) -> str:
        X = self._vectorize([problem])
        return self.model.predict(X)[0]

    def predict_batch(self, problems: list[dict]) -> list[str]:
        X = self._vectorize(problems)
        return list(self.model.predict(X))

    def save(self, path: Path):
        import joblib
        joblib.dump(
            {"model": self.model, "vectorizer": self.vectorizer, "feat_keys": self.feat_keys},
            path,
        )

    @classmethod
    def load(cls, path: Path):
        import joblib
        d = joblib.load(path)
        return cls(d["model"], d["vectorizer"], d["feat_keys"])


# ----------------------------------------------------------------------------
# Training + metrics
# ----------------------------------------------------------------------------

def load_sample() -> list[dict]:
    with open(ROOT / "data" / "sample.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_and_eval(test_size: float = 0.3, seed: int | None = None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

    if seed is None:
        with open(ROOT / "config.json") as f:
            seed = json.load(f)["seed"]

    sample = load_sample()
    correctness = load_baseline_correctness()

    problems, labels = [], []
    for p in sample:
        c = correctness.get(p["id"])
        if not c:
            continue
        problems.append(p)
        labels.append(min_budget_label(c))
    labels = np.array(labels)

    idx = np.arange(len(problems))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    train_problems = [problems[i] for i in train_idx]
    test_problems = [problems[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    vec = TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=2)
    vec.fit([p["problem"] for p in train_problems])

    lp = LearnedPredictor(model=None, vectorizer=vec)
    X_train = lp._vectorize(train_problems)
    X_test = lp._vectorize(test_problems)

    # NOTE: the 85%-low label imbalance collapses this classifier to "always
    # low" (it lands on the low baseline). Class-balanced sample weights were
    # tried and made it strictly worse (0.850 acc @ 705 tok vs @ 499) — over-
    # budgeting without recovering accuracy, confirming the text/structural
    # features carry no usable per-problem difficulty signal on this sample.
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    lp.model = clf

    y_pred = clf.predict(X_test)
    labels_order = ["low", "medium", "high"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, labels=labels_order, output_dict=True, zero_division=0
    )

    # heuristic accuracy on the same test set (text-only and level-based)
    heur_level = HeuristicPredictor(use_level=True)
    heur_text = HeuristicPredictor(use_level=False)
    y_heur_level = [heur_level.predict(p) for p in test_problems]
    y_heur_text = [heur_text.predict(p) for p in test_problems]
    acc_heur_level = accuracy_score(y_test, y_heur_level)
    acc_heur_text = accuracy_score(y_test, y_heur_text)

    RESULTS.mkdir(exist_ok=True)
    lp.save(RESULTS / "learned_predictor.joblib")

    # persist the test split ids so the router evaluates on truly held-out data
    test_ids = [p["id"] for p in test_problems]
    train_ids = [p["id"] for p in train_problems]
    with open(RESULTS / "split.json", "w") as f:
        json.dump({"seed": seed, "test_ids": test_ids, "train_ids": train_ids}, f, indent=2)

    label_dist = {b: int((labels == b).sum()) for b in labels_order}
    metrics = {
        "seed": seed,
        "label_definition": "min budget that solved the problem; unsolved->high",
        "label_distribution": label_dist,
        "n_train": len(train_problems),
        "n_test": len(test_problems),
        "learned_test_accuracy": acc,
        "learned_confusion_matrix": {"labels": labels_order, "matrix": cm.tolist()},
        "learned_classification_report": report,
        "heuristic_level_test_accuracy": acc_heur_level,
        "heuristic_text_test_accuracy": acc_heur_text,
        "misclassification_cost_note": (
            "Costs are asymmetric. Predicting hard->low loses accuracy (the problem "
            "needed more budget); predicting easy->high only wastes tokens. A good "
            "predictor should bias toward over-budgeting borderline cases."
        ),
    }
    with open(RESULTS / "predictor_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # console output
    print("=== PREDICTOR METRICS ===")
    print(f"label distribution (all {len(problems)}): {label_dist}")
    print(f"train/test = {len(train_problems)}/{len(test_problems)} (seed {seed})")
    print(f"\nlearned test accuracy : {acc:.3f}")
    print(f"heuristic(level) acc  : {acc_heur_level:.3f}")
    print(f"heuristic(text)  acc  : {acc_heur_text:.3f}")
    print("\nlearned confusion matrix (rows=true, cols=pred) order=[low,medium,high]:")
    print(f"            {'  '.join(f'{l:>7}' for l in labels_order)}")
    for i, l in enumerate(labels_order):
        print(f"  {l:>7}   {'  '.join(f'{v:>7}' for v in cm[i])}")
    print(
        "\nmisclassification cost: hard->low LOSES accuracy; easy->high only "
        "WASTES tokens (asymmetric)."
    )
    return metrics


# ----------------------------------------------------------------------------
# Router-facing factory
# ----------------------------------------------------------------------------

def get_predictor(kind: str):
    if kind == "heuristic":
        return HeuristicPredictor(use_level=True)
    if kind == "heuristic_text":
        return HeuristicPredictor(use_level=False)
    if kind == "learned":
        return LearnedPredictor.load(RESULTS / "learned_predictor.joblib")
    raise ValueError(f"unknown predictor kind {kind!r}")


if __name__ == "__main__":
    train_and_eval()
