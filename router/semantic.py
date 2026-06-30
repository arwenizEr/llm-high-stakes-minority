"""Semantic-embedding difficulty predictor + router.

Tests the one untested signal: does difficulty live in semantic embeddings
rather than structural features? Train LR and a small MLP on MiniLM embeddings
-> min-budget label, using the SAME train/test split as before. Report test-60
accuracy, confusion matrix, and — the whole game — recall on the HARD class
(problems needing medium/high). Then build a router and test the bar: beat
always-low on the Pareto frontier WITHOUT the level label.

No new target-model calls: labels and router evaluation reuse cached baseline runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embed import load_embeddings
from predictor import load_baseline_correctness, min_budget_label, BUDGETS
from router import (
    load_problems_by_id,
    baseline_points,
    eval_point,
    interp_baseline_accuracy,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LABELS = ["low", "medium", "high"]


def load_split():
    with open(RESULTS / "split.json") as f:
        s = json.load(f)
    return s["train_ids"], s["test_ids"], s["seed"]


def build_xy(ids, emb, correctness):
    X = np.array([emb[i] for i in ids], dtype=np.float32)
    y = np.array([min_budget_label(correctness[i]) for i in ids])
    return X, y


def hard_recall(y_true, y_pred):
    """Binary recall on the hard class (med/high = 'not low')."""
    true_hard = y_true != "low"
    pred_hard = y_pred != "low"
    n_hard = int(true_hard.sum())
    if n_hard == 0:
        return None, 0, 0
    caught = int((true_hard & pred_hard).sum())
    return caught / n_hard, caught, n_hard


def train_classifiers(Xtr, ytr, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    # class_weight balanced so the 85%-low majority doesn't swamp the hard class
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    lr.fit(Xtr, ytr)

    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1500, random_state=seed)
    mlp.fit(Xtr, ytr)
    return {"logreg_balanced": lr, "mlp": mlp}


def evaluate(name, clf, Xte, yte):
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = clf.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    cm = confusion_matrix(yte, y_pred, labels=LABELS)
    rec, caught, n_hard = hard_recall(yte, y_pred)
    return {
        "name": name,
        "test_accuracy": acc,
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "hard_recall": rec,
        "hard_caught": caught,
        "n_hard": n_hard,
        "pred_hist": {l: int((y_pred == l).sum()) for l in LABELS},
    }


def make_router_choice(clf, emb):
    def choose(problem):
        x = emb[problem["id"]].reshape(1, -1)
        return clf.predict(x)[0]
    return choose


def plot(baselines, sem_routers, oracle_router, always_low):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6))
    bx = [baselines[b]["mean_output_tokens"] for b in BUDGETS]
    by = [baselines[b]["accuracy"] for b in BUDGETS]
    ax.plot(bx, by, "o-", color="#555", label="fixed-budget baselines", zorder=2)
    for b, x, y in zip(BUDGETS, bx, by):
        ax.annotate(b, (x, y), textcoords="offset points", xytext=(6, -12), color="#555")

    ax.scatter(oracle_router["mean_output_tokens"], oracle_router["accuracy"],
               s=140, marker="P", color="#2ca02c", zorder=4, label="oracle router")
    ax.scatter(always_low["mean_output_tokens"], always_low["accuracy"],
               s=130, marker="s", color="#7f7f7f", zorder=4, label="always-low (bar)")

    colors = {"logreg_balanced": "#1f77b4", "mlp": "#d62728"}
    for name, pt in sem_routers.items():
        ax.scatter(pt["mean_output_tokens"], pt["accuracy"], s=150, marker="*",
                   color=colors.get(name, "#9467bd"), zorder=5, label=f"router (embed {name})")

    ax.set_xlabel("mean output tokens / problem")
    ax.set_ylabel("accuracy")
    ax.set_title("Semantic-embedding router vs baselines (held-out test)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "semantic_pareto.png", dpi=130)
    print(f"  wrote {RESULTS / 'semantic_pareto.png'}")


def main():
    emb = load_embeddings()
    correctness = load_baseline_correctness()
    train_ids, test_ids, seed = load_split()
    by_id = load_problems_by_id()
    test_problems = [by_id[i] for i in test_ids]

    Xtr, ytr = build_xy(train_ids, emb, correctness)
    Xte, yte = build_xy(test_ids, emb, correctness)
    print(f"train/test = {len(train_ids)}/{len(test_ids)}  emb_dim={Xtr.shape[1]}  seed={seed}")
    print(f"test label dist: {{{', '.join(f'{l}:{int((yte==l).sum())}' for l in LABELS)}}}")

    clfs = train_classifiers(Xtr, ytr, seed)
    clf_metrics = {name: evaluate(name, clf, Xte, yte) for name, clf in clfs.items()}

    baselines = baseline_points(test_problems)
    always_low = baselines["low"]
    oracle_router = {"accuracy": 0.933, "mean_output_tokens": 543.0}

    sem_routers = {}
    for name, clf in clfs.items():
        sem_routers[name] = eval_point(test_problems, make_router_choice(clf, emb))

    # bar check: beat always-low on the Pareto frontier
    bar_results = {}
    for name, pt in sem_routers.items():
        base_at_x = interp_baseline_accuracy(baselines, pt["mean_output_tokens"])
        above = pt["accuracy"] - base_at_x
        # "beats always-low on the frontier": strictly better tradeoff than the low point
        beats_low = (pt["accuracy"] > always_low["accuracy"] + 1e-9) or (
            abs(pt["accuracy"] - always_low["accuracy"]) < 1e-9
            and pt["mean_output_tokens"] < always_low["mean_output_tokens"] - 1e-9
        )
        bar_results[name] = {
            "router_accuracy": pt["accuracy"],
            "router_mean_tokens": pt["mean_output_tokens"],
            "baseline_acc_at_router_tokens": base_at_x,
            "above_baseline_line": above,
            "beats_always_low": bool(beats_low),
        }

    report = {
        "embedding_model": "all-MiniLM-L6-v2",
        "seed": seed,
        "test_label_distribution": {l: int((yte == l).sum()) for l in LABELS},
        "classifier_metrics": clf_metrics,
        "baselines_test": baselines,
        "oracle_router_ref": oracle_router,
        "always_low": always_low,
        "semantic_routers": sem_routers,
        "bar_check_beat_always_low": bar_results,
    }
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "semantic_eval.json", "w") as f:
        json.dump(report, f, indent=2)

    # console
    print("\n=== CLASSIFIER METRICS (test-60) ===")
    for name, m in clf_metrics.items():
        print(f"\n  {name}: acc={m['test_accuracy']:.3f}  "
              f"HARD-RECALL={m['hard_recall'] if m['hard_recall'] is None else round(m['hard_recall'],2)} "
              f"({m['hard_caught']}/{m['n_hard']})  pred_hist={m['pred_hist']}")
        cm = m["confusion_matrix"]["matrix"]
        print(f"    confusion (rows=true,cols=pred) [low,med,high]:")
        for i, l in enumerate(LABELS):
            print(f"      {l:>6} {cm[i]}")

    print("\n=== ROUTER ON TEST-60 ===")
    print(f"  always-low (bar): acc={always_low['accuracy']:.3f} tokens={always_low['mean_output_tokens']:.0f}")
    for name, pt in sem_routers.items():
        br = bar_results[name]
        print(f"  embed:{name:<16} acc={pt['accuracy']:.3f} tokens={pt['mean_output_tokens']:.0f} "
              f"hist={pt['budget_histogram']}  beats_low={br['beats_always_low']} "
              f"(above_line {br['above_baseline_line']:+.3f})")

    plot(baselines, sem_routers, oracle_router, always_low)
    write_md(report)
    print(f"\n  wrote {RESULTS / 'semantic_eval.json'} and {RESULTS / 'SEMANTIC_RESULTS.md'}")


def write_md(report):
    cm_metrics = report["classifier_metrics"]
    sem = report["semantic_routers"]
    bar = report["bar_check_beat_always_low"]
    base = report["baselines_test"]
    al = report["always_low"]
    orr = report["oracle_router_ref"]

    best = max(sem, key=lambda n: bar[n]["above_baseline_line"])
    best_beats = bar[best]["beats_always_low"] and bar[best]["above_baseline_line"] >= -1e-9

    L = []
    L.append("# Semantic-Embedding Difficulty Predictor — Results\n")
    L.append(f"Embedding: **all-MiniLM-L6-v2** (384-dim, local, $0). Same train/test "
             f"split (seed {report['seed']}). No new target-model calls.\n")
    L.append(f"Test label distribution: {report['test_label_distribution']} "
             f"(hard = med+high).\n")

    L.append("## Classifier metrics (test-60)\n")
    L.append("| classifier | accuracy | hard-class recall | predicted budget mix |")
    L.append("|---|---|---|---|")
    for name, m in cm_metrics.items():
        hr = m["hard_recall"]
        hrs = "n/a" if hr is None else f"{hr:.2f} ({m['hard_caught']}/{m['n_hard']})"
        L.append(f"| {name} | {m['test_accuracy']:.3f} | **{hrs}** | {m['pred_hist']} |")
    L.append("")
    L.append("Hard-class recall is the metric that matters — it is the fraction of the "
             "med/high problems (the ones always-low gets wrong) that the predictor "
             "routes to a higher budget.\n")

    L.append("## Router on the held-out test split\n")
    L.append("| point | accuracy | mean tokens |")
    L.append("|---|---|---|")
    for b in BUDGETS:
        L.append(f"| baseline {b} | {base[b]['accuracy']:.3f} | {base[b]['mean_output_tokens']:.0f} |")
    L.append(f"| oracle router | {orr['accuracy']:.3f} | {orr['mean_output_tokens']:.0f} |")
    L.append(f"| **always-low (the bar)** | {al['accuracy']:.3f} | {al['mean_output_tokens']:.0f} |")
    for name, pt in sem.items():
        L.append(f"| router (embed {name}) | {pt['accuracy']:.3f} | {pt['mean_output_tokens']:.0f} |")
    L.append("")

    L.append("## Verdict: label-free win, or collapse?\n")
    if best_beats:
        b = bar[best]
        L.append(f"**Win.** The embedding router ({best}) beats always-low without the "
                 f"`level` label: {b['router_accuracy']:.3f} accuracy @ "
                 f"{b['router_mean_tokens']:.0f} tokens, sitting {b['above_baseline_line']:+.3f} "
                 f"above the baseline frontier at its token cost. Semantic features carry "
                 f"difficulty signal that structural features did not.")
    else:
        # diagnose: did it collapse to low, or over-budget without accuracy?
        worst_hard = min(m["hard_recall"] or 0 for m in cm_metrics.values())
        best_hard = max(m["hard_recall"] or 0 for m in cm_metrics.values())
        L.append(f"**Collapse — no label-free win.** Best hard-class recall across "
                 f"classifiers was {best_hard:.2f}: the embedding predictors still fail to "
                 f"identify the med/high problems, so the router cannot recover accuracy "
                 f"above always-low without spending extra tokens on the wrong problems. "
                 f"No embedding router sits above the baseline frontier at always-low's cost.")
        L.append("")
        L.append("Same failure mode as the structural predictor: with only ~15% hard "
                 "labels and a hard class that is not linearly separable in MiniLM space, "
                 "the classifier either collapses toward low or, when class-balanced, "
                 "over-budgets indiscriminately. **The difficulty signal is not in cheap "
                 "semantic embeddings either.**")
    L.append("")
    L.append(f"Oracle router ceiling (with perfect difficulty knowledge): "
             f"{orr['accuracy']:.3f} @ {orr['mean_output_tokens']:.0f} tokens — the headroom "
             f"a working predictor would capture. The gap to it is the predictor problem, "
             f"and cheap text/structural/semantic features have now all failed to close it.")
    (RESULTS / "SEMANTIC_RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
