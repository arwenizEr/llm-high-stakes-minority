"""Agent 4: downstream-aware compressor + evaluation (the headline).

Features per span: intrinsic importance (GPT-2 perplexity, LLMLingua-style) PLUS
position / length / question-overlap. Train a span classifier on the downstream
labels from Agent 3 (load-bearing vs droppable). Build compress_downstream that
ranks spans by predicted droppability and drops the most-droppable until the rate
is hit, never dropping a high-confidence load-bearing span. Evaluate on held-out
test across the baseline rates; compare to LLMLingua-2.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from harness import CONFIG, MODEL, run_qa
from labeler import enumerate_spans, load_sample
from compressors import get_compressor

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RATES = CONFIG["rates"]
FEATS = ["intrinsic_ppl", "position", "n_words", "q_overlap"]

# --- intrinsic importance: GPT-2 perplexity ---------------------------------
_gpt2 = None
_gpt2_tok = None
_ppl_cache: dict[str, float] = {}


def _load_gpt2():
    global _gpt2, _gpt2_tok
    if _gpt2 is None:
        import torch  # noqa
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
        _gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    return _gpt2, _gpt2_tok


def span_perplexity(text: str) -> float:
    if text in _ppl_cache:
        return _ppl_cache[text]
    import torch
    model, tok = _load_gpt2()
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids
    if ids.shape[1] < 2:
        val = 0.0
    else:
        with torch.no_grad():
            loss = model(ids, labels=ids).loss.item()
        val = math.exp(min(loss, 20))
    _ppl_cache[text] = val
    return val


def span_features(problem, gran):
    rows = []
    for key, text, feats in enumerate_spans(problem, gran):
        rows.append({
            "key": key, "text": text,
            "intrinsic_ppl": span_perplexity(text),
            "position": feats["position"],
            "n_words": feats["n_words"],
            "q_overlap": feats["q_overlap"],
        })
    return rows


# --- multi-span removal -----------------------------------------------------

def context_without_set(problem, drop_keys: set):
    parts = []
    for pi, para in enumerate(problem["paragraphs"]):
        if (pi, None) in drop_keys:
            continue
        sents = [s for si, s in enumerate(para["sentences"]) if (pi, si) not in drop_keys]
        parts.append(para["title"] + ". " + " ".join(sents))
    return "\n\n".join(parts)


# --- classifier -------------------------------------------------------------

def load_labels():
    rows = []
    with open(RESULTS / "span_labels.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    meta = json.loads((RESULTS / "span_labels_meta.json").read_text())
    return rows, meta["granularity"]


def add_intrinsic(rows):
    for r in rows:
        r["intrinsic_ppl"] = span_perplexity(r["span_text"])
    return rows


def train_classifier(rows, seed):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

    X = np.array([[r[f] for f in FEATS] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "load_bearing" else 0 for r in rows])  # 1 = load-bearing
    groups = np.array([r["qid"] for r in rows])

    from sklearn.utils.class_weight import compute_sample_weight

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    # 90/10 droppable/load-bearing imbalance -> balance so load-bearing isn't ignored
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    y_pred = clf.predict(X[te])
    metrics = {
        "test_accuracy": accuracy_score(y[te], y_pred),
        "load_bearing_recall": recall_score(y[te], y_pred, pos_label=1, zero_division=0),
        "confusion_matrix_labels": ["droppable(0)", "load_bearing(1)"],
        "confusion_matrix": confusion_matrix(y[te], y_pred, labels=[0, 1]).tolist(),
        "n_train_spans": int(len(tr)), "n_test_spans": int(len(te)),
        "class_balance_load_bearing": float(y.mean()),
    }
    # final model on all labeled spans for the compressor
    final = GradientBoostingClassifier(random_state=seed)
    final.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return final, metrics


def make_downstream_compressor(clf, gran):
    LOAD_CONF = 0.7  # never drop a span with P(load-bearing) >= this

    def compress(problem, rate):
        rows = span_features(problem, gran)
        X = np.array([[r[f] for f in FEATS] for r in rows], dtype=float)
        p_load = clf.predict_proba(X)[:, 1]  # P(load-bearing)
        total_tok = sum(r["n_words"] for r in rows) or 1
        target = rate * total_tok
        # candidate drop order: least load-bearing first
        order = sorted(range(len(rows)), key=lambda i: p_load[i])
        drop_keys, dropped = set(), 0
        for i in order:
            if dropped >= target:
                break
            if p_load[i] >= LOAD_CONF:
                continue  # protect confident load-bearing spans
            drop_keys.add(rows[i]["key"])
            dropped += rows[i]["n_words"]
        return context_without_set(problem, drop_keys)

    return compress


# --- evaluation -------------------------------------------------------------

def eval_strategy(problems, strategy, compress_fn, rate):
    n = len(problems)
    corr = tin = 0.0
    with ThreadPoolExecutor(max_workers=CONFIG.get("concurrency", 4)) as ex:
        futs = [ex.submit(run_qa, p, rate, strategy=strategy, compress_fn=compress_fn) for p in problems]
        recs = [f.result() for f in futs]
    for r in recs:
        corr += r["correct"]; tin += r["input_tokens"]
    return {"accuracy": corr / n, "mean_input_tokens": tin / n, "rate": rate, "n": n}


def load_test_problems():
    split = json.loads((RESULTS / "compressor_split.json").read_text())
    by_id = {p["id"]: p for p in load_sample()}
    return [by_id[i] for i in split["test_ids"]], split["seed"]


def baseline_curves_on_test(test_problems):
    """Recompute baseline points on the TEST split (cached)."""
    curves = {}
    none = eval_strategy(test_problems, "none", get_compressor("none"), 0.0)
    for strat in ("random_drop", "stopword_drop", "llmlingua2"):
        pts = [none]
        for r in RATES:
            pts.append(eval_strategy(test_problems, strat, get_compressor(strat), r))
        curves[strat] = pts
    return none, curves


def at_matched(curve_ref, pt):
    """Interpolate ref curve accuracy at pt's token cost."""
    xs = sorted((p["mean_input_tokens"], p["accuracy"]) for p in curve_ref)
    x = pt["mean_input_tokens"]
    if x <= xs[0][0]:
        return xs[0][1]
    if x >= xs[-1][0]:
        return xs[-1][1]
    for (x0, y0), (x1, y1) in zip(xs, xs[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0
            return y0 + t * (y1 - y0)
    return xs[-1][1]


def plot(none, baseline_curves, ds_curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"random_drop": "#7f7f7f", "stopword_drop": "#1f77b4", "llmlingua2": "#d62728"}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(none["mean_input_tokens"], none["accuracy"], s=120, marker="P",
               color="#2ca02c", zorder=5, label="none (full)")
    for strat, pts in baseline_curves.items():
        xs = [p["mean_input_tokens"] for p in pts]; ys = [p["accuracy"] for p in pts]
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in o], [ys[i] for i in o], "o-", color=colors[strat], label=strat, zorder=3)
    xs = [p["mean_input_tokens"] for p in ds_curve]; ys = [p["accuracy"] for p in ds_curve]
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    ax.plot([xs[i] for i in o], [ys[i] for i in o], "*-", color="#ff7f0e",
            markersize=14, linewidth=2, label="downstream-aware (ours)", zorder=6)
    ax.set_xlabel("mean input tokens / question"); ax.set_ylabel("accuracy")
    ax.set_title("Downstream-aware vs LLMLingua-2 (held-out test, HotpotQA)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(RESULTS / "final_pareto.png", dpi=130)
    print(f"  wrote {RESULTS / 'final_pareto.png'}")


def main():
    test_problems, seed = load_test_problems()
    rows, gran = load_labels()
    print(f"granularity={gran}  labeled spans={len(rows)}  test questions={len(test_problems)}")

    print("computing GPT-2 perplexity on labeled spans ...")
    rows = add_intrinsic(rows)
    clf, cls_metrics = train_classifier(rows, seed)
    print(f"classifier: acc={cls_metrics['test_accuracy']:.3f}  "
          f"LOAD-BEARING RECALL={cls_metrics['load_bearing_recall']:.3f}  "
          f"class_balance(load)={cls_metrics['class_balance_load_bearing']:.2f}")
    print(f"  confusion {cls_metrics['confusion_matrix_labels']}: {cls_metrics['confusion_matrix']}")

    ds_compress = make_downstream_compressor(clf, gran)

    print("evaluating on test split ...")
    none, baseline_curves = baseline_curves_on_test(test_problems)
    ds_curve = [none] + [eval_strategy(test_problems, "downstream", ds_compress, r) for r in RATES]

    ll = baseline_curves["llmlingua2"]
    # headline: at each ds rate, accuracy vs llmlingua at matched tokens
    headline = []
    for pt in ds_curve[1:]:
        ll_at = at_matched(ll, pt)
        headline.append({
            "rate": pt["rate"], "ds_tokens": pt["mean_input_tokens"], "ds_acc": pt["accuracy"],
            "llmlingua_acc_at_matched_tokens": ll_at, "acc_gain_vs_llmlingua": pt["accuracy"] - ll_at,
        })

    report = {
        "config": {"model": MODEL, "granularity": gran, "rates": RATES, "features": FEATS},
        "classifier_metrics": cls_metrics,
        "none": none,
        "baseline_curves": baseline_curves,
        "downstream_curve": ds_curve,
        "headline_vs_llmlingua": headline,
    }
    (RESULTS / "compressor_eval.json").write_text(json.dumps(report, indent=2))
    plot(none, baseline_curves, ds_curve)
    write_md(report)
    print(f"\n  wrote compressor_eval.json, final_pareto.png, RESULTS.md")
    for h in headline:
        print(f"  rate {h['rate']}: ds {h['ds_acc']:.3f}@{h['ds_tokens']:.0f}tok  "
              f"vs llmlingua {h['llmlingua_acc_at_matched_tokens']:.3f} "
              f"(gain {h['acc_gain_vs_llmlingua']:+.3f})")


def write_md(report):
    cm = report["classifier_metrics"]
    ds = report["downstream_curve"]
    ll = {p["rate"]: p for p in report["baseline_curves"]["llmlingua2"]}
    none = report["none"]
    hl = report["headline_vs_llmlingua"]
    gains = [h["acc_gain_vs_llmlingua"] for h in hl]
    mean_gain = sum(gains) / len(gains) if gains else 0
    wins = sum(g > 1e-9 for g in gains)

    L = []
    L.append("# Downstream-Aware Prompt Compressor — Results\n")
    L.append(f"Model {report['config']['model']}, HotpotQA distractor. Granularity "
             f"{report['config']['granularity']}. Features: {report['config']['features']}.\n")
    L.append("## Span classifier (downstream label ~ features)\n")
    L.append(f"- held-out accuracy: **{cm['test_accuracy']:.3f}**")
    L.append(f"- **load-bearing recall: {cm['load_bearing_recall']:.3f}** (the spans we must not drop)")
    L.append(f"- class balance (load-bearing frac): {cm['class_balance_load_bearing']:.2f} "
             f"— {'balanced, no majority collapse' if 0.15 < cm['class_balance_load_bearing'] < 0.85 else 'imbalanced'}")
    L.append(f"- confusion {cm['confusion_matrix_labels']}: {cm['confusion_matrix']}\n")

    L.append("## Pareto: downstream-aware vs LLMLingua-2 (held-out test)\n")
    L.append("| rate | downstream acc | downstream tokens | LLMLingua-2 acc | LLMLingua-2 tokens |")
    L.append("|---|---|---|---|---|")
    L.append(f"| 0 (full) | {none['accuracy']:.3f} | {none['mean_input_tokens']:.0f} | — | — |")
    for pt in ds[1:]:
        r = pt["rate"]; lp = ll[r]
        L.append(f"| {r} | {pt['accuracy']:.3f} | {pt['mean_input_tokens']:.0f} | "
                 f"{lp['accuracy']:.3f} | {lp['mean_input_tokens']:.0f} |")
    L.append("")
    L.append("## Headline (accuracy at matched token budget)\n")
    for h in hl:
        L.append(f"- rate {h['rate']}: downstream {h['ds_acc']:.3f} @ {h['ds_tokens']:.0f} tok vs "
                 f"LLMLingua-2 {h['llmlingua_acc_at_matched_tokens']:.3f} at the same cost "
                 f"= **{h['acc_gain_vs_llmlingua']:+.3f}**")
    L.append("")
    L.append("## Verdict\n")
    if mean_gain > 1e-9 and wins >= len(gains) / 2:
        L.append(f"**Win.** Downstream-awareness beats intrinsic perplexity: the compressor's "
                 f"curve sits above LLMLingua-2 at {wins}/{len(gains)} rates, mean accuracy gain "
                 f"**{mean_gain:+.3f}** at matched token budget. Calibrating against answer "
                 f"survival keeps load-bearing spans that perplexity discards.")
    else:
        L.append(f"**No / inconclusive.** Mean accuracy gain vs LLMLingua-2 at matched tokens is "
                 f"**{mean_gain:+.3f}** ({wins}/{len(gains)} rates above). Downstream labels do not "
                 f"beat intrinsic perplexity at this scale. Likely causes: small label set "
                 f"({cm['n_train_spans']}+{cm['n_test_spans']} spans), coarse granularity, or the "
                 f"load-bearing recall ({cm['load_bearing_recall']:.2f}) too low to protect the "
                 f"spans that matter. A clean negative is still a result.")
    (RESULTS / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
