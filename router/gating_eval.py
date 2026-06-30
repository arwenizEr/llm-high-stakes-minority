"""Gating Agents 2 + 3.

A2: surface-feature classifier minority-recall (the bar) vs model-internal
    self-consistency signal minority-recall, on the identical label set.
A3: routed system (cheap, escalate on low agreement) vs always-cheap,
    always-frontier, surface-router, oracle-router. Pareto + GO/NO-GO verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import harness as H
from gating import run_cheap, load_sample, K, CHEAP_MODEL, PRICE as CHEAP_PRICE

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SONNET_PRICE = {"input": 3.0, "output": 15.0}


def load_labels():
    rows = []
    with open(RESULTS / "routing_labels.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_frontier_tokens():
    d = {}
    with open(RESULTS / "baseline_runs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["budget"] == "high":
                d[r["id"]] = (r["input_tokens"], r["output_tokens"], bool(r["correct"]))
    return d


def split_ids():
    s = json.loads((RESULTS / "split.json").read_text())
    return set(s["train_ids"]), set(s["test_ids"])


# ---------- recall helper ----------

def recall_precision(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    return {"recall": rec, "precision": prec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------- A2 ----------

def agent2(labels, train_ids, test_ids):
    by_id = {l["id"]: l for l in labels}
    test = [l for l in labels if l["id"] in test_ids]
    train = [l for l in labels if l["id"] in train_ids]
    y_test = np.array([1 if l["minority"] else 0 for l in test])

    # --- surface baseline: MiniLM embeddings -> minority ---
    from embed import load_embeddings
    from sklearn.linear_model import LogisticRegression
    emb = load_embeddings()
    Xtr = np.array([emb[l["id"]] for l in train]); ytr = np.array([1 if l["minority"] else 0 for l in train])
    Xte = np.array([emb[l["id"]] for l in test])
    surf = LogisticRegression(max_iter=2000, class_weight="balanced")
    surf.fit(Xtr, ytr)
    surf_pred = surf.predict(Xte)
    surf_metrics = recall_precision(y_test, surf_pred)

    # --- internal signal: self-consistency agreement -> minority ---
    # lower agreement = lower confidence = more likely minority. 1-feature LR.
    atr = np.array([[l["agreement"]] for l in train])
    ate = np.array([[l["agreement"]] for l in test])
    intern = LogisticRegression(max_iter=2000, class_weight="balanced")
    intern.fit(atr, ytr)
    int_pred = intern.predict(ate)
    int_metrics = recall_precision(y_test, int_pred)

    # also a pure threshold view: escalate if not unanimous (agreement < 1.0)
    thr_pred = np.array([1 if l["agreement"] < 1.0 else 0 for l in test])
    thr_metrics = recall_precision(y_test, thr_pred)

    out = {
        "n_test": len(test), "n_test_minority": int(y_test.sum()),
        "surface_embeddings": surf_metrics,
        "internal_selfconsistency_lr": int_metrics,
        "internal_threshold_not_unanimous": thr_metrics,
    }
    (RESULTS / "signal_comparison.json").write_text(json.dumps(out, indent=2))
    print("=== A2: SURFACE vs INTERNAL (minority recall on test-%d) ===" % len(test))
    print(f"  test minority count: {int(y_test.sum())}/{len(test)}")
    print(f"  surface embeddings    recall={surf_metrics['recall']}  prec={surf_metrics['precision']}")
    print(f"  internal self-consist recall={int_metrics['recall']}  prec={int_metrics['precision']}")
    print(f"  internal threshold(!unanimous) recall={thr_metrics['recall']}  prec={thr_metrics['precision']}")
    return out, surf, intern


# ---------- A3 ----------

def cheap_cost(problem):
    tin = tout = 0
    for s in range(K):
        r = run_cheap(problem, s)  # cached
        tin += r["input_tokens"]; tout += r["output_tokens"]
    return tin / 1e6 * CHEAP_PRICE[CHEAP_MODEL]["input"] + tout / 1e6 * CHEAP_PRICE[CHEAP_MODEL]["output"]


def frontier_cost(ftoks):
    return ftoks[0] / 1e6 * SONNET_PRICE["input"] + ftoks[1] / 1e6 * SONNET_PRICE["output"]


def agent3(labels, test_ids):
    by_id = {p["id"]: p for p in load_sample()}
    frontier = load_frontier_tokens()
    test = [l for l in labels if l["id"] in test_ids]
    n = len(test)

    # precompute per-problem cheap cost + frontier cost/correct
    pc = {}
    for l in test:
        p = by_id[l["id"]]
        ft = frontier[l["id"]]
        pc[l["id"]] = {
            "cheap_cost": cheap_cost(p), "cheap_correct": l["cheap_correct"],
            "frontier_cost": frontier_cost(ft), "frontier_correct": ft[2],
            "agreement": l["agreement"], "minority": l["minority"],
        }

    def system(escalate_fn):
        acc = cost = esc = 0
        for l in test:
            d = pc[l["id"]]
            cost += d["cheap_cost"]  # always pay the cheap pass
            if escalate_fn(d):
                esc += 1
                cost += d["frontier_cost"]
                acc += d["frontier_correct"]
            else:
                acc += d["cheap_correct"]
        return {"accuracy": acc / n, "cost_usd_per_q": cost / n, "escalation_rate": esc / n}

    # anchors
    always_cheap = system(lambda d: False)
    always_frontier = {
        "accuracy": sum(pc[l["id"]]["frontier_correct"] for l in test) / n,
        "cost_usd_per_q": sum(pc[l["id"]]["frontier_cost"] for l in test) / n,
        "escalation_rate": 1.0,
    }
    oracle = system(lambda d: not d["cheap_correct"])  # escalate iff cheap wrong (ceiling)

    # internal-signal router at two thresholds
    router_T1 = system(lambda d: d["agreement"] < 1.0)     # escalate if not unanimous
    router_T05 = system(lambda d: d["agreement"] < 0.5)    # escalate only if all-disagree

    # surface-feature router: escalate if surface classifier predicts minority
    from embed import load_embeddings
    from sklearn.linear_model import LogisticRegression
    train_ids = set(json.loads((RESULTS / "split.json").read_text())["train_ids"])
    train = [l for l in labels if l["id"] in train_ids]
    emb = load_embeddings()
    surf = LogisticRegression(max_iter=2000, class_weight="balanced")
    surf.fit(np.array([emb[l["id"]] for l in train]),
             np.array([1 if l["minority"] else 0 for l in train]))
    surf_pred = {l["id"]: int(surf.predict(emb[l["id"]].reshape(1, -1))[0]) for l in test}
    surface_router = system(lambda d, did=None: False)  # placeholder, replaced below
    # recompute surface router properly (needs id)
    acc = cost = esc = 0
    for l in test:
        d = pc[l["id"]]; cost += d["cheap_cost"]
        if surf_pred[l["id"]] == 1:
            esc += 1; cost += d["frontier_cost"]; acc += d["frontier_correct"]
        else:
            acc += d["cheap_correct"]
    surface_router = {"accuracy": acc / n, "cost_usd_per_q": cost / n, "escalation_rate": esc / n}

    points = {
        "always_cheap": always_cheap, "always_frontier": always_frontier,
        "oracle_router": oracle, "internal_router_not_unanimous": router_T1,
        "internal_router_all_disagree": router_T05, "surface_router": surface_router,
    }

    # fraction of oracle savings captured (vs always-frontier cost, at ~matched acc)
    af = always_frontier
    def savings_frac(pt):
        denom = af["cost_usd_per_q"] - oracle["cost_usd_per_q"]
        return (af["cost_usd_per_q"] - pt["cost_usd_per_q"]) / denom if denom else 0.0

    report = {"n_test": n, "points": points,
              "internal_router_savings_frac_of_oracle": savings_frac(router_T1),
              "surface_router_savings_frac_of_oracle": savings_frac(surface_router)}
    (RESULTS / "routing_system_eval.json").write_text(json.dumps(report, indent=2))

    print("\n=== A3: ROUTED SYSTEM (test-%d) ===" % n)
    for name, pt in points.items():
        print(f"  {name:<32} acc={pt['accuracy']:.3f}  cost=${pt['cost_usd_per_q']:.5f}/q  "
              f"escalate={pt['escalation_rate']:.2f}")
    return points, report


def plot(points):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 6))
    style = {
        "always_cheap": ("o", "#7f7f7f"), "always_frontier": ("s", "#2ca02c"),
        "oracle_router": ("X", "#17becf"), "internal_router_not_unanimous": ("*", "#d62728"),
        "internal_router_all_disagree": ("*", "#ff7f0e"), "surface_router": ("P", "#1f77b4"),
    }
    for name, pt in points.items():
        m, c = style[name]
        ax.scatter(pt["cost_usd_per_q"] * 1000, pt["accuracy"], s=150, marker=m, color=c, label=name, zorder=4)
    ax.set_xlabel("cost (milli-USD / question)"); ax.set_ylabel("system accuracy")
    ax.set_title("Model-routing: internal signal vs surface vs oracle (MATH test)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "final_routing_pareto.png", dpi=130)
    print(f"  wrote {RESULTS / 'final_routing_pareto.png'}")


def write_verdict(a2, points, report):
    surf_rec = a2["surface_embeddings"]["recall"] or 0.0
    int_rec = a2["internal_selfconsistency_lr"]["recall"] or 0.0
    thr_rec = a2["internal_threshold_not_unanimous"]["recall"] or 0.0
    best_int = max(int_rec, thr_rec)
    af = points["always_frontier"]; ir = points["internal_router_not_unanimous"]

    # (a) internal clearly beats surface bar
    a_go = best_int >= surf_rec + 0.25  # "wide margin"
    # (b) routed system above/left of always-frontier: >= matched accuracy at lower cost
    b_go = (ir["accuracy"] >= af["accuracy"] - 0.02) and (ir["cost_usd_per_q"] < af["cost_usd_per_q"])
    go = a_go and b_go

    L = []
    L.append("# Model-Internal Signal — GO / NO-GO Verdict\n")
    L.append("Signal: **self-consistency** (Anthropic API exposes no token logprobs). "
             f"Cheap={CHEAP_MODEL}, frontier=sonnet-4-6 effort=high (cached).\n")
    L.append(f"High-stakes minority on test: {a2['n_test_minority']}/{a2['n_test']}.\n")
    L.append("## (a) Does the internal signal beat the surface bar on minority recall?\n")
    L.append(f"- surface embeddings recall: **{surf_rec:.2f}**")
    L.append(f"- internal self-consistency recall (LR): **{int_rec:.2f}**")
    L.append(f"- internal threshold (escalate if not unanimous): **{thr_rec:.2f}**")
    L.append(f"- best internal: **{best_int:.2f}** vs surface **{surf_rec:.2f}** "
             f"→ {'YES, wide margin' if a_go else 'NO, not a wide margin'}\n")
    L.append("## (b) Does the routed system beat always-frontier on cost at matched accuracy?\n")
    L.append(f"- always-frontier: acc {af['accuracy']:.3f} @ ${af['cost_usd_per_q']*1000:.2f} m$/q")
    L.append(f"- internal router (not-unanimous): acc {ir['accuracy']:.3f} @ ${ir['cost_usd_per_q']*1000:.2f} m$/q, "
             f"escalate {ir['escalation_rate']:.2f}")
    L.append(f"- oracle router: acc {points['oracle_router']['accuracy']:.3f} @ "
             f"${points['oracle_router']['cost_usd_per_q']*1000:.2f} m$/q")
    L.append(f"- internal router captures {report['internal_router_savings_frac_of_oracle']*100:.0f}% of oracle savings; "
             f"surface router {report['surface_router_savings_frac_of_oracle']*100:.0f}% "
             f"→ {'YES' if b_go else 'NO'}\n")
    L.append("## VERDICT\n")
    if go:
        L.append("**GO.** Model-internal self-consistency identifies the high-stakes minority "
                 "with recall well above the surface bar, and the routed system lands above/left "
                 "of always-frontier — real savings without tail collapse. Build the tail-aware "
                 "optimizer; this experiment is its foundation.")
    else:
        reasons = []
        if not a_go:
            reasons.append(f"internal recall ({best_int:.2f}) does not clear the surface bar "
                           f"({surf_rec:.2f}) by a wide margin")
        if not b_go:
            reasons.append("the routed system does not beat always-frontier on cost at matched accuracy")
        L.append(f"**NO-GO.** {'; '.join(reasons)}. Minority detection from a cheap model's own "
                 "confidence is not decisively better than surface features here. This is the "
                 "fourth and final negative completing the arc — build the quality-aware lever "
                 "advisor instead of the tail-aware optimizer.")
    (RESULTS / "VERDICT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n  wrote VERDICT.md  ->  {'GO' if go else 'NO-GO'}")


def main():
    labels = load_labels()
    train_ids, test_ids = split_ids()
    a2, _, _ = agent2(labels, train_ids, test_ids)
    points, report = agent3(labels, test_ids)
    plot(points)
    write_verdict(a2, points, report)


if __name__ == "__main__":
    main()
