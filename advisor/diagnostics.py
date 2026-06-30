"""The diagnostic engine — the three numbers no incumbent tool produces:
minority size, tail-cost concentration (the GO/NO-GO variable), confidence-
signal recall.
"""

from __future__ import annotations

import numpy as np


def minority_mask(records):
    return np.array([(not r["cheap_correct"]) and r["frontier_correct"] for r in records])


def compute(records, workload):
    n = len(records)
    M = minority_mask(records)
    r = float(M.mean())
    fcost = np.array([x["frontier_cost"] for x in records])
    ccost = np.array([x["cheap_cost"] for x in records])

    mean_front = fcost.mean()
    mean_cheap = ccost.mean()
    g = mean_front / mean_cheap if mean_cheap > 0 else float("inf")

    # NARRATIVE concentration (spec's high-stakes minority = cheap-wrong AND
    # frontier-right): mean frontier cost in that minority / overall. Explains
    # WHERE cost sits; not the decision variable on its own.
    if M.sum() > 0 and mean_front > 0:
        C = fcost[M].mean() / mean_front
    else:
        C = 0.0

    # DECISION variable — the g-aware break-even, derived for the actual oracle
    # policy (escalate iff cheap wrong). Phase 1 falsified the fixed-C threshold:
    # the GO/NO-GO line moves with the signal cost (which sets g). Pays iff
    #   cheap_cost + w * (frontier cost on cheap-wrong) < frontier_cost
    # => C_w < (1 - 1/g)/w,  where w = cheap-wrong fraction,
    #    C_w = mean frontier cost on cheap-wrong / mean overall.
    cheap_wrong = np.array([not x["cheap_correct"] for x in records])
    w = float(cheap_wrong.mean())
    C_w = (fcost[cheap_wrong].mean() / mean_front) if (cheap_wrong.sum() and mean_front > 0) else 0.0
    if g <= 1:
        C_w_max = 0.0          # cheap pass already costs >= frontier; escalation can't pay
    elif w <= 0:
        C_w_max = float("inf")  # cheap never wrong; never escalate; trivially pays
    else:
        C_w_max = (1 - 1 / g) / w
    concentration_go = (g > 1) and (C_w < C_w_max)
    # back-compat field names kept; C_max now reflects the g-aware decision threshold
    C_max = C_w_max

    # confidence-signal recall on the minority (1-feature CV)
    sig = np.array([x["signal"] for x in records]).reshape(-1, 1)
    recall_cv = signal_recall(sig, M)
    # training-free: escalate if signal below its max (not fully confident)
    smax = sig.max()
    flagged = (sig.ravel() < smax)
    tp = int((flagged & M).sum())
    rec_thr = tp / M.sum() if M.sum() else None
    prec_thr = tp / flagged.sum() if flagged.sum() else None

    out = {
        "workload": workload, "n": n,
        "minority_size": r, "minority_count": int(M.sum()),
        "mean_frontier_cost": mean_front, "mean_cheap_cost": mean_cheap,
        "cost_gap_g": g,
        "tail_cost_concentration_C": C,
        "cheap_wrong_frac_w": w,
        "escalation_concentration_Cw": C_w,
        "Cw_max_threshold": C_w_max,
        "C_max_threshold": C_max,
        "concentration_go": bool(concentration_go),
        "signal_recall_cv": recall_cv,
        "signal_recall_threshold": rec_thr,
        "signal_precision_threshold": prec_thr,
    }
    return out


def signal_recall(sig, M):
    if M.sum() < 3 or M.sum() == len(M):
        # too few positives for CV; fall back to threshold recall
        smax = sig.max()
        flagged = (sig.ravel() < smax)
        tp = int((flagged & M).sum())
        return tp / M.sum() if M.sum() else None
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    y = M.astype(int)
    folds = min(5, int(M.sum()))
    pred = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        sig, y, cv=folds)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tp / (tp + fn) if (tp + fn) else None
