"""Generate summary figures for the consolidated research doc + curate existing
Pareto plots into figures/."""
import json, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- Figure 1: the detection ceiling (confident-error recall across mechanisms) ----
mechs = ["self-consistency\n(k=3)", "self-verification\n(n=28)",
         "weak cross-model\n(genuine)", "capable cross-provider\n(gpt-4o-mini)"]
recall = [0.00, 0.21, 0.21, 0.21]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(len(mechs)), recall, color=["#999", "#1f77b4", "#9467bd", "#d62728"], zorder=3)
ax.axhline(0.21, ls="--", color="#444", lw=1.2, zorder=2, label="ceiling = 0.21")
# annotate the 0.86 artifact
ax.annotate("0.86 literal\n= weak-model artifact\n(blanket disagreement)", xy=(2, 0.21), xytext=(2.05, 0.55),
            fontsize=8, color="#9467bd", ha="center",
            arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1))
for i, v in enumerate(recall):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
ax.set_xticks(range(len(mechs))); ax.set_xticklabels(mechs, fontsize=8)
ax.set_ylabel("confident-error recall"); ax.set_ylim(0, 0.75)
ax.set_title("Cheap-detection ceiling: confident errors caught, by mechanism")
ax.legend(loc="upper left"); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "detection_ceiling.png", dpi=130); plt.close(fig)
print("wrote detection_ceiling.png")

# ---- Figure 2: self-gating separation (the capstone) ----
d = json.load(open(ROOT / "advisor" / "results" / "phase3_eval.json"))
t = sorted(d["table"], key=lambda r: r["disagree_rate"])
xs = [r["disagree_rate"] * 100 for r in t]
labels = [r["workload"] for r in t]
colors = ["#2ca02c" if r["realized_go"] else "#d62728" for r in t]
markers = ["*" if r["held_out"] else "o" for r in t]
fig, ax = plt.subplots(figsize=(9, 5))
for i, r in enumerate(t):
    ax.scatter(xs[i], i, s=200 if r["held_out"] else 130, marker=markers[i], color=colors[i], zorder=4,
               edgecolors="black", linewidths=0.6)
    ax.annotate(labels[i] + ("  (held-out)" if r["held_out"] else ""), (xs[i], i),
                textcoords="offset points", xytext=(12, 0), va="center", fontsize=9)
ax.axvline(50, ls="--", color="#1f77b4", lw=1.5, label="frozen threshold = 50%")
ax.axvspan(0, 50, alpha=0.06, color="#2ca02c"); ax.axvspan(50, 105, alpha=0.06, color="#d62728")
ax.text(25, 2.3, "GO-eligible\n(route on disagreement)", ha="center", color="#2ca02c", fontsize=9)
ax.text(78, 1.2, "blind-spot\n(abstain → frontier)", ha="center", color="#d62728", fontsize=9)
ax.set_yticks([]); ax.set_ylim(-0.7, len(t) - 0.3); ax.set_xlim(0, 105)
ax.set_xlabel("a-priori A/B disagreement rate (%)  —  no frontier")
ax.set_title("Self-gating: disagreement rate detects B's competence (green=GO, red=NO-GO)")
ax.legend(loc="lower right"); ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "self_gating.png", dpi=130); plt.close(fig)
print("wrote self_gating.png")

# ---- Figure 3: routing GO per workload (P2.7 / P3 economics) ----
xs2 = [r["disagree_rate"] * 100 for r in d["table"]]
fig, ax = plt.subplots(figsize=(8, 5))
for r in d["table"]:
    col = "#2ca02c" if r["realized_go"] else "#d62728"
    mk = "*" if r["held_out"] else "o"
    ax.scatter(r["disagree_rate"] * 100, (r["gated_cost_mq"] / r["frontier_cost_mq"]), s=150, marker=mk,
               color=col, edgecolors="black", linewidths=0.6, zorder=4)
    ax.annotate(r["workload"], (r["disagree_rate"] * 100, r["gated_cost_mq"] / r["frontier_cost_mq"]),
                textcoords="offset points", xytext=(7, 3), fontsize=8)
ax.axhline(1.0, ls="--", color="#444", lw=1, label="always-frontier cost")
ax.axvline(50, ls=":", color="#1f77b4", lw=1)
ax.set_xlabel("a-priori A/B disagreement rate (%)"); ax.set_ylabel("gated cost / always-frontier cost")
ax.set_title("Routing economics: gated cost vs always-frontier (below 1.0 = cheaper = GO)")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "routing_economics.png", dpi=130); plt.close(fig)
print("wrote routing_economics.png")

# ---- curate existing Pareto plots ----
curate = {
    "router/results/final_pareto.png": "output_router_pareto.png",
    "router/results/final_routing_pareto.png": "model_routing_pareto.png",
    "compressor/results/final_pareto.png": "compressor_pareto.png",
    "advisor/results/banking77_tail.png": "advisor_banking_tail.png",
    "advisor/results/math_tail.png": "advisor_math_tail.png",
    "advisor/results/phase1_calibration.png": "phase1_calibration.png",
}
for src, dst in curate.items():
    p = ROOT / src
    if p.exists():
        shutil.copy(p, FIG / dst)
        print(f"curated {dst}")
print("done")
