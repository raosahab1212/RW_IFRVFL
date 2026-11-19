"""
Summary figure from the merged result files (results/<exp>_summary.csv).

    python make_figures.py   -> figures/summary.png (+ .pdf)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Reference palette, light mode (validated slot order: blue, orange, aqua, yellow)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b2", "#e6e5e0"
SERIES = {"RW-IFRVFL": "#2a78d6", "RVFL": "#eb6834", "IFRVFLC": "#1baf7a", "SVM": "#eda100"}
HIGHLIGHT, OTHER = SERIES["RW-IFRVFL"], "#c9c8c2"
NAMES = {"abl:fuzzy": "Fuzzy weight only", "abl:nonmem": "Neighbour weight only",
         "abl:pw_unscaled": "Unscaled μ, product w", "abl:sf_unscaled": "Unscaled μ, score w",
         "abl:pw_scaled": "Scaled μ, product w", "abl:rw_ha": "Scaled μ, Ha score",
         "abl:rw_shuffled": "Shuffled RW weights", "RVFL": "RVFL (no weights)",
         "IFRVFLC": "IFRVFLC", "RW-IFRVFL": "RW-IFRVFL (proposed)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def load(exp):
    p = os.path.join(R, f"{exp}_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def hbar(ax, values, title, xlabel, ref=None, fmt="{:.2f}", lower_better=False, dots=False):
    """Bars when the axis starts at zero; dots when the axis is zoomed (never truncated bars)."""
    v = values.sort_values(ascending=not lower_better)
    colors = [HIGHLIGHT if k.startswith("RW-IFRVFL") else OTHER for k in v.index]
    if dots:
        ax.scatter(v.values, range(len(v)), s=70, c=colors, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    else:
        ax.barh(range(len(v)), v.values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(v)), v.index)
    for i, (k, x) in enumerate(v.items()):
        bold = k.startswith("RW-IFRVFL")
        ax.text(x, i, ("   " if dots else " ") + fmt.format(x), va="center", ha="left", fontsize=8,
                color=INK if bold else INK2, fontweight="bold" if bold else "normal")
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


e1, e3, e4, e7 = load("e1"), load("e3"), load("e4_label"), load("e7")
fig, axes = plt.subplots(2, 2, figsize=(12, 9.2), constrained_layout=True)

# (a) E1 mean rank (accuracy) over 14 datasets, lower is better
t = e1.pivot(index="dataset", columns="model", values="accuracy")
rank = t.rank(axis=1, ascending=False).mean()
hbar(axes[0, 0], rank, "(a) Benchmark: mean accuracy rank over 14 datasets (1 = best)",
     "Mean rank across datasets (shorter is better)", lower_better=True)
axes[0, 0].set_xlim(0, 14)

# (b) E4 label noise: mean accuracy over 5 datasets vs noise level
ax = axes[0, 1]
noise_ds = ["australian", "wisconsin", "heart", "ionosphere", "magic"]
clean = e1[e1.dataset.isin(noise_ds)].assign(label_noise=0.0)
p = pd.concat([clean, e4]).groupby(["model", "label_noise"]).accuracy.mean().unstack(0)
for m, c in SERIES.items():
    y = p[m]
    ax.plot(y.index * 100, y.values, color=c, lw=2, marker="o", ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=m, zorder=3)
# direct end labels, nudged apart so equal end values do not collide
ends = sorted(((p[m].values[-1], m) for m in SERIES), reverse=True)
span = p[list(SERIES)].values.max() - p[list(SERIES)].values.min()
placed = []
for val, m in ends:
    yv = val if not placed else min(val, placed[-1] - 0.045 * span)
    placed.append(yv)
    ax.annotate(m, (40, val), xytext=(46, yv), textcoords="data", va="center", fontsize=8,
                color=INK, fontweight="bold" if m == "RW-IFRVFL" else "normal",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=4))
others = [m for m in p.columns if m not in SERIES]
for m in others:
    ax.plot(p.index * 100, p[m].values, color=GRID, lw=1, zorder=1)
ax.plot([], [], color=GRID, lw=1, label="other 10 models")
ax.set_title("(b) Label noise: mean accuracy over 5 datasets")
ax.set_xlabel("Training labels flipped (%)")
ax.set_ylabel("Test accuracy (clean labels)")
ax.set_xlim(-1, 58)
ax.set_xticks([0, 5, 10, 20, 30, 40])
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left", fontsize=8)

# (c) E3 ablation: mean accuracy over 14 datasets per weighting variant
ab = pd.concat([e3, e1[e1.model.isin(["RVFL", "IFRVFLC", "RW-IFRVFL"])]])
acc = ab.groupby("model").accuracy.mean().rename(NAMES)
hbar(axes[1, 0], acc, "(c) Ablation: mean accuracy over 14 datasets",
     "Mean accuracy (dashed = unweighted RVFL)", ref=acc["RVFL (no weights)"], fmt="{:.4f}",
     dots=True)
axes[1, 0].set_xlim(acc.min() - 0.004, acc.max() + 0.006)

# (d) E7 halophilic case study
ax = axes[1, 1]
if e7 is not None:
    s = e7.set_index("model").accuracy
    chance = s[s <= 0.51]                       # models stuck at chance level, noted instead
    s = s.drop(chance.index)
    hbar(ax, s, "(d) Halophilic proteins (Zhang & Ge 2013): 10-fold accuracy",
         "Accuracy" + ("".join(f"   ({m}: {v:.3f}, chance level, not shown)" for m, v in chance.items())),
         fmt="{:.3f}", dots=True)
    ax.set_xlim(s.min() - 0.02, min(1.0, s.max() + 0.02))
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "E7 results pending", ha="center", va="center", color=INK2)

fig.suptitle("RW-IFRVFL rerun: nested 10-fold CV, 30-configuration tuning per model",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"summary.{ext}"), dpi=200)
print("wrote", os.path.join(OUT, "summary.png"))
