"""
Figures for Sections 5-6 of the manuscript, from the result files.

    python paper_figures.py <out_dir>

fig_noise.pdf  mean accuracy against label- and feature-noise level
fig_cd.pdf     critical-difference diagram of the clean-benchmark accuracy ranks
fig_sens.pdf   one-at-a-time sensitivity of RW-IFRVFL (needs the sens_* results)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "paper")
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID, CONTEXT = "#0b0b0b", "#52514e", "#b9b8b2", "#e6e5e0", "#cfcec8"
# first three slots of the reference categorical palette (validated for all pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEW = "RW-IFRVFL3"
BASE = ["SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "NB", "KNN", "NF-RVFL", "ANFIS",
        "GBTSVM", "TRKM", "RVFL", "IFRVFLC"]
NAME = {m: m for m in BASE} | {NEW: "RW-IFRVFL", "RF": "Random Forest"}
STYLE = {NEW: dict(color=BLUE, lw=2.0, zorder=5), "IFRVFLC": dict(color=ORANGE, lw=1.6, zorder=4),
         "RVFL": dict(color=AQUA, lw=1.6, zorder=4), "SVM": dict(color=INK2, lw=1.3, ls="--", zorder=3)}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42})


def load(*names):
    return pd.concat([pd.read_csv(os.path.join(R, f"{n}_all.csv")) for n in names], ignore_index=True)


def repel(ys, gap):
    """Spread label positions so neighbours are at least `gap` apart (keeps order)."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ noise curves
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, (title, col, exps) in zip(axes, [("Label noise", "label_noise", ("e4_label", "v3_label")),
                                          ("Feature noise", "feat_noise", ("e4_feature", "v3_feature"))]):
    df = load(*exps)
    lv = df[df.model.isin(BASE + [NEW])].groupby([col, "dataset", "model"]).accuracy.mean().groupby(level=[0, 2]).mean().unstack()
    x = lv.index.values * 100
    for m in lv.columns:
        if m not in STYLE:
            ax.plot(x, lv[m], color=CONTEXT, lw=0.9, zorder=1)
    named = [NEW, "IFRVFLC", "RVFL", "SVM"]
    for m in named:
        ax.plot(x, lv[m], marker="o", ms=3.5, mec="white", mew=0.6, **STYLE[m])
    ys = repel([lv[m].iloc[-1] for m in named], gap=(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.055)
    for m, yv in zip(named, ys):
        ax.annotate(NAME[m], (x[-1], lv[m].iloc[-1]), xytext=(x[-1] + 1.6, yv), textcoords="data",
                    va="center", fontsize=7.5, color=INK, annotation_clip=False)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}%" for v in x])
    ax.set_xlim(x[0] - 1.5, x[-1] + 1.5)
    ax.set_xlabel("Noise level")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Mean accuracy (5 datasets)")
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ms=3.5, mec="white", mew=0.6, label=NAME[m], **STYLE[m])
           for m in [NEW, "IFRVFLC", "RVFL", "SVM"]] + [Line2D([], [], color=CONTEXT, lw=0.9, label="other ten baselines")]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=6)
save(fig, "fig_noise")

# ------------------------------------------------------------------ critical-difference diagram
clean = load("e1", "v3_e1")
clean = clean[clean.model.isin(BASE + [NEW])]
t = clean.groupby(["dataset", "model"]).accuracy.mean().unstack()
ranks = t.rank(axis=1, ascending=False).mean().sort_values()
k, n = len(ranks), len(t)
cd = 3.354 * np.sqrt(k * (k + 1) / (6 * n))
fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.set_xlim(1, k)
ax.set_ylim(0, 1)
ax.axis("off")
y0 = 0.80
ax.plot([1, k], [y0, y0], color=INK, lw=0.8)
for r in range(1, k + 1):
    ax.plot([r, r], [y0, y0 + 0.03], color=INK, lw=0.8)
    ax.text(r, y0 + 0.06, str(r), ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.plot([1, 1 + cd], [y0 + 0.17, y0 + 0.17], color=INK, lw=1.2)
for xv in (1, 1 + cd):
    ax.plot([xv, xv], [y0 + 0.15, y0 + 0.19], color=INK, lw=1.2)
ax.text(1 + cd / 2, y0 + 0.21, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
names = list(ranks.index)
half = (k + 1) // 2
for i, m in enumerate(names):
    r = ranks[m]
    left = i < half
    row = i if left else k - 1 - i
    yl = y0 - 0.09 - row * 0.085
    xl = 0.75 if left else k + 0.25
    c = BLUE if m == NEW else INK
    w = "bold" if m == NEW else "normal"
    ax.plot([r, r, xl + (0.1 if left else -0.1)], [y0, yl, yl], color=c, lw=0.8)
    ax.text(xl, yl, f"{NAME[m]} ({r:.2f})", ha="right" if left else "left", va="center",
            fontsize=7.5, color=c, fontweight=w)
# groups whose rank range is below CD (maximal cliques)
groups, rv = [], ranks.values
for i in range(k):
    j = max(j for j in range(i, k) if rv[j] - rv[i] < cd)
    if j > i and not any(a <= i and j <= b for a, b in groups):
        groups.append((i, j))
for gi, (a, b) in enumerate(groups):
    yg = y0 - 0.035 - gi * 0.03
    ax.plot([rv[a] - 0.05, rv[b] + 0.05], [yg, yg], color=INK, lw=2.2, solid_capstyle="round")
save(fig, "fig_cd")
print(f"CD diagram: CD = {cd:.3f}, {len(groups)} groups:",
      [(NAME[names[a]], NAME[names[b]]) for a, b in groups])

# ------------------------------------------------------------------ sensitivity
if all(os.path.exists(os.path.join(R, f"{e}_all.csv")) for e in ("sens_e1", "sens_label")):
    import models as M
    panels = [("alpha", "Trust $\\alpha$"), ("k", "Neighbourhood size $k$"), ("sigma_scale", "Spread scale $s$"),
              ("n_hidden", "Hidden neurons $L$"), ("lam", "Regularisation $\\lambda$"), ("seed", "Random seed")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey=False)
    rows = []
    for setting, exp, color in [("clean, 14 datasets", "sens_e1", BLUE), ("30% label noise, 5 datasets", "sens_label", ORANGE)]:
        df = load(exp)
        per = df.groupby(["model", "dataset"]).accuracy.mean().groupby(level=0).mean()
        for ax, (p, title) in zip(axes.flat, panels):
            vals = M.SENS_SWEEPS[p]
            ys = []
            for v in vals:
                name = "sens:default" if (p != "seed" and v == M.SENS_DEFAULT[p]) else f"sens:{p}={v:g}"
                ys.append(per[name])
                rows.append(dict(setting=setting, param=p, value=v, accuracy=per[name]))
            xs = np.arange(len(vals))
            ax.plot(xs, ys, marker="o", ms=3.5, mec="white", mew=0.6, color=color, lw=1.6, label=setting)
            if p != "seed":
                d = vals.index(M.SENS_DEFAULT[p])
                ax.plot([d], [ys[d]], marker="o", ms=6, mfc="none", mec=INK2, mew=0.8, zorder=6)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=7)
            ax.set_title(title, loc="left", fontsize=8.5, color=INK)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean accuracy")
    axes.flat[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.text(0.01, -0.02, "Open circle: default value (L = 100, lambda = 0.1, k = 9, s = 1.5, alpha = 0.75); "
             "seeds 0-4 at the default settings.", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "fig_sens")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
else:
    print("sensitivity results not available yet")
