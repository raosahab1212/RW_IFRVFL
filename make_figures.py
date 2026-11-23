"""Figures of the manuscript and the extended result figures from results/*_all.csv: python make_figures.py <out_dir>."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import models as M
from experiments import NOISE_DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#cfcec8"
AXIS = "#b9b8b2"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TOL = 1e-3
NEW = "RW-IFRVFL"
BASE = M.BASELINES
MODELS = BASE + [NEW]
NAME = {m: m for m in MODELS} | {"RF": "Random Forest"}
DS = {"adult": "Adult", "australian": "Australian", "breast": "Breast", "diabetes": "Diabetes",
      "heart": "Heart", "horse-colic": "Horse-Colic", "ionosphere": "Ionosphere", "magic": "Magic",
      "mammographic": "Mammographic", "monk-2": "Monk-2", "mushroom": "Mushroom", "phoneme": "Phoneme",
      "pima": "Pima", "wisconsin": "Wisconsin", "halophilic": "Halophilic"}
STYLE = {NEW: dict(color=BLUE, lw=2.0, zorder=5), "IFRVFLC": dict(color=ORANGE, lw=1.6, zorder=4),
         "RVFL": dict(color=AQUA, lw=1.6, zorder=4), "SVM": dict(color=INK2, lw=1.3, ls="--", zorder=3)}
NAMED = [NEW, "IFRVFLC", "RVFL", "SVM"]
DOT = {NEW: BLUE, "IFRVFLC": ORANGE, "RVFL": AQUA, "SVM": INK2}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})


def load(*names):
    return pd.concat([pd.read_csv(os.path.join(R, f"{n}_all.csv")) for n in names], ignore_index=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight",
                metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print("wrote", name)


def repel(ys, gap):
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def grid_y(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def grid_x(ax):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def model_legend(fig, y=1.07, others="other ten baselines"):
    handles = [Line2D([], [], marker="o", ms=3.5, mec="white", mew=0.6, label=NAME[m], **STYLE[m]) for m in NAMED]
    handles.append(Line2D([], [], color=CONTEXT, lw=0.9, label=others))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, y))


def wtl(a, b):
    d = a - b
    return int((d > TOL).sum()), int((d.abs() <= TOL).sum()), int((d < -TOL).sum())


clean = load("benchmark")
clean = clean[clean.model.isin(MODELS)]
label = load("label_noise")
feature = load("feature_noise")


def per_dataset(df, metric="accuracy"):
    return df.groupby(["dataset", "model"])[metric].mean().unstack()[MODELS]


def noise_levels(df, col):
    return df[df.model.isin(MODELS)].groupby([col, "dataset", "model"]).accuracy.mean().unstack()[MODELS]


def fig_noise():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, (title, col, df) in zip(axes, [("Label noise", "label_noise", label),
                                           ("Feature noise", "feat_noise", feature)]):
        lv = noise_levels(df, col).groupby(level=0).mean()
        x = lv.index.values * 100
        for m in lv.columns:
            if m not in STYLE:
                ax.plot(x, lv[m], color=CONTEXT, lw=0.9, zorder=1)
        for m in NAMED:
            ax.plot(x, lv[m], marker="o", ms=3.5, mec="white", mew=0.6, **STYLE[m])
        ys = repel([lv[m].iloc[-1] for m in NAMED], gap=(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.055)
        for m, yv in zip(NAMED, ys):
            ax.annotate(NAME[m], (x[-1], lv[m].iloc[-1]), xytext=(x[-1] + 1.6, yv), textcoords="data",
                        va="center", fontsize=7.5, color=INK, annotation_clip=False)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.0f}%" for v in x])
        ax.set_xlim(x[0] - 1.5, x[-1] + 1.5)
        ax.set_xlabel("Noise level")
        grid_y(ax)
    axes[0].set_ylabel("Mean accuracy (5 datasets)")
    model_legend(fig)
    fig.tight_layout(w_pad=6)
    save(fig, "fig_noise")


def fig_cd(metric, name):
    t = per_dataset(clean, metric)
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
        ax.plot([r, r, xl + (0.1 if left else -0.1)], [y0, yl, yl], color=BLUE if m == NEW else INK,
                lw=1.4 if m == NEW else 0.8)
        ax.text(xl, yl, f"{NAME[m]} ({r:.2f})", ha="right" if left else "left", va="center",
                fontsize=7.5, color=INK, fontweight="bold" if m == NEW else "normal")
    groups, rv = [], ranks.values
    for i in range(k):
        j = max(j for j in range(i, k) if rv[j] - rv[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    for gi, (a, b) in enumerate(groups):
        yg = y0 - 0.035 - gi * 0.03
        ax.plot([rv[a] - 0.05, rv[b] + 0.05], [yg, yg], color=INK, lw=2.2, solid_capstyle="round")
    save(fig, name)
    print(f"   {metric}: CD = {cd:.3f}, groups", [(names[a], names[b]) for a, b in groups])


def fig_sens():
    panels = [("alpha", "Trust $\\alpha$"), ("k", "Neighbourhood size $k$"), ("sigma_scale", "Spread scale $s$"),
              ("n_hidden", "Hidden neurons $L$"), ("lam", "Regularisation $\\lambda$"), ("seed", "Random seed")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    rows = []
    for setting, exp, color in [("clean, 14 datasets", "sensitivity", BLUE),
                                ("30% label noise, 5 datasets", "sensitivity_label", ORANGE)]:
        per = load(exp).groupby(["model", "dataset"]).accuracy.mean().groupby(level=0).mean()
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
            grid_y(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean accuracy")
    axes.flat[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.text(0.01, -0.02, "Open circle: default value (L = 100, lambda = 0.1, k = 9, s = 1.5, alpha = 0.75); "
             "seeds 0-4 at the default settings.", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "fig_sens")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)


def fig_ranks():
    t = per_dataset(clean)
    rk = t.rank(axis=1, ascending=False)
    order = list(rk.mean().sort_values().index)
    rows = sorted(t.index, key=lambda d: DS[d])
    data = rk.loc[rows, order].values
    cmap = LinearSegmentedColormap.from_list("rank", RAMP[::-1])
    norm = Normalize(1, len(order))
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.pcolormesh(data, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r, g, b = to_rgb(cmap(norm(data[i, j])))
            light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45
            v = data[i, j]
            ax.text(j + 0.5, i + 0.5, f"{v:g}", ha="center", va="center", fontsize=6.5,
                    color=INK if light else "white")
    ax.set_xticks(np.arange(len(order)) + 0.5)
    ax.set_xticklabels([f"{NAME[m]} ({rk[m].mean():.2f})" for m in order], rotation=40, ha="right")
    for lbl, m in zip(ax.get_xticklabels(), order):
        if m == NEW:
            lbl.set_fontweight("bold")
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([DS[d] for d in rows])
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Accuracy rank on each benchmark dataset (1 = best; average rank in brackets)",
                 loc="left", fontsize=9, color=INK)
    save(fig, "fig_ranks")


def fig_noise_datasets():
    fig, axes = plt.subplots(2, len(NOISE_DATASETS), figsize=(7.2, 3.9), sharex=True)
    for r, (title, col, df) in enumerate([("Label noise", "label_noise", label),
                                          ("Feature noise", "feat_noise", feature)]):
        g = noise_levels(df, col)
        for c, ds in enumerate(NOISE_DATASETS):
            ax = axes[r, c]
            sub = g.xs(ds, level="dataset")
            x = sub.index.values * 100
            for m in MODELS:
                if m not in STYLE:
                    ax.plot(x, sub[m], color=CONTEXT, lw=0.8, zorder=1)
            for m in NAMED:
                st = dict(STYLE[m], lw=STYLE[m]["lw"] * 0.75)
                ax.plot(x, sub[m], marker="o", ms=2.6, mec="white", mew=0.5, **st)
            if r == 0:
                ax.set_title(DS[ds], loc="left", fontsize=8.5, color=INK)
            ax.set_xticks(x)
            ticks = ax.set_xticklabels([f"{v:.0f}" for v in x], fontsize=6.5)
            if len(ticks) > 1:
                ticks[0].set_ha("right")
                ticks[1].set_ha("left")
            ax.tick_params(axis="y", labelsize=6.5)
            grid_y(ax)
        axes[r, 0].set_ylabel(f"{title}\naccuracy")
    for ax in axes[1]:
        ax.set_xlabel("Noise level (%)")
    model_legend(fig, y=1.05)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, "fig_noise_datasets")


def dot_panels(rows, panels, title, name, colors, height):
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, height), sharey=True)
    y = np.arange(len(rows))
    for ax, (ptitle, values) in zip(axes, panels):
        for yi, (key, _) in zip(y, rows):
            v = values[key]
            c = colors.get(key, MUTED)
            ax.plot([v], [yi], marker="o", ms=5.5 if key == NEW else 4.5, color=c, mec="white", mew=0.6,
                    zorder=4 if key == NEW else 3)
        if "RVFL" in values.index:
            ax.axvline(values["RVFL"], color=AQUA, lw=0.8, ls=(0, (3, 2)), zorder=2)
        lo, hi = values[[k for k, _ in rows]].min(), values[[k for k, _ in rows]].max()
        pad = max((hi - lo) * 0.12, 0.002)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(ptitle, loc="left", fontsize=8.5, color=INK)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=2, steps=[1, 2, 5, 10]))
        grid_x(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lbl for _, lbl in rows])
    for lbl, (key, _) in zip(axes[0].get_yticklabels(), rows):
        if key == NEW:
            lbl.set_fontweight("bold")
    axes[0].set_ylim(len(rows) - 0.4, -0.6)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(w_pad=1.0)
    save(fig, name)


def fig_ablation():
    rows = [("RVFL", "$w=1$ (plain RVFL)"), ("abl:centroid", "$w=g$"), ("abl:neighbour", "$w=1-\\nu$"),
            ("abl:product_unscaled", "$w=g(1-\\nu)$"), ("abl:score_unscaled", "$w=(1+g-\\nu)/2$"),
            ("abl:product", "$w=\\mu(1-\\nu)$"), ("abl:score", "$w=(1+\\mu-\\nu)/2$ (Revision 1)"),
            ("abl:ha", "Ha score of $(\\mu,\\nu)$"), ("abl:shuffled", "Revision-1 weights, shuffled"),
            ("IFRVFLC", "IFRVFLC"), (NEW, "RW-IFRVFL (signed score)")]
    keys = [k for k, _ in rows]
    c = load("benchmark", "ablation")
    c = c[c.model.isin(keys)].groupby(["dataset", "model"]).accuracy.mean().unstack().mean()
    lab = load("label_noise", "ablation_label")
    lab = lab[lab.model.isin(keys)].groupby(["label_noise", "dataset", "model"]).accuracy.mean().unstack().mean()
    fea = load("feature_noise", "ablation_feature")
    fea = fea[fea.model.isin(keys)].groupby(["feat_noise", "dataset", "model"]).accuracy.mean().unstack().mean()
    dot_panels(rows, [("Clean, 14 datasets", c), ("Label noise, 25 pairs", lab), ("Feature noise, 25 pairs", fea)],
               "Ablation of the sample weighting: mean accuracy (dashed line: plain RVFL)", "fig_ablation",
               {NEW: BLUE, "IFRVFLC": ORANGE, "RVFL": AQUA}, 3.3)


def fig_components():
    rows = [(NEW, "RW-IFRVFL (full)"), ("comp:alpha_half", "trust fixed at $\\alpha=1/2$"),
            ("comp:small_k", "$k\\leq 15$ instead of $k\\leq 50$"),
            ("comp:balance", "weight $1-\\tilde{\\pi}$, balance target"),
            ("comp:discounted", "non-membership discounted by $g$")]
    keys = [k for k, _ in rows]
    c = load("benchmark", "ablation").groupby(["dataset", "model"]).accuracy.mean().unstack()[keys]
    lab = load("label_noise", "ablation_label").groupby(["label_noise", "dataset", "model"]).accuracy.mean().unstack()[keys]
    fea = load("feature_noise", "ablation_feature").groupby(["feat_noise", "dataset", "model"]).accuracy.mean().unstack()[keys]
    case = load("case_study").groupby("model").accuracy.mean()[keys]
    high = lab[lab.index.get_level_values(0) >= 0.3]
    dot_panels(rows, [("Clean", c.mean()), ("Label noise", lab.mean()), ("Label 30-40%", high.mean()),
                      ("Feature noise", fea.mean()), ("Halophilic", case)],
               "Components of RW-IFRVFL, each changed in turn: mean accuracy", "fig_components", {NEW: BLUE}, 1.9)


def fig_time():
    acc = per_dataset(clean).mean()
    ft = clean.groupby(["dataset", "model"]).fit_s.median().unstack()[MODELS]
    gm = np.exp(np.log(ft.clip(lower=1e-4)).mean())
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in MODELS:
        ax.scatter(gm[m], acc[m], s=42 if m in DOT else 30, color=DOT.get(m, MUTED), edgecolors="white",
                   linewidths=0.8, zorder=4 if m in DOT else 3)
    offsets = {m: (5, 0, "left") for m in MODELS}
    offsets.update({"LightGBM": (5, 4, "left"), "RF": (5, -4, "left"), "CatBoost": (-5, 0, "right"),
                    "XGBoost": (5, -1, "left"), "IFRVFLC": (5, -5, "left"), "RVFL": (-5, 2, "right"),
                    NEW: (5, 5, "left"), "NF-RVFL": (-5, -3, "right"), "SVM": (5, 3, "left")})
    for m in MODELS:
        dx, dy, ha = offsets[m]
        ax.annotate(NAME[m], (gm[m], acc[m]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=7, color=INK if m in DOT else INK2, fontweight="bold" if m == NEW else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("Training time per fit, s (geometric mean over 14 datasets of the fold median)")
    ax.set_ylabel("Mean accuracy (14 datasets)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig_time")


def fig_case_study():
    cs = load("case_study")
    cs = cs[cs.model.isin(MODELS)]
    t = cs.groupby("model")[["accuracy", "f1_macro", "auc"]].agg(["mean", "std"])
    order = list(t[("accuracy", "mean")].sort_values(ascending=False).index)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True)
    y = np.arange(len(order))
    for ax, (met, title) in zip(axes, [("accuracy", "Accuracy"), ("f1_macro", "Macro-F1"), ("auc", "AUC")]):
        for yi, m in zip(y, order):
            mu, sd = t.loc[m, (met, "mean")], t.loc[m, (met, "std")]
            c = DOT.get(m, MUTED)
            ax.plot([mu - sd, mu + sd], [yi, yi], color=c, lw=1.0, alpha=0.55, solid_capstyle="round", zorder=2)
            ax.plot([mu], [yi], marker="o", ms=5.5 if m == NEW else 4.5, color=c, mec="white", mew=0.6, zorder=3)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.tick_params(axis="x", labelsize=6.5)
        grid_x(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([NAME[m] for m in order])
    for lbl, m in zip(axes[0].get_yticklabels(), order):
        if m == NEW:
            lbl.set_fontweight("bold")
    axes[0].invert_yaxis()
    fig.suptitle("Halophilic protein case study: mean and one standard deviation over the 10 outer folds",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_case_study")


def fig_wtl():
    settings = [("Clean, 14 datasets", per_dataset(clean)),
                ("Label noise, 25 pairs", noise_levels(label, "label_noise")),
                ("Feature noise, 25 pairs", noise_levels(feature, "feat_noise"))]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    y = np.arange(len(BASE))
    for ax, (title, t) in zip(axes, settings):
        n = len(t)
        min_label = max(2, int(np.ceil(0.1 * n)))
        for yi, b in zip(y, BASE):
            w, ti, lo = wtl(t[NEW], t[b])
            segs = [(-ti / 2 - lo, lo, RED, "white"), (-ti / 2, ti, CONTEXT, INK), (ti / 2, w, BLUE, "white")]
            for left, width, color, txt in segs:
                if width == 0:
                    continue
                ax.barh(yi, width, left=left, height=0.7, color=color, edgecolor="white", linewidth=1.0)
                if width >= min_label:
                    ax.text(left + width / 2, yi, str(width), ha="center", va="center", fontsize=6.5, color=txt)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
        ax.set_xlim(-n, n)
        ax.set_xticks([-n, -(n // 2), 0, n // 2, n])
        ax.set_xticklabels([str(abs(v)) for v in ax.get_xticks()], fontsize=6.5)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([NAME[b] for b in BASE])
    axes[0].invert_yaxis()
    handles = [Patch(color=BLUE, label="RW-IFRVFL more accurate"), Patch(color=CONTEXT, label="tie (within 0.001)"),
               Patch(color=RED, label="baseline more accurate")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_wtl")


fig_noise()
fig_cd("accuracy", "fig_cd")
fig_cd("f1_macro", "fig_cd_f1")
fig_cd("auc", "fig_cd_auc")
fig_sens()
fig_ranks()
fig_noise_datasets()
fig_ablation()
fig_components()
fig_time()
fig_case_study()
fig_wtl()
