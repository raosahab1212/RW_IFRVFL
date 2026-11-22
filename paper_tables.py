"""
LaTeX tables and figures for Sections 5-6 of the manuscript, from the result files.

    python paper_tables.py <out_dir>      (reads results/*_all.csv)

RW-IFRVFL in the paper is the revised method (model RW-IFRVFL3 in the harness); the
Revision-1 formulation (model RW-IFRVFL) appears only in the ablation.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "paper")
os.makedirs(OUT, exist_ok=True)
TOL = 1e-3
NEW = "RW-IFRVFL3"
BASE = ["SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "NB", "KNN", "NF-RVFL", "ANFIS",
        "GBTSVM", "TRKM", "RVFL", "IFRVFLC"]
MODELS = BASE + [NEW]
CITE = {"SVM": "SVM~\\cite{hearst1998}", "RF": "RF~\\cite{breiman2001}", "XGBoost": "XGBoost~\\cite{chen2016}",
        "LightGBM": "LightGBM~\\cite{ke2017}", "CatBoost": "CatBoost~\\cite{prokhorenkova2018}",
        "NB": "NB~\\cite{zhang2004}", "KNN": "KNN~\\cite{cunningham2021}", "NF-RVFL": "NF-RVFL~\\cite{Sajid2024}",
        "ANFIS": "ANFIS~\\cite{Jang1993}", "GBTSVM": "GBTSVM~\\cite{quadir2025}", "TRKM": "TRKM~\\cite{quadir2025trkm}",
        "RVFL": "RVFL~\\cite{Pao1992}", "IFRVFLC": "IFRVFLC~\\cite{Mishra2023}", NEW: "\\textbf{RW-IFRVFL}"}
PLAIN = {m: m for m in MODELS} | {"RF": "Random Forest", NEW: "RW-IFRVFL"}
DS = {"adult": "Adult", "australian": "Australian", "breast": "Breast", "diabetes": "Diabetes",
      "heart": "Heart", "horse-colic": "Horse-Colic", "ionosphere": "Ionosphere", "magic": "Magic",
      "mammographic": "Mammographic", "monk-2": "Monk-2", "mushroom": "Mushroom", "phoneme": "Phoneme",
      "pima": "Pima", "wisconsin": "Wisconsin", "halophilic": "Halophilic"}
LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40]


def load(*names):
    return pd.concat([pd.read_csv(os.path.join(R, f"{n}_all.csv")) for n in names], ignore_index=True)


def write(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", name)


def holm(p):
    """Holm-adjusted p-values (same order as given)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wtl(a, b):
    d = a - b
    return int((d > TOL).sum()), int((d.abs() <= TOL).sum()), int((d < -TOL).sum())


def wilc(a, b):
    try:
        return wilcoxon(a, b).pvalue
    except ValueError:
        return 1.0


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def per_dataset(df, metric, agg="mean"):
    t = df.groupby(["dataset", "model"])[metric].agg(agg).unstack()
    return t[[m for m in MODELS if m in t.columns]]


# ------------------------------------------------------------------ benchmark tables
clean = load("e1", "v3_e1")
clean = clean[clean.model.isin(MODELS)]


def benchmark_table(metric, label, caption, lower_better=False, digits=4):
    t = per_dataset(clean, metric, "median" if lower_better else "mean")
    ranks = t.rank(axis=1, ascending=lower_better)
    rows = []
    for ds in sorted(t.index, key=lambda d: DS[d]):
        r = t.loc[ds]
        best = r.min() if lower_better else r.max()
        cells = []
        for m in t.columns:
            v = r[m]
            s = (f"{v:.{digits}f}" if not lower_better else (f"{v:.3g}" if v >= 0.001 else "$<$0.001"))
            cells.append(f"\\textbf{{{s}}}" if abs(v - best) <= (1e-4 if not lower_better else 1e-9) else s)
        rows.append(f"{DS[ds]} & " + " & ".join(cells) + " \\\\")
    if not lower_better:
        avg = t.mean()
        rows.append("\\hline\nAverage & " + " & ".join(
            (f"\\textbf{{{avg[m]:.{digits}f}}}" if avg[m] == avg.max() else f"{avg[m]:.{digits}f}") for m in t.columns) + " \\\\")
    rk = ranks.mean()
    rows.append(("\\hline\n" if lower_better else "") + "Average rank & " + " & ".join(
        (f"\\textbf{{{rk[m]:.2f}}}" if rk[m] == rk.min() else f"{rk[m]:.2f}") for m in t.columns) + " \\\\")
    arrow = "$\\downarrow$" if lower_better else "$\\uparrow$"
    head = " & ".join(f"\\rotatebox{{90}}{{{CITE[m]}}}" for m in t.columns)
    write(f"tab_{label}.tex", f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{{caption} ({arrow}). Mean over the 10 outer test folds; best value per dataset in bold.}}
\\label{{tab:{label}}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l{'c' * len(t.columns)}}}
\\hline
\\textbf{{Dataset}} & {head} \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}}}
\\end{{table*}}
""")
    return t, ranks


acc, acc_r = benchmark_table("accuracy", "accuracy", "Accuracy of all models on the 14 benchmark datasets")
f1, f1_r = benchmark_table("f1_macro", "f1", "Macro-averaged F1-score of all models on the 14 benchmark datasets")
auc, auc_r = benchmark_table("auc", "auc", "AUC of all models on the 14 benchmark datasets")
_ = benchmark_table("fit_s", "execution_time",
                    "Training time in seconds (median final fit over the 10 folds, neighbourhood search included)",
                    lower_better=True)

# ------------------------------------------------------------------ statistics (benchmark)
stat_rows, summary = [], {}
for metric, t, rk in [("accuracy", acc, acc_r), ("f1_macro", f1, f1_r), ("auc", auc, auc_r)]:
    fr = friedmanchisquare(*[t[m] for m in t.columns])
    ps = [wilc(t[NEW], t[b]) for b in BASE]
    summary[metric] = dict(friedman_p=fr.pvalue, ranks=rk.mean(), holm=dict(zip(BASE, holm(ps))),
                           wtl={b: wtl(t[NEW], t[b]) for b in BASE})
k, n = len(MODELS), len(acc)
cd = 3.354 * np.sqrt(k * (k + 1) / (6 * n))
for b in BASE:
    cells = []
    for metric in ["accuracy", "f1_macro", "auc"]:
        s = summary[metric]
        w, ti, l = s["wtl"][b]
        p = s["holm"][b]
        diff = s["ranks"][b] - s["ranks"][NEW]
        sig = "$^\\dagger$" if abs(diff) > cd else ""
        cells.append(f"{w}/{ti}/{l} & {fmt_p(p)}{'*' if p < 0.05 else ''} & {diff:+.2f}{sig}")
    stat_rows.append(f"{PLAIN[b]} & " + " & ".join(cells) + " \\\\")
write("tab_wilcoxon.tex", f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{RW-IFRVFL against each baseline on the 14 datasets: wins/ties/losses, Holm-corrected Wilcoxon $p$-value ($^*$: $p<0.05$) and the difference in average Friedman rank (baseline minus RW-IFRVFL; positive means RW-IFRVFL ranks better; $^\\dagger$: larger than the Nemenyi critical difference {cd:.2f}).}}
\\label{{tab:wilcoxon}}
\\small
\\begin{{tabular}}{{l ccc ccc ccc}}
\\hline
 & \\multicolumn{{3}}{{c}}{{Accuracy}} & \\multicolumn{{3}}{{c}}{{Macro-F1}} & \\multicolumn{{3}}{{c}}{{AUC}} \\\\
Baseline & W/T/L & $p_{{\\mathrm{{Holm}}}}$ & $\\Delta$rank & W/T/L & $p_{{\\mathrm{{Holm}}}}$ & $\\Delta$rank & W/T/L & $p_{{\\mathrm{{Holm}}}}$ & $\\Delta$rank \\\\
\\hline
{chr(10).join(stat_rows)}
\\hline
\\end{{tabular}}
\\end{{table*}}
""")

# ------------------------------------------------------------------ noise tables
def noise_table(kind, col, base_exp, v3_exp):
    df = load(base_exp, v3_exp)
    df = df[df.model.isin(MODELS)]
    g = df.groupby([col, "dataset", "model"]).accuracy.mean().unstack()[MODELS]
    lv = g.groupby(level=0).mean()
    rk = g.rank(axis=1, ascending=False).mean()
    hi = g[g.index.get_level_values(0) >= 0.3]
    p_all = holm([wilc(g[NEW], g[b]) for b in BASE])
    p_hi = holm([wilc(hi[NEW], hi[b]) for b in BASE])
    rows = []
    for m in sorted(MODELS, key=lambda m: rk[m]):
        cells = []
        for L in LEVELS:
            v = lv.loc[L, m]
            cells.append(f"\\textbf{{{v:.3f}}}" if abs(v - lv.loc[L].max()) < 1e-9 else f"{v:.3f}")
        if m == NEW:
            extra = "--- & --- & --- & ---"
        else:
            i = BASE.index(m)
            w, ti, l = wtl(g[NEW], g[m])
            w2, t2, l2 = wtl(hi[NEW], hi[m])
            extra = f"{w}/{ti}/{l} & {fmt_p(p_all[i])}{'*' if p_all[i] < 0.05 else ''} & {w2}/{t2}/{l2} & {fmt_p(p_hi[i])}{'*' if p_hi[i] < 0.05 else ''}"
        name = "\\textbf{RW-IFRVFL}" if m == NEW else PLAIN[m]
        rows.append(f"{name} & " + " & ".join(cells) + f" & {rk[m]:.2f} & {extra} \\\\")
    write(f"tab_{kind}_noise.tex", f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Accuracy ($\\uparrow$) under {kind} noise: mean over the five noise datasets at each level (best in bold), average rank over all 25 dataset--level pairs, and RW-IFRVFL against each model over all pairs and over the 30--40\\% pairs (wins/ties/losses, Holm-corrected Wilcoxon $p$; $^*$: $p<0.05$). Rows are sorted by average rank.}}
\\label{{tab:{kind}_noise}}
\\small
\\begin{{tabular}}{{l ccccc c cc cc}}
\\hline
 & \\multicolumn{{5}}{{c}}{{Noise level}} & & \\multicolumn{{2}}{{c}}{{All levels}} & \\multicolumn{{2}}{{c}}{{30--40\\%}} \\\\
Model & 5\\% & 10\\% & 20\\% & 30\\% & 40\\% & Rank & W/T/L & $p_{{\\mathrm{{Holm}}}}$ & W/T/L & $p_{{\\mathrm{{Holm}}}}$ \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\end{{table*}}
""")
    return g, lv


label_g, label_lv = noise_table("label", "label_noise", "e4_label", "v3_label")
feat_g, feat_lv = noise_table("feature", "feat_noise", "e4_feature", "v3_feature")

# ------------------------------------------------------------------ ablation: weighting variants
VARIANTS = [("RVFL", "$w=1$ (plain RVFL)"), ("abl:fuzzy", "$w=g$"), ("abl:nonmem", "$w=1-\\nu$"),
            ("abl:pw_unscaled", "$w=g(1-\\nu)$"), ("abl:sf_unscaled", "$w=(1+g-\\nu)/2$"),
            ("abl:pw_scaled", "$w=\\mu(1-\\nu)$"), ("RW-IFRVFL", "$w=(1+\\mu-\\nu)/2$ (Revision 1)"),
            ("abl:rw_ha", "$w=$ Ha score of $(\\mu,\\nu)$"), ("abl:rw_shuffled", "Revision-1 weights, shuffled"),
            ("IFRVFLC", "IFRVFLC~\\cite{Mishra2023}"), (NEW, "\\textbf{RW-IFRVFL (signed score)}")]
names = [v for v, _ in VARIANTS]
ac = load("e1", "e3", "v3_e1"); ac = ac[ac.model.isin(names)]
al = load("e4_label", "e4_label_abl", "v3_label"); al = al[al.model.isin(names)]
af = load("e4_feature", "e4_feature_abl", "v3_feature"); af = af[af.model.isin(names)]
tc = ac.groupby(["dataset", "model"]).accuracy.mean().unstack()[names]
tl = al.groupby(["label_noise", "dataset", "model"]).accuracy.mean().unstack()[names]
tf = af.groupby(["feat_noise", "dataset", "model"]).accuracy.mean().unstack()[names]
rc = tc.rank(axis=1, ascending=False).mean()
rows = []
for v, lbl in VARIANTS:
    def cmp(t, ref):
        if v == ref:
            return "---"
        w, ti, l = wtl(t[v], t[ref])
        p = wilc(t[v], t[ref])
        return f"{w}/{ti}/{l}, {fmt_p(p)}{'*' if p < 0.05 else ''}"
    rows.append(f"{lbl} & {tc[v].mean():.4f} & {rc[v]:.2f} & {tl[v].mean():.4f} & {tf[v].mean():.4f} & "
                f"{cmp(tl, 'RVFL')} & {cmp(tl, 'IFRVFLC')} \\\\")
write("tab_ablation.tex", f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Ablation of the sample weighting. Mean accuracy ($\\uparrow$) on the 14 clean datasets (with the average rank among these 11 rows) and over the 25 dataset--level pairs under label and feature noise; last two columns: each row against plain RVFL and against IFRVFLC under label noise (wins/ties/losses of the row, Wilcoxon $p$; $^*$: $p<0.05$). Rows 2--9 use the $(\\mu,\\nu)$ of Section~\\ref{{sec:method}} and the grid of the previous version; $g$ is the Gaussian centroid similarity.}}
\\label{{tab:ablation}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l cc c c cc}}
\\hline
 & \\multicolumn{{2}}{{c}}{{Clean}} & Label noise & Feature noise & \\multicolumn{{2}}{{c}}{{Label noise: row vs}} \\\\
Weight & Acc. & Rank & Acc. & Acc. & RVFL & IFRVFLC \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}}}
\\end{{table*}}
""")

# ------------------------------------------------------------------ ablation: components of the revised method
COMP = [(NEW, "RW-IFRVFL (full)"), ("v3:half", "trust fixed at $\\alpha=\\tfrac12$"),
        ("v3:smallk", "$k\\le 15$ instead of $k\\le 50$"), ("v3:balance", "weight $1-\\tilde\\pi$, target $\\tilde y=\\mathrm{balance}$"),
        ("v3:sym", "non-membership discounted by $g$")]
cn = [c for c, _ in COMP]
cc = load("v3_e1"); cl = load("v3_label"); cf = load("v3_feature"); ce = load("v3_e7")
gc = cc.groupby(["dataset", "model"]).accuracy.mean().unstack()[cn]
gl = cl.groupby(["label_noise", "dataset", "model"]).accuracy.mean().unstack()[cn]
gf = cf.groupby(["feat_noise", "dataset", "model"]).accuracy.mean().unstack()[cn]
hl = gl[gl.index.get_level_values(0) >= 0.3]
ec = ce.groupby("model").accuracy.mean()
rows = []
for c, lbl in COMP:
    def cm(t):
        if c == NEW:
            return "---"
        w, ti, l = wtl(t[NEW], t[c])
        p = wilc(t[NEW], t[c])
        return f"{w}/{ti}/{l}, {fmt_p(p)}{'*' if p < 0.05 else ''}"
    rows.append(f"{lbl} & {gc[c].mean():.4f} & {gl[c].mean():.4f} & {hl[c].mean():.4f} & {gf[c].mean():.4f} & {ec[c]:.4f} & {cm(gc)} & {cm(gl)} \\\\")
write("tab_components.tex", f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Components of the revised RW-IFRVFL, each changed in turn. Mean accuracy ($\\uparrow$) on the clean benchmark, under label noise (all levels and 30--40\\%), under feature noise and on the halophilic case study; last two columns: the full model against each variant on the clean data and under label noise (wins/ties/losses of the full model, Wilcoxon $p$; $^*$: $p<0.05$). All rows use the grid of Table~\\ref{{tab:hyperparameter_search}}.}}
\\label{{tab:components}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l ccccc cc}}
\\hline
 & Clean & \\multicolumn{{2}}{{c}}{{Label noise}} & Feature & Halo- & \\multicolumn{{2}}{{c}}{{Full model vs row}} \\\\
Variant & & all & 30--40\\% & noise & philic & clean & label noise \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}}}
\\end{{table*}}
""")

# ------------------------------------------------------------------ case study
cs = load("e7", "v3_e7")
cs = cs[cs.model.isin(MODELS)]
t = cs.groupby("model")[["accuracy", "f1_macro", "auc", "fit_s"]].agg(["mean", "std"])
rows = []
order = t[("accuracy", "mean")].sort_values(ascending=False).index
for m in order:
    cells = []
    for met in ["accuracy", "f1_macro", "auc"]:
        mu, sd = t.loc[m, (met, "mean")], t.loc[m, (met, "std")]
        best = t[(met, "mean")].max()
        s = f"{mu:.4f} $\\pm$ {sd:.4f}"
        cells.append(f"\\textbf{{{s}}}" if abs(mu - best) < 1e-9 else s)
    ft = cs[cs.model == m].fit_s.median()
    name = "\\textbf{RW-IFRVFL}" if m == NEW else CITE[m]
    rows.append(f"{name} & " + " & ".join(cells) + f" & {ft:.3g} \\\\")
write("tab_case_study.tex", f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Halophilic protein classification (278 proteins, 442 features): mean $\\pm$ standard deviation over the 10 outer folds ($\\uparrow$) and median training time in seconds. Rows sorted by accuracy; best in bold.}}
\\label{{tab:case_study}}
\\scriptsize
\\begin{{tabular}}{{lcccr}}
\\hline
Model & Accuracy & Macro-F1 & AUC & Time (s) \\\\
\\hline
{chr(10).join(rows)}
\\hline
\\end{{tabular}}
\\end{{table}}
""")

# ------------------------------------------------------------------ numbers for the text
lines = [f"CD = {cd:.3f}"]
for metric in ["accuracy", "f1_macro", "auc"]:
    s = summary[metric]
    lines.append(f"{metric}: Friedman p = {s['friedman_p']:.2e}; RW rank {s['ranks'][NEW]:.2f}; "
                 "ranks " + ", ".join(f"{m} {s['ranks'][m]:.2f}" for m in sorted(MODELS, key=lambda m: s['ranks'][m])))
    lines.append("   significant (Holm): " + ", ".join(f"{b} {s['wtl'][b]} p={s['holm'][b]:.3f}" for b in BASE if s['holm'][b] < 0.05))
lines.append("clean mean acc: " + ", ".join(f"{m} {acc[m].mean():.4f}" for m in MODELS))
lines.append("label levels:\n" + label_lv.round(3).to_string())
lines.append("feature levels:\n" + feat_lv.round(3).to_string())
write("numbers.txt", "\n".join(lines) + "\n")
