"""Summary of the adversarial experiments: python adv_report.py -> results/adversarial_report.md."""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
TOL = 1e-3
NEW = "RW-IFRVFL"
BASE = ["SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "NB", "KNN", "NF-RVFL", "ANFIS",
        "GBTSVM", "TRKM", "RVFL", "IFRVFLC"]
MODELS = BASE + [NEW]


def load(name):
    return pd.read_csv(os.path.join(R, f"{name}_all.csv"))


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
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


def compare(g, hi):
    """RW-IFRVFL against each baseline over all pairs and over the high-noise pairs."""
    p_all = holm([wilc(g[NEW], g[b]) for b in BASE])
    p_hi = holm([wilc(hi[NEW], hi[b]) for b in BASE])
    rows = []
    for i, b in enumerate(BASE):
        rows.append(dict(model=b, wtl_all="/".join(map(str, wtl(g[NEW], g[b]))), p_all=p_all[i],
                         wtl_high="/".join(map(str, wtl(hi[NEW], hi[b]))), p_high=p_hi[i]))
    return pd.DataFrame(rows).set_index("model")


def level_table(g, col):
    lv = g.groupby(level=0).mean().T
    lv.columns = [f"{c:g}" for c in lv.columns]
    lv["rank"] = g.rank(axis=1, ascending=False).mean()
    return lv.sort_values("rank")


def label_section(exp, title):
    df = load(exp)
    g = df[df.model.isin(MODELS)].groupby(["label_noise", "dataset", "model"]).accuracy.mean().unstack()[MODELS]
    hi = g[g.index.get_level_values(0) >= 0.3]
    rnd = load("label_noise")
    gr = rnd[rnd.model.isin(MODELS)].groupby(["label_noise", "model"]).accuracy.mean().unstack()[MODELS]
    lv = level_table(g, "label_noise")
    lv["random_40"] = gr.loc[0.40]
    out = [f"## {title} (`{exp}`)", "", "Mean accuracy over the five datasets at each flip rate; "
           "`rank` is the average rank over the 25 dataset-rate pairs; `random_40` is the random-flip "
           "accuracy at 40% from `label_noise`.", "", lv.round(3).to_markdown(), "",
           "RW-IFRVFL against each model (wins/ties/losses, Holm-corrected Wilcoxon p), all rates and 30-40%:",
           "", compare(g, hi).round(3).to_markdown(), ""]
    return out, g


def evasion_section(exp, title, scope):
    df = load(exp)
    df = df[df.model.isin(MODELS)]
    clean = df[df.attack == "none"].groupby(["dataset", "model"]).accuracy.mean().unstack()[MODELS]
    out = [f"## {title} (`{exp}`)", "",
           "Models trained on clean data; test samples moved by L-inf PGD on an MLP surrogate (budget "
           f"eps x training SD) or by a random-sign perturbation of the same budget; {scope}. Mean accuracy "
           "over the five datasets.", ""]
    tabs = {}
    for attack in ("pgd", "random"):
        d = df[df.attack == attack]
        g = d.groupby(["adv_eps", "dataset", "model"]).accuracy.mean().unstack()[MODELS]
        tabs[attack] = g
        lv = level_table(g, "adv_eps")
        lv.insert(0, "clean", clean.mean())
        out += [f"### {attack} ({exp})", "", lv.round(3).to_markdown(), ""]
    g = tabs["pgd"]
    hi = g[g.index.get_level_values(0) >= 0.3]
    drop = (clean.mean() - g.xs(0.4, level=0).mean()).sort_values()
    out += ["Accuracy lost to PGD at eps = 0.4 (clean minus attacked):", "",
            drop.round(3).to_frame("drop").to_markdown(), "",
            "RW-IFRVFL against each model under PGD (all eps, and eps >= 0.3):", "",
            compare(g, hi).round(3).to_markdown(), ""]
    sur = os.path.join(R, "adv", "surrogates_cont.csv" if exp.endswith("_cont") else "surrogates.csv")
    if os.path.exists(sur):
        s = pd.read_csv(sur).groupby("dataset").mean(numeric_only=True)
        cols = ["mlp_test_acc"] + [c for c in s.columns if c.startswith("mlp_pgd_acc")]
        out += ["White-box accuracy of the MLP surrogate itself:", "", s[cols].round(3).to_markdown(), ""]
    return out


def pgd_by_dataset(eps=0.4):
    """PGD accuracy at one budget per dataset, all numeric columns versus continuous columns only."""
    cols = {}
    for exp, tag in (("adv_evasion", "all"), ("adv_evasion_cont", "cont")):
        d = load(exp)
        d = d[d.model.isin(MODELS) & (d.attack == "pgd") & (d.adv_eps == eps)]
        g = d.groupby(["model", "dataset"]).accuracy.mean().unstack()
        for ds in g.columns:
            cols[(ds, tag)] = g[ds]
    t = pd.DataFrame(cols).loc[MODELS]
    t.columns = [f"{ds} {tag}" for ds, tag in t.columns]
    return [f"PGD accuracy at eps = {eps:g} per dataset, all numeric columns perturbed (`all`) or "
            "continuous columns only (`cont`); wisconsin and magic have no integer-coded columns, so their "
            "two columns must agree:", "", t.round(3).to_markdown(), ""]


def main():
    lines = ["# Adversarial perturbation experiments", ""]
    for exp, title in (("adv_label_far", "Adversarial label flips, farthest-first"),
                       ("adv_label_near", "Adversarial label flips, nearest-first")):
        if os.path.exists(os.path.join(R, f"{exp}_all.csv")):
            lines += label_section(exp, title)[0]
    if os.path.exists(os.path.join(R, "adv_evasion_all.csv")):
        lines += evasion_section("adv_evasion", "Test-time evasion, all numeric columns",
                                 "all numeric columns are perturbed")
    if os.path.exists(os.path.join(R, "adv_evasion_cont_all.csv")):
        lines += evasion_section("adv_evasion_cont", "Test-time evasion, continuous columns only",
                                 f"numeric columns with at most 10 distinct training values (integer codes) "
                                 "are left unperturbed")
        lines += pgd_by_dataset()
    if os.path.exists(os.path.join(R, "adv_evasion_all.csv")):
        lines += ["## Notes", "",
                  "- The clean (`attack = none`) rows reproduce `benchmark` exactly for all 700 "
                  "(dataset, model, fold) triples.",
                  "- The attack is a transfer attack from a two-layer MLP, so it is strongest against models "
                  "with a similar smooth, dense decision function (the RVFL family, SVM) and weaker against "
                  "trees and KNN.",
                  "- Integer-coded columns (8 of 13 numeric columns of heart) and a class-constant binary "
                  "column of ionosphere are perturbed like continuous features. This explains the drop of "
                  "LightGBM, XGBoost (heart) and NB (ionosphere) already at eps = 0.05 under the random "
                  "control, whose split thresholds or class variances sit on those discrete values. "
                  "`adv_evasion_cont` leaves those columns unperturbed; its wisconsin results equal "
                  "`adv_evasion` exactly and its magic results differ in one of 2,380 rows by one test "
                  "sample (ANFIS, fold 0, eps = 0.3), from floating-point differences between runners.", ""]
    with open(os.path.join(R, "adversarial_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
