"""Combine per-task result files into results/<experiment>_all.csv and report coverage."""
import glob
import os
import sys

import pandas as pd

from experiments import EXPERIMENTS, all_tasks, result_name

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for exp in EXPERIMENTS:
    files = sorted(glob.glob(os.path.join(root, exp, "*.csv")))
    if not files:
        continue
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv(os.path.join(root, f"{exp}_all.csv"), index=False)
    expected = {result_name(t) for t in all_tasks(exp)}
    have = {f"{exp}/{os.path.basename(f)}" for f in files}
    missing = sorted(expected - have)
    print(f"{exp}: {len(have & expected)}/{len(expected)} tasks done, {len(missing)} missing")
    for m in missing[:20]:
        print("   missing", m)
    keys = ["dataset", "model", "label_noise", "feat_noise"] + [c for c in ("attack", "adv_eps") if c in df]
    summ = df.groupby(keys)[["accuracy", "f1_macro", "auc", "fit_s"]].mean()
    summ["folds"] = df.groupby(keys).size()
    summ.round(4).to_csv(os.path.join(root, f"{exp}_summary.csv"))
