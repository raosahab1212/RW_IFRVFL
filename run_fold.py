"""
Run the full protocol for one (dataset, outer fold) and a list of models.

    python run_fold.py <dataset> <fold> <model1,model2,...> [tag]

Appends one row per model to results/fold_runs.csv as soon as it finishes, and
prints one line per model with its wall-clock time (tuning + final fit).
"""
import os
import sys
import time
import traceback
import warnings

import pandas as pd

import datasets as D
import evaluate as E

warnings.filterwarnings("ignore")
ds, fold, mods = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
tag = sys.argv[4] if len(sys.argv) > 4 else "e1"
out = os.path.join(E.HERE, "results", "fold_runs.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)

X, y, _ = D.load(ds)
for m in mods:
    t = time.perf_counter()
    try:
        rows = E.run_outer(m, X, y, ds, folds=[fold])
        for r in rows:
            r["tag"], r["wall_s"] = tag, time.perf_counter() - t
        pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        r = rows[0]
        print(f"{ds} f{fold} {m:10s} {r['wall_s']:8.1f}s acc={r['accuracy']:.4f} "
              f"f1={r['f1_macro']:.4f} auc={r['auc']:.4f} tune={r['tune_s']:.0f}s "
              f"fit={r['fit_s']:.1f}s n={r['train_n']}", flush=True)
    except Exception as e:
        print(f"{ds} f{fold} {m:10s} FAIL {e!r}", flush=True)
        traceback.print_exc()
