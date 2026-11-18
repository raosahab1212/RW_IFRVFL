"""Phase-0 smoke test: every benchmark model, 2 outer folds, small tuning budget."""
import sys
import time
import traceback
import warnings

import pandas as pd

import datasets as D
import evaluate as E
import models as M

warnings.filterwarnings("ignore")
dsets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["heart", "pima", "ionosphere"]
mods = sys.argv[2].split(",") if len(sys.argv) > 2 else M.BENCHMARK_MODELS
rows = []
for ds in dsets:
    X, y, _ = D.load(ds)
    for m in mods:
        t = time.perf_counter()
        try:
            rows += E.run_outer(m, X, y, ds, folds=[0, 1], n_configs=4)
            status = "ok"
        except Exception as e:
            status = "FAIL " + repr(e)
            traceback.print_exc(limit=3)
        print(f"{ds:11s} {m:10s} {time.perf_counter() - t:7.1f}s {status}", flush=True)
df = pd.DataFrame(rows)
df.to_csv("smoke_results.csv", index=False)
print(df.groupby(["model", "dataset"])[["accuracy", "f1_macro", "auc"]].mean()
      .round(4).unstack(1).to_string(), flush=True)
