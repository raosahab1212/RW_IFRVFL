"""RW-IFRVFL with alpha = 0 against plain RVFL (Proposition 2): python checks/compare_alpha0.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, datasets as D, evaluate as E
from rvfl_variants import FuzzyRVFL
from rw_ifrvfl import RWIFRVFL

worst, cases, same_pred = 0.0, 0, 0
for ds in ["heart", "pima", "wisconsin", "ionosphere"]:
    X, y, _ = D.load(ds)
    for tr, te in E.outer_splits(ds, y):
        pre = D.make_preprocessor(X.iloc[tr]).fit(X.iloc[tr])
        A, B = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
        for L, lam in [(50, 1e-3), (100, 0.1), (300, 1.0)]:
            rw = RWIFRVFL(n_hidden=L, lam=lam, alpha=0.0, random_state=42).fit(A, y[tr])
            rv = FuzzyRVFL(variant="rvfl", n_hidden=L, lam=lam, random_state=42).fit(A, y[tr])
            s1, s2 = E.scores_of(rw, B), E.scores_of(rv, B)
            worst = max(worst, float(np.abs(s1 - s2).max() / max(np.abs(s2).max(), 1e-300)))
            same_pred += bool(np.array_equal(rw.predict(B), rv.predict(B))); cases += 1
print(f"alpha=0 vs RVFL: identical predictions in {same_pred}/{cases} (dataset, fold, setting) cases; "
      f"max relative difference of the decision values {worst:.1e}")
