"""Distance concentration of the centroid membership: python concentration.py

For every dataset and outer training part, the relative spread (std/mean) of the distances
||x_i - c_{y_i}|| and the mean and spread of the centroid similarity g at spread scales 0.25 and
1.5 are computed exactly as in RW-IFRVFL (features preprocessed on the training part), together
with the mean non-membership and the share of relabelled training samples at the default
configuration of the sensitivity analysis. For the
halophilic case study, the share of training labels that the signed score relabels (S_i < 0) is
also reported for the spread scales of the sensitivity analysis. Writes
results/distance_concentration.csv and results/relabel_halophilic.csv.
"""
import os
import warnings

import numpy as np
import pandas as pd

import datasets as D
import evaluate as E
import models as M
from rw_ifrvfl import RWIFRVFL, ifs_triple

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SCALES = (0.25, 1.5)


def train_features(X, y, tr):
    pre = D.make_preprocessor(X.iloc[tr]).fit(X.iloc[tr])
    A = np.asarray(pre.transform(X.iloc[tr]), dtype=float)
    return A, np.where(y[tr] == 1, 1.0, -1.0)


rows = []
for ds in D.BENCHMARK + D.CASE_STUDIES:
    X, y, _ = D.load(ds)
    for f, (tr, _) in enumerate(E.outer_splits(ds, y)):
        A, yy = train_features(X, y, tr)
        cent = {c: A[yy == c].mean(axis=0) for c in (-1.0, 1.0)}
        dist = np.linalg.norm(A - np.where(yy[:, None] > 0, cent[1.0], cent[-1.0]), axis=1)
        r = dict(dataset=ds, fold=f, n_train=len(tr), d_enc=A.shape[1], dist_cv=dist.std() / dist.mean())
        for s in SCALES:
            g = np.exp(-dist ** 2 / (2.0 * (s * dist.mean()) ** 2))
            r[f"g_mean_s{s:g}"], r[f"g_std_s{s:g}"] = g.mean(), g.std()
        p = M.SENS_DEFAULT
        mu, nu, _, _ = ifs_triple(A, yy, p["k"], p["sigma_scale"], 50)
        score = (1.0 - p["alpha"]) + p["alpha"] * (mu - nu)
        r["nu_mean_default"], r["relabelled_default"] = nu.mean(), float(np.mean(score < 0))
        rows.append(r)
        print(ds, f, flush=True)
conc = pd.DataFrame(rows).groupby("dataset").mean(numeric_only=True).drop(columns="fold")
conc.round(4).to_csv(os.path.join(OUT, "distance_concentration.csv"))
print(conc.sort_values("d_enc").round(3).to_string())

rows = []
X, y, _ = D.load("halophilic")
for s in M.SENS_SWEEPS["sigma_scale"]:
    for f, (tr, _) in enumerate(E.outer_splits("halophilic", y)):
        A, yy = train_features(X, y, tr)
        model = RWIFRVFL(random_state=M.SEED, **{**M.SENS_DEFAULT, "sigma_scale": s}).fit(A, yy)
        rows.append(dict(sigma_scale=s, fold=f, relabelled=float(np.mean(model.targets_ != yy)),
                         zero_weight=float(np.mean(model.weights_ == 0))))
rel = pd.DataFrame(rows).groupby("sigma_scale")[["relabelled", "zero_weight"]].mean()
rel.round(4).to_csv(os.path.join(OUT, "relabel_halophilic.csv"))
print(rel.round(3).to_string())
