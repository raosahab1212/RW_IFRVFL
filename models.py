"""
Model registry and hyperparameter search spaces for the RW-IFRVFL experiments.

Every model:
  * is built by `build(name, **params)` as a scikit-learn style estimator,
  * runs single-threaded (fair timing; parallelism is applied across folds),
  * gets the same tuning budget: up to N_CONFIGS configurations sampled with a
    fixed seed from its space (the full grid when the grid is smaller).
"""

import io
import os
import sys
import contextlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "legacy"))

from rvfl_family import FuzzyRVFL          # noqa: E402
from rw_ifrvfl import EvidenceRVFL, RWIFRVFL, SignedScoreRVFL, SEARCH_SPACE as RW_SPACE  # noqa: E402
from gbtsvm import GBTSVMClassifier        # noqa: E402
from trkm import TRKMClassifier            # noqa: E402

N_CONFIGS = 30
SEED = 42

HIDDEN = [50, 100, 150, 200, 300]
LAMBDA = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


class ANFISClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn wrapper around legacy/ANFIS.py (HybridAlgorithmClassifier)."""

    def __init__(self, L=3, eta=0.1, num_iterations=100, num_cluster=3,
                 fuzziness_index=2, a=1.0, epsilon=1e-5, lambda_reg=1e-3):
        self.L = L
        self.eta = eta
        self.num_iterations = num_iterations
        self.num_cluster = num_cluster
        self.fuzziness_index = fuzziness_index
        self.a = a
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg

    def fit(self, X, y):
        import torch
        torch.manual_seed(SEED)
        from ANFIS import HybridAlgorithmClassifier
        with contextlib.redirect_stdout(io.StringIO()):
            self.model_ = HybridAlgorithmClassifier(
                np.asarray(X, float), np.asarray(y), self.L, self.eta,
                self.num_iterations, self.num_cluster, self.fuzziness_index,
                self.a, self.epsilon, self.lambda_reg, lambda1=1e-2, lambda2=1e-1)
            self.model_.forward()
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.model_.predict_proba(np.asarray(X, float))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _nfrvfl(**p):
    with contextlib.redirect_stdout(io.StringIO()):
        from NFRVFL import NeuroFuzzyRVFLClassifier
    return NeuroFuzzyRVFLClassifier(random_state=SEED, **p)


def _xgb(**p):
    from xgboost import XGBClassifier
    return XGBClassifier(n_jobs=1, random_state=SEED, eval_metric="logloss", **p)


def _lgbm(**p):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_jobs=1, random_state=SEED, verbose=-1, **p)


def _cat(**p):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(thread_count=1, random_seed=SEED, verbose=False,
                              allow_writing_files=False, **p)


def _rvfl(variant):
    return lambda **p: FuzzyRVFL(variant=variant, random_state=SEED, **p)


RVFL_SPACE = {"n_hidden": HIDDEN, "lam": LAMBDA}
# ablation variants share RW-IFRVFL's grid (manuscript Table 5), minus unused parameters
GAUSS_SPACE = {p: v for p, v in RW_SPACE.items() if p != "k"}
NONMEM_SPACE = {p: v for p, v in RW_SPACE.items() if p != "sigma_scale"}
# proposed revision: RVFL's own (n_hidden, lam) grid plus the IF parameters and the trust alpha;
# k reaches 15 because averaging out heavy label noise needs larger neighbourhoods
V2_SPACE = {"n_hidden": HIDDEN, "lam": LAMBDA, "k": [3, 5, 9, 15],
            "sigma_scale": [0.5, 1.0, 1.5, 2.0], "alpha": [0.0, 0.5, 0.75, 1.0]}


def _v2(**fixed):
    return lambda **p: EvidenceRVFL(random_state=SEED, **fixed, **p)


def _drop(space, *names):
    return {p: v for p, v in space.items() if p not in names}


# v3: same (n_hidden, lam) grid as every RVFL baseline; neighbourhoods up to 50 so that the
# correction can average out heavy label noise; alpha = 1/2 is the published weight
V3_SPACE = {"n_hidden": HIDDEN, "lam": LAMBDA, "k": [3, 9, 25, 50],
            "sigma_scale": [0.5, 1.0, 1.5, 2.0], "alpha": [0.0, 0.5, 0.75, 1.0]}


def _v3(**fixed):
    return lambda **p: SignedScoreRVFL(random_state=SEED, **fixed, **p)

REGISTRY = {
    "SVM": (lambda **p: SVC(kernel="rbf", **p),
            {"C": [0.01, 0.1, 1, 10, 100, 1000],
             "gamma": ["scale", 1e-3, 1e-2, 0.1, 1, 10]}),
    "RF": (lambda **p: RandomForestClassifier(n_jobs=1, random_state=SEED, **p),
           {"n_estimators": [100, 200, 500], "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", "log2", 0.5]}),
    "XGBoost": (_xgb,
                {"n_estimators": [100, 300, 500], "max_depth": [3, 5, 7],
                 "learning_rate": [0.01, 0.05, 0.1, 0.2],
                 "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.7, 0.85, 1.0]}),
    "LightGBM": (_lgbm,
                 {"n_estimators": [100, 300, 500], "num_leaves": [15, 31, 63],
                  "learning_rate": [0.01, 0.05, 0.1, 0.2],
                  "min_child_samples": [10, 20, 50]}),
    "CatBoost": (_cat,
                 {"iterations": [200, 500], "depth": [4, 6, 8],
                  "learning_rate": [0.03, 0.1, 0.2], "l2_leaf_reg": [1, 3, 10]}),
    "NB": (lambda **p: GaussianNB(**p),
           {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}),
    "KNN": (lambda **p: KNeighborsClassifier(n_jobs=1, **p),
            {"n_neighbors": [3, 5, 7, 9, 11, 15, 21], "weights": ["uniform", "distance"]}),
    "NF-RVFL": (_nfrvfl, {"n_hidden": HIDDEN, "reg": LAMBDA}),
    "ANFIS": (lambda **p: ANFISClassifier(**p),
              {"L": [2, 3], "eta": [0.01, 0.1], "num_cluster": [2, 3, 4]}),
    "GBTSVM": (lambda **p: GBTSVMClassifier(random_state=0, **p),
               {"d1": [1e-5, 1e-3, 1e-1, 1, 10], "purity": [0.85, 0.9, 0.925, 0.95],
                "min_ball_size": [1, 2, 4]}),
    "TRKM": (lambda **p: TRKMClassifier(**p),
             {"eta1": [0.01, 0.1, 1, 10], "lambd1": [1e-3, 1e-1, 1],
              "mew": [2.0 ** e for e in range(-3, 6)]}),
    "RVFL": (_rvfl("rvfl"), RVFL_SPACE),
    # kernel width and neighbourhood radius grids from Mishra et al. (2023), Sec. 4
    "IFRVFLC": (_rvfl("ifrvflc"), {**RVFL_SPACE, "gamma": [2.0 ** e for e in range(-5, 6)],
                                   "radius": [round(0.1 * i, 1) for i in range(1, 11)]}),
    "RW-IFRVFL": (lambda **p: RWIFRVFL(random_state=SEED, **p), RW_SPACE),
    "abl:fuzzy": (_rvfl("fuzzy"), GAUSS_SPACE),
    "abl:nonmem": (_rvfl("nonmem"), NONMEM_SPACE),
    "abl:pw_unscaled": (_rvfl("pw_unscaled"), RW_SPACE),
    "abl:sf_unscaled": (_rvfl("sf_unscaled"), RW_SPACE),
    "abl:pw_scaled": (_rvfl("pw_scaled"), RW_SPACE),
    "abl:rw_ha": (_rvfl("rw_ha"), RW_SPACE),
    "abl:rw_shuffled": (_rvfl("rw_shuffled"), RW_SPACE),
    # proposed revision and its pre-registered ablations (same 30 sampled configurations
    # wherever the space is the same)
    "RW-IFRVFL2": (_v2(), V2_SPACE),
    "v2:asym": (_v2(symmetric=False), V2_SPACE),                       # manuscript's nu_i = f_i
    "v2:alpha1": (_v2(alpha=1.0), _drop(V2_SPACE, "alpha")),          # no trust calibration
    "v2:noweight": (_v2(weight="none"), V2_SPACE),                     # hesitation unused
    "v2:nocorr": (_v2(target="label"), V2_SPACE),                      # weights only
    "v2:pignistic": (_v2(weight="none", target="pignistic"), V2_SPACE),
    "v2:knnonly": (_v2(centroid=False), _drop(V2_SPACE, "sigma_scale")),
    # v3 (signed fused score) and its pre-registered ablations
    "RW-IFRVFL3": (_v3(), V3_SPACE),
    "v3:sym": (_v3(symmetric=True), V3_SPACE),                         # discounted nu = g f
    "v3:smallk": (_v3(), {**V3_SPACE, "k": [3, 5, 9, 15]}),            # v2's neighbourhoods
    "v3:half": (_v3(alpha=0.5), _drop(V3_SPACE, "alpha")),            # published weight only
    "v3:balance": (_v3(rule="balance"), V3_SPACE),                     # v2's weight and target
}
V3_MODELS = ["RW-IFRVFL3", "v3:sym", "v3:smallk", "v3:half", "v3:balance"]
V2_MODELS = ["RW-IFRVFL2", "v2:asym", "v2:alpha1", "v2:noweight", "v2:nocorr", "v2:pignistic",
             "v2:knnonly"]

BENCHMARK_MODELS = ["SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "NB", "KNN",
                    "NF-RVFL", "ANFIS", "GBTSVM", "TRKM", "RVFL", "IFRVFLC", "RW-IFRVFL"]
ABLATION_MODELS = ["RVFL", "abl:fuzzy", "abl:nonmem", "abl:pw_unscaled", "abl:sf_unscaled",
                   "abl:pw_scaled", "RW-IFRVFL", "abl:rw_ha", "abl:rw_shuffled", "IFRVFLC"]
# Models whose fit memory/time is O(N^2) or worse: training capped on large data
CAPPED = {"SVM": 20000, "TRKM": 8000, "ANFIS": 20000, "GBTSVM": 20000}

# GBTSVM: tie the second box constraint to the first, as in the reference main.py
_TIED = {"GBTSVM": ("d1", "d2"), "TRKM": [("eta1", "eta2"), ("lambd1", "lambd2")]}


def build(name, **params):
    factory, _ = REGISTRY[name]
    tied = _TIED.get(name, [])
    for a, b in ([tied] if isinstance(tied, tuple) else tied):
        if a in params:
            params[b] = params[a]
    return factory(**params)


def configs(name, n=N_CONFIGS, seed=SEED):
    """Deterministic list of configurations to try for `name`."""
    space = REGISTRY[name][1]
    grid = ParameterGrid(space)
    if len(grid) <= n:
        return list(grid)
    return list(ParameterSampler(space, n_iter=n, random_state=seed))
