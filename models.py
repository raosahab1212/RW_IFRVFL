"""Model registry, hyperparameter grids and the equal tuning budget (at most 30 configurations)."""
import contextlib
import io

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from gbtsvm import GBTSVMClassifier
from rvfl_variants import EvidenceRVFL, FuzzyRVFL
from rw_ifrvfl import RWIFRVFL
from trkm import TRKMClassifier

N_CONFIGS = 30
SEED = 42
HIDDEN = [50, 100, 150, 200, 300]
LAMBDA = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


class ANFISClassifier(BaseEstimator, ClassifierMixin):
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
        from anfis import HybridAlgorithmClassifier
        torch.manual_seed(SEED)
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
    from nfrvfl import NeuroFuzzyRVFLClassifier
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


def _variant(name):
    return lambda **p: FuzzyRVFL(variant=name, random_state=SEED, **p)


def _rw(**fixed):
    return lambda **p: RWIFRVFL(random_state=SEED, **fixed, **p)


def _intermediate(**fixed):
    return lambda **p: EvidenceRVFL(random_state=SEED, **fixed, **p)


def _drop(space, *names):
    return {p: v for p, v in space.items() if p not in names}


RVFL_SPACE = {"n_hidden": HIDDEN, "lam": LAMBDA}
RW_SPACE = {"n_hidden": HIDDEN, "lam": LAMBDA, "k": [3, 9, 25, 50],
            "sigma_scale": [0.5, 1.0, 1.5, 2.0], "alpha": [0.0, 0.5, 0.75, 1.0]}
ABL_SPACE = {"n_hidden": HIDDEN, "lam": [1e-4, 1e-3, 1e-2, 1e-1], "k": [3, 5, 7, 9],
             "sigma_scale": [0.5, 1.0, 1.5, 2.0]}
INT_SPACE = {**RW_SPACE, "k": [3, 5, 9, 15]}

REGISTRY = {
    "SVM": (lambda **p: SVC(kernel="rbf", **p),
            {"C": [0.01, 0.1, 1, 10, 100, 1000], "gamma": ["scale", 1e-3, 1e-2, 0.1, 1, 10]}),
    "RF": (lambda **p: RandomForestClassifier(n_jobs=1, random_state=SEED, **p),
           {"n_estimators": [100, 200, 500], "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", "log2", 0.5]}),
    "XGBoost": (_xgb, {"n_estimators": [100, 300, 500], "max_depth": [3, 5, 7],
                       "learning_rate": [0.01, 0.05, 0.1, 0.2],
                       "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.7, 0.85, 1.0]}),
    "LightGBM": (_lgbm, {"n_estimators": [100, 300, 500], "num_leaves": [15, 31, 63],
                         "learning_rate": [0.01, 0.05, 0.1, 0.2], "min_child_samples": [10, 20, 50]}),
    "CatBoost": (_cat, {"iterations": [200, 500], "depth": [4, 6, 8],
                        "learning_rate": [0.03, 0.1, 0.2], "l2_leaf_reg": [1, 3, 10]}),
    "NB": (lambda **p: GaussianNB(**p), {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}),
    "KNN": (lambda **p: KNeighborsClassifier(n_jobs=1, **p),
            {"n_neighbors": [3, 5, 7, 9, 11, 15, 21], "weights": ["uniform", "distance"]}),
    "NF-RVFL": (_nfrvfl, {"n_hidden": HIDDEN, "reg": LAMBDA}),
    "ANFIS": (lambda **p: ANFISClassifier(**p), {"L": [2, 3], "eta": [0.01, 0.1], "num_cluster": [2, 3, 4]}),
    "GBTSVM": (lambda **p: GBTSVMClassifier(random_state=0, **p),
               {"d1": [1e-5, 1e-3, 1e-1, 1, 10], "purity": [0.85, 0.9, 0.925, 0.95],
                "min_ball_size": [1, 2, 4]}),
    "TRKM": (lambda **p: TRKMClassifier(**p),
             {"eta1": [0.01, 0.1, 1, 10], "lambd1": [1e-3, 1e-1, 1], "mew": [2.0 ** e for e in range(-3, 6)]}),
    "RVFL": (_variant("rvfl"), RVFL_SPACE),
    "IFRVFLC": (_variant("ifrvflc"), {**RVFL_SPACE, "gamma": [2.0 ** e for e in range(-5, 6)],
                                      "radius": [round(0.1 * i, 1) for i in range(1, 11)]}),
    "RW-IFRVFL": (_rw(), RW_SPACE),
    "abl:centroid": (_variant("centroid"), _drop(ABL_SPACE, "k")),
    "abl:neighbour": (_variant("neighbour"), _drop(ABL_SPACE, "sigma_scale")),
    "abl:product_unscaled": (_variant("product_unscaled"), ABL_SPACE),
    "abl:score_unscaled": (_variant("score_unscaled"), ABL_SPACE),
    "abl:product": (_variant("product"), ABL_SPACE),
    "abl:score": (_variant("score"), ABL_SPACE),
    "abl:ha": (_variant("ha"), ABL_SPACE),
    "abl:shuffled": (_variant("shuffled"), ABL_SPACE),
    "comp:alpha_half": (_rw(alpha=0.5), _drop(RW_SPACE, "alpha")),
    "comp:small_k": (_rw(), {**RW_SPACE, "k": [3, 5, 9, 15]}),
    "comp:balance": (_rw(rule="balance"), RW_SPACE),
    "comp:discounted": (_rw(symmetric=True), RW_SPACE),
    "int:balance": (_intermediate(), INT_SPACE),
    "int:no_weight": (_intermediate(weight="none"), INT_SPACE),
    "int:pignistic": (_intermediate(weight="none", target="pignistic"), INT_SPACE),
}

BASELINES = ["SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "NB", "KNN", "NF-RVFL", "ANFIS",
             "GBTSVM", "TRKM", "RVFL", "IFRVFLC"]
MAIN_MODELS = BASELINES + ["RW-IFRVFL"]
ABLATION_MODELS = [m for m in REGISTRY if m.split(":")[0] in ("abl", "comp", "int")]
COMPONENT_MODELS = [m for m in REGISTRY if m.startswith("comp:")]

SENS_DEFAULT = {"n_hidden": 100, "lam": 0.1, "k": 9, "sigma_scale": 1.5, "alpha": 0.75}
SENS_SWEEPS = {"alpha": [0.0, 0.25, 0.5, 0.75, 1.0], "k": [3, 5, 9, 15, 25, 50],
               "sigma_scale": [0.25, 0.5, 1.0, 1.5, 2.0, 4.0], "n_hidden": [50, 100, 150, 200, 300],
               "lam": [1e-4, 1e-3, 1e-2, 1e-1, 1.0], "seed": [0, 1, 2, 3, 4]}
REGISTRY["sens:default"] = (_rw(), {p: [v] for p, v in SENS_DEFAULT.items()})
SENS_MODELS = ["sens:default"]
for _p, _values in SENS_SWEEPS.items():
    for _v in _values:
        if _p != "seed" and _v == SENS_DEFAULT[_p]:
            continue
        _cfg = {**SENS_DEFAULT, **({} if _p == "seed" else {_p: _v})}
        _fixed = {"random_state": _v} if _p == "seed" else {}
        REGISTRY[f"sens:{_p}={_v:g}"] = (
            lambda _fixed=_fixed, **q: RWIFRVFL(**{"random_state": SEED, **_fixed, **q}),
            {c: [x] for c, x in _cfg.items()})
        SENS_MODELS.append(f"sens:{_p}={_v:g}")

CAPPED = {"SVM": 20000, "TRKM": 8000, "ANFIS": 20000, "GBTSVM": 20000}
_TIED = {"GBTSVM": [("d1", "d2")], "TRKM": [("eta1", "eta2"), ("lambd1", "lambd2")]}


def build(name, **params):
    factory, _ = REGISTRY[name]
    for a, b in _TIED.get(name, []):
        if a in params:
            params[b] = params[a]
    return factory(**params)


def configs(name, n=N_CONFIGS, seed=SEED):
    space = REGISTRY[name][1]
    grid = ParameterGrid(space)
    if len(grid) <= n:
        return list(grid)
    return list(ParameterSampler(space, n_iter=n, random_state=seed))
