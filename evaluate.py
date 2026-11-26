"""Nested cross-validation: saved outer 10-fold splits, inner model selection, train-only noise."""
import os
import time
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import adversarial
import gbtsvm
import models as M
import rvfl_variants
import rw_ifrvfl
from datasets import make_preprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(HERE, "splits")
TUNE_MAX = 5000
OUTER_FOLDS = 10
SEED = 42


def outer_splits(dataset, y):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    path = os.path.join(SPLIT_DIR, f"{dataset}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if len(z["fold"]) != len(y):
            raise ValueError(f"{path} does not match dataset size")
        fold = z["fold"]
    else:
        fold = np.empty(len(y), dtype=int)
        skf = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
        for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold[te] = f
        np.savez(path, fold=fold)
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(OUTER_FOLDS)]


def _stratified_subset(idx, y, n, seed=SEED):
    if len(idx) <= n:
        return idx
    keep, _ = train_test_split(idx, train_size=n, stratify=y[idx], random_state=seed)
    return np.sort(keep)


def flip_labels(y, rate, seed):
    y = y.copy()
    n = int(round(rate * len(y)))
    if n:
        idx = np.random.default_rng(seed).choice(len(y), size=n, replace=False)
        y[idx] = 1 - y[idx]
    return y


def gaussian_feature_noise(Xt, rate, seed):
    if rate == 0:
        return Xt
    sd = Xt.std(axis=0, keepdims=True)
    return Xt + np.random.default_rng(seed).normal(0.0, 1.0, Xt.shape) * rate * sd


def train_labels(y, tr, rate, seed, kind="random", adv=None):
    """Noisy training labels: random flips, or flips chosen by the adversarial surrogate (far / near)."""
    if not rate:
        return y[tr]
    if kind == "random":
        return flip_labels(y[tr], rate, seed)
    return adversarial.flipped_labels(y, tr, rate, kind, adv)


def scores_of(model, X):
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float)
        if s.ndim == 2:
            s = s[:, 1] - s[:, 0] if s.shape[1] == 2 else s.ravel()
    else:
        s = model.predict_proba(X)[:, 1]
    return np.asarray(s, dtype=float).ravel()


def metrics(y, pred, score):
    out = dict(accuracy=accuracy_score(y, pred),
               f1_macro=f1_score(y, pred, average="macro", zero_division=0),
               f1_minority=f1_score(y, pred, pos_label=1, zero_division=0))
    try:
        out["auc"] = roc_auc_score(y, score) if np.isfinite(score).all() else np.nan
    except ValueError:
        out["auc"] = np.nan
    return out


def fit_score(name, params, Xtr, ytr, Xte, yte, feat_noise=0.0, noise_seed=0, extra_tests=None):
    pre = make_preprocessor(Xtr).fit(Xtr)
    A, B = pre.transform(Xtr), pre.transform(Xte)
    A = gaussian_feature_noise(A, feat_noise, noise_seed)
    model = M.build(name, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        model.fit(A, ytr)
        t1 = time.perf_counter()
        pred = model.predict(B)
        score = scores_of(model, B)
        t2 = time.perf_counter()
        extra = {}
        for key, Xe in (extra_tests or {}).items():
            E = pre.transform(Xe)
            extra[key] = metrics(yte, model.predict(E), scores_of(model, E))
    out = {**metrics(yte, pred, score), "fit_s": t1 - t0, "predict_s": t2 - t1}
    if extra_tests:
        out["_extra"] = extra
    return out


def tune(name, X, y, train_idx, label_noise=0.0, feat_noise=0.0, noise_seed=0, n_configs=M.N_CONFIGS,
         label_kind="random", adv=None):
    cands = M.configs(name, n_configs)
    if len(cands) == 1:
        return cands[0], np.nan, 0.0
    idx = _stratified_subset(train_idx, y, TUNE_MAX)
    folds = 3 if len(train_idx) > TUNE_MAX else 5
    ytr_all = y.copy()
    if label_noise:
        ytr_all[train_idx] = train_labels(y, train_idx, label_noise, noise_seed, label_kind, adv)
    inner = list(StratifiedKFold(folds, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(idx)), ytr_all[idx]))
    t0 = time.perf_counter()
    best, best_s = cands[0], -np.inf
    for p in cands:
        vals = []
        for a, b in inner:
            ia, ib = idx[a], idx[b]
            try:
                r = fit_score(name, dict(p), X.iloc[ia], ytr_all[ia], X.iloc[ib], ytr_all[ib],
                              feat_noise, noise_seed)
                vals.append(r["f1_macro"])
            except Exception:
                vals.append(-np.inf)
        s = float(np.mean(vals))
        if s > best_s:
            best, best_s = p, s
    return best, best_s, time.perf_counter() - t0


def run_outer(name, X, y, dataset, label_noise=0.0, feat_noise=0.0, tune_params=True,
              fixed_params=None, folds=None, n_configs=M.N_CONFIGS, label_kind="random", evasion=False):
    rows = []
    for f, (tr, te) in enumerate(outer_splits(dataset, y)):
        if folds is not None and f not in folds:
            continue
        seed = SEED + f
        adv = adversarial.load(dataset, f) if evasion or (label_noise and label_kind != "random") else None
        if tune_params:
            params, inner_f1, tune_s = tune(name, X, y, tr, label_noise, feat_noise, seed, n_configs,
                                            label_kind, adv)
        else:
            params, inner_f1, tune_s = dict(fixed_params or {}), np.nan, 0.0
        ytr = train_labels(y, tr, label_noise, seed, label_kind, adv)
        cap = M.CAPPED.get(name)
        keep = np.arange(len(tr))
        if cap and len(tr) > cap:
            keep = np.searchsorted(tr, _stratified_subset(tr, _full_labels(tr, ytr), cap))
        rw_ifrvfl.clear_cache()
        rvfl_variants._D2_CACHE.clear()
        gbtsvm._BALL_CACHE.clear()
        extra = adversarial.evasion_tests(X, te, adv) if evasion else None
        r = fit_score(name, dict(params), X.iloc[tr[keep]], ytr[keep], X.iloc[te], y[te], feat_noise, seed,
                      extra_tests=extra)
        base = dict(dataset=dataset, model=name, fold=f, label_noise=label_noise, feat_noise=feat_noise,
                    train_n=len(keep), params=repr(params), inner_f1=inner_f1, tune_s=tune_s)
        if label_kind != "random":
            base["label_kind"] = label_kind
        extra_res = r.pop("_extra", {})
        if evasion:
            base.update(attack="none", adv_eps=0.0)
        rows.append({**base, **r})
        for (attack, eps), m in extra_res.items():
            rows.append({**base, **r, **m, "attack": attack, "adv_eps": eps})
    return rows


def _full_labels(tr, ytr):
    full = np.zeros(tr.max() + 1, dtype=int)
    full[tr] = ytr
    return full
