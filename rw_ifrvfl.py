"""RW-IFRVFL: reliability-weighted intuitionistic fuzzy RVFL classifier."""
import hashlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

_CACHE = {}


def neighbours(X, k_max):
    """Indices of the k_max nearest other samples of each row."""
    key = (hashlib.md5(np.ascontiguousarray(X).view(np.uint8)).hexdigest(), X.shape, k_max)
    if key not in _CACHE:
        n = X.shape[0]
        idx = NearestNeighbors(n_neighbors=min(k_max + 1, n)).fit(X).kneighbors(X, return_distance=False)
        out = np.empty((n, idx.shape[1] - 1), dtype=int)
        for r in range(n):
            keep = idx[r] != r
            if keep.all():
                keep[-1] = False
            out[r] = idx[r][keep]
        if len(_CACHE) >= 64:
            _CACHE.clear()
        _CACHE[key] = out
    return _CACHE[key]


def clear_cache():
    _CACHE.clear()


def sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def ifs_triple(Xn, y, k, sigma_scale, k_max, symmetric=False):
    """Membership, non-membership, hesitation and spread for labels y in {-1, +1}."""
    idx = neighbours(Xn, k_max)
    if k > idx.shape[1]:
        raise ValueError(f"k={k} exceeds the available neighbours ({idx.shape[1]}).")
    nu = (y[idx[:, :k]] != y[:, None]).mean(axis=1)
    cent = {c: Xn[y == c].mean(axis=0) for c in (-1.0, 1.0)}
    dist = np.linalg.norm(Xn - np.where(y[:, None] > 0, cent[1.0], cent[-1.0]), axis=1)
    sigma_bar = dist.mean()
    sigma = sigma_scale * (sigma_bar if sigma_bar > 0 else 1.0)
    g = np.exp(-dist ** 2 / (2.0 * sigma ** 2))
    mu = (1.0 - nu) * g
    if symmetric:
        return mu, g * nu, 1.0 - g, sigma
    return mu, nu, 1.0 - mu - nu, sigma


class RWIFRVFL(BaseEstimator, ClassifierMixin):
    """RW-IFRVFL for binary classification."""

    def __init__(self, n_hidden=100, lam=0.1, k=9, sigma_scale=1.5, alpha=0.75,
                 symmetric=False, rule="signed", random_state=42):
        self.n_hidden = n_hidden
        self.lam = lam
        self.k = k
        self.sigma_scale = sigma_scale
        self.alpha = alpha
        self.symmetric = symmetric
        self.rule = rule
        self.random_state = random_state

    def _augment(self, Xn):
        return np.hstack([Xn, sigmoid(Xn @ self.W_.T + self.b_)])

    def _setup(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("RW-IFRVFL is a binary classifier.")
        y = np.where(y == self.classes_[1], 1.0, -1.0)
        self.scaler_ = MinMaxScaler().fit(X)
        Xn = self.scaler_.transform(X)
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.uniform(-1.0, 1.0, size=(self.n_hidden, Xn.shape[1]))
        self.b_ = rng.uniform(-1.0, 1.0, size=self.n_hidden)
        return Xn, self._augment(Xn), y

    def _solve(self, Z, w, t):
        A = Z.T @ (Z * w[:, None]) + self.lam * np.eye(Z.shape[1])
        self.beta_ = np.linalg.solve(A, Z.T @ (w * t))
        self.weights_, self.targets_ = w, t

    def fit(self, X, y):
        Xn, Z, y = self._setup(X, y)
        mu, nu, _, self.sigma_ = ifs_triple(Xn, y, self.k, self.sigma_scale, 50, self.symmetric)
        mu, nu = (1.0 - self.alpha) + self.alpha * mu, self.alpha * nu
        s = mu - nu
        if self.rule == "signed":
            w, t = np.abs(s), np.where(s < 0, -y, y)
        else:
            r = mu + nu
            w, t = r, y * np.divide(s, r, out=np.ones_like(r), where=r > 1e-12)
        self._solve(Z, w, t)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "beta_")
        return self._augment(self.scaler_.transform(check_array(X))) @ self.beta_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])
