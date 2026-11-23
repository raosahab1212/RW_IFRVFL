"""RVFL, IFRVFLC (Mishra et al., 2023) and the sample-weighting variants of the ablation study."""
import hashlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from rw_ifrvfl import RWIFRVFL, ifs_triple, neighbours, sigmoid

VARIANTS = ("rvfl", "centroid", "neighbour", "product_unscaled", "score_unscaled",
            "product", "score", "ha", "shuffled", "ifrvflc")
K_MAX = 15

_D2_CACHE = {}
_D2_CACHE_N = 6000
_BLOCK = 10_000_000


def _sq_dist_blocks(X):
    n = X.shape[0]
    key = hashlib.md5(np.ascontiguousarray(X).view(np.uint8)).hexdigest() + str(X.shape)
    hit = _D2_CACHE.get(key)
    if hit is not None:
        yield slice(0, n), hit
        return
    sq = np.einsum("ij,ij->i", X, X)
    step = n if n <= _D2_CACHE_N else max(1, _BLOCK // n)
    for s in range(0, n, step):
        rows = slice(s, min(s + step, n))
        d2 = np.maximum(sq[rows, None] + sq[None, :] - 2.0 * (X[rows] @ X.T), 0.0)
        if n <= _D2_CACHE_N:
            _D2_CACHE.clear()
            _D2_CACHE[key] = d2
        yield rows, d2


def _mishra_if(X, y, gamma, radius, delta):
    """Membership and non-membership of IFRVFLC, Eqs. 22-23 of Mishra et al. (2023)."""
    n = X.shape[0]
    same_k, cnt, het = np.empty(n), np.empty(n), np.empty(n)
    r2 = radius ** 2
    for rows, d2 in _sq_dist_blocks(X):
        same = y[rows, None] == y[None, :]
        inside = d2 <= r2
        cnt[rows] = inside.sum(axis=1)
        het[rows] = (inside & ~same).sum(axis=1)
        same_k[rows] = np.where(same, np.exp(-gamma * d2), 0.0).sum(axis=1)
    mu = np.empty(n)
    for c in (-1.0, 1.0):
        m = y == c
        s = same_k[m] / m.sum()
        dist = np.sqrt(np.maximum(1.0 - 2.0 * s + s.mean(), 0.0))
        mu[m] = 1.0 - dist / (dist.max() + delta)
    rho = het / np.maximum(cnt, 1)
    return mu, (1.0 - mu) * rho


def _ha_score(mu, nu):
    s = np.where(nu == 0, mu, 0.0)
    mid = (nu > 0) & (mu > nu)
    s[mid] = (1.0 - nu[mid]) / (2.0 - mu[mid] - nu[mid])
    return s


class FuzzyRVFL(BaseEstimator, ClassifierMixin):
    """Weighted RVFL whose sample weights come from one of VARIANTS."""

    def __init__(self, variant="score", n_hidden=100, lam=1e-3, k=5, sigma_scale=1.0,
                 gamma=1.0, radius=0.5, delta=1e-4, random_state=42):
        self.variant = variant
        self.n_hidden = n_hidden
        self.lam = lam
        self.k = k
        self.sigma_scale = sigma_scale
        self.gamma = gamma
        self.radius = radius
        self.delta = delta
        self.random_state = random_state

    def _weights(self, X, y):
        v = self.variant
        if v == "rvfl":
            return np.ones(X.shape[0])
        if v == "ifrvflc":
            return _ha_score(*_mishra_if(X, y, self.gamma, self.radius, self.delta))
        cents = {c: X[y == c].mean(axis=0) for c in (-1.0, 1.0)}
        dist = np.linalg.norm(X - np.where(y[:, None] > 0, cents[1.0], cents[-1.0]), axis=1)
        sigma_bar = dist.mean()
        sigma = self.sigma_scale * (sigma_bar if sigma_bar > 0 else 1.0)
        g = np.exp(-dist ** 2 / (2.0 * sigma ** 2))
        if v == "centroid":
            return g
        idx = neighbours(X, K_MAX)
        if self.k > idx.shape[1]:
            raise ValueError(f"k={self.k} exceeds the available neighbours ({idx.shape[1]}).")
        nu = (y[idx[:, :self.k]] != y[:, None]).mean(axis=1)
        if v == "neighbour":
            return 1.0 - nu
        if v == "product_unscaled":
            return g * (1.0 - nu)
        if v == "score_unscaled":
            return (1.0 + g - nu) / 2.0
        mu = (1.0 - nu) * g
        if v == "product":
            return mu * (1.0 - nu)
        w = (1.0 + mu - nu) / 2.0
        if v == "score":
            return w
        if v == "ha":
            return _ha_score(mu, nu)
        if v == "shuffled":
            return np.random.default_rng(self.random_state).permutation(w)
        raise ValueError(f"unknown variant {v!r}; choose from {VARIANTS}")

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("FuzzyRVFL is a binary classifier.")
        y_ = np.where(y == self.classes_[0], -1.0, 1.0)
        self.scaler_ = MinMaxScaler().fit(X)
        Xn = self.scaler_.transform(X)
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.uniform(-1.0, 1.0, size=(self.n_hidden, Xn.shape[1]))
        self.b_ = rng.uniform(-1.0, 1.0, size=self.n_hidden)
        Z = self._augment(Xn)
        w = np.clip(self._weights(Xn, y_), 1e-12, None)
        self.weights_ = w
        A = Z.T @ (Z * w[:, None]) + self.lam * np.eye(Z.shape[1])
        self.beta_ = np.linalg.solve(A, Z.T @ (w * y_))
        return self

    def _augment(self, Xn):
        return np.hstack([Xn, sigmoid(Xn @ self.W_.T + self.b_)])

    def decision_function(self, X):
        check_is_fitted(self, "beta_")
        return self._augment(self.scaler_.transform(check_array(X))) @ self.beta_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


class EvidenceRVFL(RWIFRVFL):
    """Intermediate formulation: weight 1 - hesitation and an evidence-balance target."""

    def __init__(self, n_hidden=100, lam=1e-3, k=5, sigma_scale=1.0, alpha=1.0,
                 weight="determinacy", target="balance", random_state=42):
        super().__init__(n_hidden=n_hidden, lam=lam, k=k, sigma_scale=sigma_scale, alpha=alpha,
                         symmetric=True, random_state=random_state)
        self.weight = weight
        self.target = target

    def fit(self, X, y):
        Xn, Z, y = self._setup(X, y)
        mu, nu, _, self.sigma_ = ifs_triple(Xn, y, self.k, self.sigma_scale, K_MAX, True)
        mu, nu = (1.0 - self.alpha) + self.alpha * mu, self.alpha * nu
        r = mu + nu
        if self.target == "balance":
            bal = np.divide(mu - nu, r, out=np.ones_like(r), where=r > 1e-12)
        else:
            bal = mu - nu
        w = r if self.weight == "determinacy" else np.ones_like(r)
        self._solve(Z, w, bal * y)
        return self
