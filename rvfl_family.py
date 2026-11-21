"""
RVFL family used in the RW-IFRVFL experiments.

One estimator, `FuzzyRVFL`, covers plain RVFL, the proposed RW-IFRVFL,
the IFRVFLC baseline and every ablation variant, so that all of them share
exactly the same random layer, scaling, solver and neighbour search.
Only the sample-weight construction differs between variants.

Notation (all computed on the min-max scaled *training* data only):
    g_i   = exp(-||x_i - c_{y_i}||^2 / (2 sigma^2))     Gaussian centroid similarity
    nu_i  = share of the k nearest neighbours with a different label
    sigma = sigma_scale * ||c_+ - c_-||                  (distance between the two
                                                          class centroids)

Variants (`variant=`):
    rvfl          w = 1
    fuzzy         w = g                                   classical fuzzy weight
    nonmem        w = 1 - nu                              neighbourhood only
    pw_unscaled   w = g (1 - nu)                          unscaled product weight
    sf_unscaled   w = (1 + g - nu) / 2                    IFRVFLC-SF
    pw_scaled     mu = (1-nu) g,  w = mu (1 - nu)
    rw            mu = (1-nu) g,  w = (1 + mu - nu) / 2   proposed RW-IFRVFL
    rw_ha         mu = (1-nu) g,  w = Ha et al. (2013) IF score
    rw_shuffled   RW weights randomly permuted across samples (control)
    ifrvflc       Mishra, Gupta & Hazarika (2023), Eqs. 22-24:
                  mu  = 1 - ||phi(x) - C_y|| / (r_y + delta)   Gaussian-kernel space,
                        k(u, v) = exp(-gamma ||u - v||^2), r_y = class radius
                  rho = share of other-class points within input-space radius
                        `radius` (self included, as in Eq. 23)
                  nu  = (1 - mu) rho
                  w   = Ha score
"""

import hashlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

VARIANTS = (
    "rvfl", "fuzzy", "nonmem", "pw_unscaled", "sf_unscaled",
    "pw_scaled", "rw", "rw_ha", "rw_shuffled", "ifrvflc",
)

K_MAX = 15  # neighbours cached per training matrix; covers every k in the grids


_NN_CACHE = {}
_NN_CACHE_MAX = 64


def _neighbours(X):
    """Return (dist, idx) of the K_MAX nearest *other* samples of each row."""
    key = hashlib.md5(np.ascontiguousarray(X).view(np.uint8)).hexdigest() + str(X.shape)
    hit = _NN_CACHE.get(key)
    if hit is not None:
        return hit
    n = X.shape[0]
    kq = min(K_MAX + 1, n)
    dist, idx = NearestNeighbors(n_neighbors=kq).fit(X).kneighbors(X)
    # Drop the sample itself. With duplicate rows the self index is not always
    # in column 0, so remove it explicitly and keep the first K_MAX others.
    out_d = np.empty((n, kq - 1))
    out_i = np.empty((n, kq - 1), dtype=int)
    rows = np.arange(n)
    for r in rows:
        keep = idx[r] != r
        if keep.all():          # self not returned (many exact duplicates)
            keep[-1] = False
        out_i[r] = idx[r][keep]
        out_d[r] = dist[r][keep]
    if len(_NN_CACHE) >= _NN_CACHE_MAX:
        _NN_CACHE.clear()
    _NN_CACHE[key] = (out_d, out_i)
    return out_d, out_i


_D2_CACHE = {}
_D2_CACHE_N = 6000          # keep full distance matrices only for tuning-sized data
_BLOCK = 10_000_000         # elements per distance block for large data


def _sq_dist_blocks(X):
    """Yield (rows, squared distances of those rows to all of X)."""
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
    """Membership and non-membership of Mishra et al. (2023), Eqs. 22-23."""
    n = X.shape[0]
    same_k = np.empty(n)            # sum of k(x_i, x_j) over same-class j
    cnt = np.empty(n)
    het = np.empty(n)
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
        s = same_k[m] / m.sum()                      # mean k(x_i, class members)
        dist = np.sqrt(np.maximum(1.0 - 2.0 * s + s.mean(), 0.0))
        mu[m] = 1.0 - dist / (dist.max() + delta)
    rho = het / np.maximum(cnt, 1)
    return mu, (1.0 - mu) * rho


def _non_membership(X, y, k):
    _, idx = _neighbours(X)
    if k > idx.shape[1]:
        raise ValueError(f"k={k} exceeds cached neighbours ({idx.shape[1]}).")
    return (y[idx[:, :k]] != y[:, None]).mean(axis=1)


def _ha_score(mu, nu):
    """Ha et al. (2013) / Rezvani et al. (2019) intuitionistic fuzzy score."""
    s = np.where(nu == 0, mu, 0.0)
    mid = (nu > 0) & (mu > nu)
    s[mid] = (1.0 - nu[mid]) / (2.0 - mu[mid] - nu[mid])
    return s


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))   # overflow-safe


class FuzzyRVFL(BaseEstimator, ClassifierMixin):
    """Weighted RVFL with pluggable sample-weight construction (binary)."""

    def __init__(self, variant="rw", n_hidden=100, lam=1e-3, k=5,
                 sigma_scale=1.0, gamma=1.0, radius=0.5, delta=1e-4,
                 random_state=42):
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
        n = X.shape[0]
        if v == "rvfl":
            return np.ones(n)

        if v == "ifrvflc":
            mu, nu = _mishra_if(X, y, self.gamma, self.radius, self.delta)
            self.mu_, self.nu_ = mu, nu
            return _ha_score(mu, nu)

        cents = {c: X[y == c].mean(axis=0) for c in (-1.0, 1.0)}
        dist_c = np.linalg.norm(X - np.where(y[:, None] > 0, cents[1.0], cents[-1.0]), axis=1)

        sep = np.linalg.norm(cents[1.0] - cents[-1.0])
        self.sigma_ = self.sigma_scale * (sep if sep > 0 else 1.0)
        g = np.exp(-dist_c ** 2 / (2.0 * self.sigma_ ** 2))

        if v == "fuzzy":
            return g

        nu = _non_membership(X, y, self.k)
        if v == "nonmem":
            return 1.0 - nu
        if v == "pw_unscaled":
            return g * (1.0 - nu)
        if v == "sf_unscaled":
            return (1.0 + g - nu) / 2.0

        mu = (1.0 - nu) * g                      # IFS-valid scaled membership
        self.mu_, self.nu_ = mu, nu
        self.pi_ = 1.0 - mu - nu
        if v == "pw_scaled":
            return mu * (1.0 - nu)
        w = (1.0 + mu - nu) / 2.0
        if v == "rw":
            return w
        if v == "rw_ha":
            return _ha_score(mu, nu)
        if v == "rw_shuffled":
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
        d = Xn.shape[1]

        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.uniform(-1.0, 1.0, size=(self.n_hidden, d))
        self.b_ = rng.uniform(-1.0, 1.0, size=self.n_hidden)
        Z = self._augment(Xn)

        w = np.clip(self._weights(Xn, y_), 1e-12, None)
        self.weights_ = w
        A = Z.T @ (Z * w[:, None]) + self.lam * np.eye(Z.shape[1])
        self.beta_ = np.linalg.solve(A, Z.T @ (w * y_))
        return self

    def _augment(self, Xn):
        return np.hstack([Xn, _sigmoid(Xn @ self.W_.T + self.b_)])

    def decision_function(self, X):
        check_is_fitted(self, "beta_")
        Xn = self.scaler_.transform(check_array(X))
        return self._augment(Xn) @ self.beta_

    def predict(self, X):
        s = self.decision_function(X)
        return np.where(s >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])
