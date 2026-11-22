"""
RW-IFRVFL implemented directly from the manuscript: Section 3 and Algorithm 1
for the model, Section 4.4 for the spread sigma, Table 5 for the search space.

Training
    1  min-max scale the features to [0, 1] (fitted on the training data)
    2  draw W (L x d) and b (L) from U[-1, 1]
    3  h_i = g(W x_i + b) with g the sigmoid;  z_i = [x_i; h_i]
    4  class centroids c_{+1}, c_{-1}
    5  nu_i = |{x_j in N_k(x_i) : y_j != y_i}| / k        Euclidean k-NN, x_i excluded
    6  mu_i = (1 - nu_i) exp(-||x_i - c_{y_i}||^2 / (2 sigma^2))
    7  pi_i = 1 - mu_i - nu_i                              >= 0 by construction
    8  w_i  = (1 + mu_i - nu_i) / 2                         IFS score function
    9  beta = (Z^T D Z + lambda I)^{-1} Z^T D y,  D = diag(w_1, ..., w_N)
Inference
       y_hat = sign(beta^T z)

sigma = sigma_scale * sigma_bar, where sigma_bar is the mean intra-class centroid
distance (Section 4.4): the mean of ||x_i - c_{y_i}|| over the training samples.

EvidenceRVFL keeps steps 1-4 and replaces steps 5-9 (proposed revision):
    with symmetric=True the neighbourhood evidence <1 - f_i, f_i> (f_i = share of
    the k neighbours with another label) is discounted by the typicality
    g_i = exp(-||x_i - c_{y_i}||^2 / (2 sigma^2)), as in Shafer's discounting:
        mu_i = g_i (1 - f_i)   (as in step 6),   nu_i = g_i f_i,   pi_i = 1 - g_i;
    symmetric=False keeps the manuscript's nu_i = f_i (ablation). Then the annotation,
    the certain IF number <1, 0>, is fused with the data evidence using a trust alpha,
        mu~_i = (1 - alpha) + alpha mu_i,   nu~_i = alpha nu_i,   pi~_i = alpha pi_i,
    and the two parts of the fused number are used separately:
        reliability weight   c_i = mu~_i + nu~_i = 1 - pi~_i
        corrected target     t_i = y_i (mu~_i - nu~_i) / (mu~_i + nu~_i)
        beta = (Z^T C Z + lambda I)^{-1} Z^T C t,   C = diag(c_1, ..., c_N).
    alpha = 0 gives plain RVFL; t_i changes sign (label correction) exactly when
    alpha (nu_i - mu_i) > 1 - alpha, i.e. alpha g_i (2 f_i - 1) > 1 - alpha when symmetric.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from rvfl_family import _neighbours     # Euclidean k-NN, self excluded, cached per matrix

# Table 5 of the manuscript
SEARCH_SPACE = {
    "n_hidden": [50, 100, 150, 200, 300],
    "lam": [1e-4, 1e-3, 1e-2, 1e-1],
    "k": [3, 5, 7, 9],
    "sigma_scale": [0.5, 1.0, 1.5, 2.0],
}


def _sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))       # 1 / (1 + e^-t) without overflow


def ifs_triple(Xn, y, k, sigma_scale, centroid=True, symmetric=False):
    """Steps 4-7: (mu, nu, pi) and sigma for scaled features Xn, labels y in {-1, +1}."""
    _, idx = _neighbours(Xn)
    if k > idx.shape[1]:
        raise ValueError(f"k={k} exceeds the cached neighbours ({idx.shape[1]}).")
    nu = (y[idx[:, :k]] != y[:, None]).mean(axis=1)
    if not centroid:                             # ablation: neighbourhood evidence only
        return 1.0 - nu, nu, np.zeros_like(nu), np.inf
    cent = {c: Xn[y == c].mean(axis=0) for c in (-1.0, 1.0)}
    dist = np.linalg.norm(Xn - np.where(y[:, None] > 0, cent[1.0], cent[-1.0]), axis=1)
    sigma_bar = dist.mean()
    sigma = sigma_scale * (sigma_bar if sigma_bar > 0 else 1.0)
    g = np.exp(-dist ** 2 / (2.0 * sigma ** 2))
    mu = (1.0 - nu) * g
    if symmetric:                                # typicality discounts both kinds of evidence
        return mu, g * nu, 1.0 - g, sigma
    return mu, nu, 1.0 - mu - nu, sigma


class RWIFRVFL(BaseEstimator, ClassifierMixin):
    """Reliability-weighted intuitionistic fuzzy RVFL (binary)."""

    def __init__(self, n_hidden=100, lam=1e-3, k=5, sigma_scale=1.0, random_state=42):
        self.n_hidden = n_hidden
        self.lam = lam
        self.k = k
        self.sigma_scale = sigma_scale
        self.random_state = random_state

    def _augment(self, Xn):
        return np.hstack([Xn, _sigmoid(Xn @ self.W_.T + self.b_)])

    def _setup(self, X, y):
        """Steps 1-3; returns scaled X, augmented Z and labels in {-1, +1}."""
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

    def fit(self, X, y):
        Xn, Z, y = self._setup(X, y)
        mu, nu, pi, self.sigma_ = ifs_triple(Xn, y, self.k, self.sigma_scale)
        self.mu_, self.nu_, self.pi_ = mu, nu, pi
        self.weights_ = (1.0 + mu - nu) / 2.0                   # step 8
        self._solve(Z, self.weights_, y)                         # step 9
        return self

    def decision_function(self, X):
        check_is_fitted(self, "beta_")
        return self._augment(self.scaler_.transform(check_array(X))) @ self.beta_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


class EvidenceRVFL(RWIFRVFL):
    """Proposed revision: IF evidence gives the sample weight and a corrected target.

    weight="none", target="label" / "pignistic", centroid=False and
    symmetric=False are ablations.
    """

    def __init__(self, n_hidden=100, lam=1e-3, k=5, sigma_scale=1.0, alpha=1.0,
                 weight="determinacy", target="balance", centroid=True, symmetric=True,
                 random_state=42):
        super().__init__(n_hidden, lam, k, sigma_scale, random_state)
        self.alpha = alpha
        self.weight = weight
        self.target = target
        self.centroid = centroid
        self.symmetric = symmetric

    def fit(self, X, y):
        Xn, Z, y = self._setup(X, y)
        mu, nu, pi, self.sigma_ = ifs_triple(Xn, y, self.k, self.sigma_scale,
                                             self.centroid, self.symmetric)
        a = self.alpha
        mu_f, nu_f = (1.0 - a) + a * mu, a * nu                  # fused IF number
        self.mu_, self.nu_, self.pi_ = mu_f, nu_f, a * pi
        r = mu_f + nu_f                                          # determinacy = 1 - pi~
        if self.target == "balance":
            bal = np.divide(mu_f - nu_f, r, out=np.ones_like(r), where=r > 1e-12)
        elif self.target == "pignistic":
            bal = mu_f - nu_f
        elif self.target == "label":
            bal = np.ones_like(r)
        else:
            raise ValueError(f"unknown target {self.target!r}")
        self.targets_ = bal * y
        self.weights_ = r if self.weight == "determinacy" else np.ones_like(r)
        self._solve(Z, self.weights_, self.targets_)
        return self
