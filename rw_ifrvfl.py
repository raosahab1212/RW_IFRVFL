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

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("RW-IFRVFL is a binary classifier.")
        y = np.where(y == self.classes_[1], 1.0, -1.0)

        # steps 1-3
        self.scaler_ = MinMaxScaler().fit(X)
        Xn = self.scaler_.transform(X)
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.uniform(-1.0, 1.0, size=(self.n_hidden, Xn.shape[1]))
        self.b_ = rng.uniform(-1.0, 1.0, size=self.n_hidden)
        Z = self._augment(Xn)

        # step 4 and the spread
        cent = {c: Xn[y == c].mean(axis=0) for c in (-1.0, 1.0)}
        dist = np.linalg.norm(Xn - np.where(y[:, None] > 0, cent[1.0], cent[-1.0]), axis=1)
        sigma_bar = dist.mean()
        self.sigma_ = self.sigma_scale * (sigma_bar if sigma_bar > 0 else 1.0)

        # steps 5-8
        _, idx = _neighbours(Xn)
        if self.k > idx.shape[1]:
            raise ValueError(f"k={self.k} exceeds the cached neighbours ({idx.shape[1]}).")
        nu = (y[idx[:, :self.k]] != y[:, None]).mean(axis=1)
        mu = (1.0 - nu) * np.exp(-dist ** 2 / (2.0 * self.sigma_ ** 2))
        self.mu_, self.nu_, self.pi_ = mu, nu, 1.0 - mu - nu
        w = self.weights_ = (1.0 + mu - nu) / 2.0

        # step 9
        A = Z.T @ (Z * w[:, None]) + self.lam * np.eye(Z.shape[1])
        self.beta_ = np.linalg.solve(A, Z.T @ (w * y))
        return self

    def decision_function(self, X):
        check_is_fitted(self, "beta_")
        return self._augment(self.scaler_.transform(check_array(X))) @ self.beta_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])
