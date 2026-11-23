"""Neuro-fuzzy RVFL (NF-RVFL) of Sajid, Malik, Tanveer and Suganthan (IEEE TFS, 2024).

Port of the authors' public MATLAB code (github.com/mtanveer1/NeuroFuzzy-RVFL,
Neuro_Fuzzy_RVFL_train.m). As there, the inputs are z-scored, each of the N fuzzy
rules has a centre from k-means and a Gaussian membership of unit width, the rule
firing strengths are normalised over the rules, the TSK consequents are x @ Alpha,
the fuzzy-layer output [F, 0.1] feeds a tansig hidden layer, and the output layer
sees [F, H, x] (direct link) and is solved by ridge regression with one-hot targets.
The only change is numerical: the normalised product of memberships is computed as a
softmax of the summed log-memberships, which is identical in exact arithmetic but does
not underflow in high dimensions.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def firing_strengths(X, centres, width=1.0):
    """Normalised firing strength of every rule: prod_j exp(-(x_j - c_rj)^2 / width), normalised."""
    d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    z = -d2 / width
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NeuroFuzzyRVFLClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=100, reg=1e-3, n_rules=15, centres="kmeans", random_state=42):
        self.n_hidden = n_hidden
        self.reg = reg
        self.n_rules = n_rules
        self.centres = centres
        self.random_state = random_state

    def _features(self, Xs):
        omega = firing_strengths(Xs, self.centres_)
        F = omega * (Xs @ self.alpha_)
        F1 = np.hstack([F, 0.1 * np.ones((F.shape[0], 1))])
        H = np.tanh(F1 @ self.w_hidden_)
        return np.hstack([F, H, Xs])

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        n_rules = min(self.n_rules, Xs.shape[0])
        rng = np.random.default_rng(self.random_state)
        self.alpha_ = rng.uniform(0.0, 1.0, size=(Xs.shape[1], n_rules))
        self.w_hidden_ = rng.uniform(0.0, 1.0, size=(n_rules + 1, self.n_hidden))
        if self.centres == "kmeans":
            self.centres_ = KMeans(n_clusters=n_rules, n_init=1, max_iter=100,
                                   random_state=self.random_state).fit(Xs).cluster_centers_
        elif self.centres == "random":
            self.centres_ = Xs[rng.choice(Xs.shape[0], size=n_rules, replace=False)]
        else:
            raise ValueError(f"unknown centres {self.centres!r}")
        M = self._features(Xs)
        n, p = M.shape
        if p < n:
            self.beta_ = np.linalg.solve(M.T @ M + self.reg * np.eye(p), M.T @ Y)
        else:
            self.beta_ = M.T @ np.linalg.solve(self.reg * np.eye(n) + M @ M.T, Y)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "beta_")
        return self._features(self.scaler_.transform(check_array(X))) @ self.beta_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
