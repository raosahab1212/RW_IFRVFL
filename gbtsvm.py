"""
Granular-Ball Twin SVM (linear), scikit-learn wrapper.

Follows the authors' public reference implementation: the QP linear terms use
the granular-ball radii, and a freshly split ball is re-examined before moving on.
"""

import hashlib
from collections import Counter

import numpy as np
from cvxopt import matrix, solvers
from scipy import linalg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

solvers.options["show_progress"] = False


def _ball_stats(Xb, yb):
    count = Counter(yb)
    label = max(count, key=count.get)
    center = Xb.mean(axis=0)
    radius = np.linalg.norm(Xb - center, axis=1).mean()
    return label, count[label] / len(yb), center, radius


_BALL_CACHE = {}


def granular_balls(X, y, purity=0.925, min_sample=1, min_ball_size=4, seed=0):
    """Return centers, radii, labels of the granular balls (reference algorithm).

    Cached: tuning refits the same training matrix with different box
    constraints (d1, d2), which do not affect the balls.
    """
    key = (hashlib.md5(np.ascontiguousarray(X).view(np.uint8)).hexdigest(),
           hashlib.md5(np.ascontiguousarray(y).view(np.uint8)).hexdigest(),
           X.shape, purity, min_sample, min_ball_size, seed)
    if key not in _BALL_CACHE:
        if len(_BALL_CACHE) > 256:
            _BALL_CACHE.clear()
        _BALL_CACHE[key] = _granular_balls(X, y, purity, min_sample, min_ball_size, seed)
    return _BALL_CACHE[key]


def _granular_balls(X, y, purity, min_sample, min_ball_size, seed):
    balls = [np.arange(len(y))]
    i = 0
    while i < len(balls):
        idx = balls[i]
        _, pur, _, _ = _ball_stats(X[idx], y[idx])
        if pur < purity and len(idx) > min_sample:
            lab = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X[idx]).labels_
            if (lab == 0).any() and (lab == 1).any():
                a, b = idx[lab == 0], idx[lab == 1]
            else:                                   # reference fallback
                a, b = idx[:1], idx[1:]
            balls[i] = a
            balls.append(b)                         # do NOT advance i
        else:
            i += 1
    balls = [b for b in balls if len(b) >= min_ball_size]
    stats = [_ball_stats(X[b], y[b]) for b in balls]
    labels = np.array([s[0] for s in stats], dtype=float)
    centers = np.array([s[2] for s in stats])
    radii = np.array([s[3] for s in stats])
    return centers, radii, labels


class GBTSVMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, d1=1e-5, d2=1e-5, eps1=0.05, eps2=0.05, purity=0.925,
                 min_sample=1, min_ball_size=4, random_state=0):
        self.d1 = d1
        self.d2 = d2
        self.eps1 = eps1
        self.eps2 = eps2
        self.purity = purity
        self.min_sample = min_sample
        self.min_ball_size = min_ball_size
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("GBTSVM is a binary classifier.")
        y_ = np.where(y == self.classes_[1], 1.0, -1.0)
        self.scaler_ = MinMaxScaler().fit(X)
        Xn = self.scaler_.transform(X)

        C, R, L = granular_balls(Xn, y_, self.purity, self.min_sample,
                                 self.min_ball_size, self.random_state)
        self.n_balls_ = len(L)
        if not ((L == 1).any() and (L == -1).any()):
            raise ValueError("Ball generation left only one class; lower min_ball_size.")
        C1, R1 = C[L == 1], R[L == 1]
        C2, R2 = C[L == -1], R[L == -1]
        H = np.hstack([C1, np.ones((len(C1), 1))])
        G = np.hstack([C2, np.ones((len(C2), 1))])

        HHG = linalg.solve(H.T @ H + self.eps1 * np.eye(H.shape[1]), G.T)
        K1 = G @ HHG
        a1 = self._box_qp((K1 + K1.T) / 2, -(1.0 + R2), self.d1)
        z1 = -HHG @ a1
        QQP = linalg.solve(G.T @ G + self.eps2 * np.eye(G.shape[1]), H.T)
        K2 = H @ QQP
        a2 = self._box_qp((K2 + K2.T) / 2, -(1.0 + R1), self.d2)
        z2 = QQP @ a2

        self.w1_, self.b1_ = z1[:-1], z1[-1]
        self.w2_, self.b2_ = z2[:-1], z2[-1]
        return self

    @staticmethod
    def _box_qp(P, q, ub):
        m = len(q)
        Gc = np.vstack([np.eye(m), -np.eye(m)])
        hc = np.concatenate([ub * np.ones(m), np.zeros(m)])
        sol = solvers.qp(matrix(P, tc="d"), matrix(q, tc="d"),
                         matrix(Gc, tc="d"), matrix(hc, tc="d"))
        return np.asarray(sol["x"]).ravel()

    def decision_function(self, X):
        """Positive => positive class. Same quantities the reference uses to predict."""
        check_is_fitted(self, "w1_")
        Xn = self.scaler_.transform(check_array(X))
        return np.abs(Xn @ self.w2_ + self.b2_) - np.abs(Xn @ self.w1_ + self.b1_)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])
