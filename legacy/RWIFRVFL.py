
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def sigmoid(x):
    """
    Element-wise sigmoid activation.
    """
    return 1.0 / (1.0 + np.exp(-x))


def init_random_layer(d, L, random_state=42):
    """
    Randomly initialise hidden-layer parameters.

    Parameters
    ----------
    d : int
        Number of input features.

    L : int
        Number of hidden neurons.

    Returns
    -------
    W : ndarray, shape (L, d)
        Weight matrix drawn from U[-1, 1].

    b : ndarray, shape (L,)
        Bias vector drawn from U[-1, 1].
    """

    rng = np.random.default_rng(random_state)
    W = rng.uniform(-1.0, 1.0, size=(L, d))
    b = rng.uniform(-1.0, 1.0,size=(L,))
    return W, b


def augment_features(X, W, b):
    """
    Build RVFL augmented matrix

        Z = [X | H]

    where

        H = sigmoid(X W^T + b)

    Parameters
    ----------
    X : ndarray
        Input feature matrix, shape (N, d).

    W : ndarray
        Hidden-layer weights, shape (L, d).

    b : ndarray
        Hidden-layer biases, shape (L,).

    Returns
    -------
    Z : ndarray
        Augmented feature matrix, shape (N, d + L).
    """

    H = sigmoid(X @ W.T + b)
    Z = np.hstack([X,H])
    return Z


def compute_non_membership(X, y, k=5):
    """
    Compute intuitionistic fuzzy non-membership degree.

    ν_i = fraction of k nearest neighbours having
          a different class label.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Normalised feature matrix.

    y : ndarray, shape (N,)
        Internal labels {-1, +1}.

    k : int
        Number of nearest neighbours.

    Returns
    -------
    nu : ndarray, shape (N,)
        Non-membership values in [0, 1].
    """

    N = X.shape[0]

    if k >= N:
        raise ValueError(
            f"k must be smaller than the number of "
            f"training samples. Got k={k}, N={N}."
        )

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm='auto'
    )

    nn.fit(X)

    _, indices = nn.kneighbors(X)

    neighbour_idx = indices[:, 1:]

    neighbour_labels = y[
        neighbour_idx
    ]

    nu = (
        neighbour_labels != y[:, None]
    ).sum(axis=1) / k

    return nu


def compute_class_centroids(X, y):
    """
    Compute the mean feature vector for each class.

    Parameters
    ----------
    X : ndarray
        Feature matrix.

    y : ndarray
        Internal labels {-1, +1}.

    Returns
    -------
    centroids : dict
        Dictionary {class_label: centroid}.
    """

    return {
        c: X[y == c].mean(axis=0)
        for c in np.unique(y)
    }


def mean_intraclass_distance(centroids):
    """
    Compute mean pairwise distance between class centroids.

    This is used as the default Gaussian spread.
    """

    cls = list(centroids.keys())

    dists = [
        np.linalg.norm(
            centroids[cls[i]]
            -
            centroids[cls[j]]
        )
        for i in range(len(cls))
        for j in range(i + 1, len(cls))
    ]

    return (
        float(np.mean(dists))
        if dists
        else 1.0
    )


def compute_membership(
    X,
    y,
    nu,
    centroids,
    sigma
):
    """
    Original IFRVFLC membership construction.

    μ_i =
        (1 − ν_i)
        exp(
            − ||x_i − c_yi||²
            / (2σ²)
        )

    IMPORTANT
    ---------
    ν must be computed before μ.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Normalised feature matrix.

    y : ndarray, shape (N,)
        Internal labels {-1, +1}.

    nu : ndarray, shape (N,)
        Non-membership degrees.

    centroids : dict
        Class centroids.

    sigma : float
        Gaussian spread.

    Returns
    -------
    mu : ndarray, shape (N,)
        Membership degrees.
    """

    if sigma <= 0:
        raise ValueError(
            "sigma must be greater than zero."
        )

    N = X.shape[0]

    mu = np.zeros(N)

    for i in range(N):

        c = centroids[y[i]]

        dist_sq = np.sum(
            (X[i] - c) ** 2
        )

        mu[i] = (
            (1.0 - nu[i])
            *
            np.exp(
                -dist_sq
                /
                (2.0 * sigma ** 2)
            )
        )

    return np.clip(
        mu,
        0.0,
        1.0
    )


def compute_hesitation(mu, nu):
    """
    Intuitionistic fuzzy hesitation degree.

        π_i = 1 − μ_i − ν_i
    """

    pi = 1.0 - mu - nu

    return np.clip(
        pi,
        0.0,
        1.0
    )


def compute_product_weight(mu, nu):
    """
    Original IFRVFLC product weighting strategy.

        w_i = μ_i (1 − ν_i)

    This function represents the ORIGINAL IFRVFLC weight.
    """

    w = mu * (1.0 - nu)

    return np.clip(
        w,
        0.0,
        1.0
    )


def compute_ifs_score(
    mu,
    nu,
    pi,
    gamma=1.0
):
    """
    Proposed IFS score used by IFRVFLC-SF.

    IMPORTANT
    ---------
    The exact definitions of d^(gamma) and m should match
    the mathematical formulation in your manuscript.

    Current implementation uses:

        m = μ + ν

        d^(gamma) = |μ − ν|^gamma

    and

        S =
        1 − sin[
            π/4 * d^(gamma) * (1 + m)
        ]

    Modify ONLY this function if your manuscript defines
    d^(gamma) or m differently.
    """

    m = mu + nu

    d_gamma = np.abs(
        mu - nu
    ) ** gamma

    score = (
        1.0
        -
        np.sin(
            (np.pi / 4.0)
            *
            d_gamma
            *
            (1.0 + m)
        )
    )

    return np.clip(
        score,
        0.0,
        1.0
    )


def solve_output_weights(
    Z,
    y,
    weights,
    lam
):
    """
    Solve weighted ridge-regression output layer.

    β =
        (Z^T W Z + λI)^(-1)
        Z^T W y

    where W = diag(weights).

    Parameters
    ----------
    Z : ndarray
        Augmented RVFL matrix.

    y : ndarray
        Internal target vector {-1,+1}.

    weights : ndarray
        Sample reliability weights.

    lam : float
        Ridge regularisation parameter.

    Returns
    -------
    beta : ndarray
        Output weights.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    weights = np.clip(
        weights,
        1e-12,
        None
    )

    Z_weighted = Z * weights[:, None]

    A = (
        Z.T @ Z_weighted
        +
        lam * np.eye(
            Z.shape[1]
        )
    )

    b = (
        Z.T
        @
        (weights * y)
    )

    beta = np.linalg.solve(
        A,
        b
    )

    return beta

def compute_ifs_weights(mu, nu):
    """
    Given µᵢ and νᵢ, compute hesitation πᵢ and IFS score weight wᵢ.

    Returns
    -------
    pi : (N,)  hesitation degrees (≥ 0 by construction)
    w  : (N,)  Atanassov score weights in [0, 1]
    """
    pi = 1.0 - mu - nu
    pi = np.clip(pi, 0.0, 1.0)          # floating-point guard

    w  = (1.0 + mu - nu) / 2.0
    w  = np.clip(w,  0.0, 1.0)
    return pi, w

class RWIFRVFL(BaseEstimator, ClassifierMixin):
    """
    Reliability-Weighted Intuitionistic Fuzzy RVFL.

    Parameters
    ----------
    n_hidden    : int   — hidden neurons L  (paper explores {50,100,150,200})
    lam         : float — ridge regularisation λ
    k           : int   — k-NN neighbourhood size
    sigma       : float or None — Gaussian spread; None → auto σ̄
    sigma_scale : float — multiplier on σ̄  (paper tests 0.5, 1, 2, 4)
    random_state: int or None
    normalize   : bool  — apply Min-Max scaling inside the model
    """

    def __init__(self, n_hidden=100, lam=1e-3, k=5,
                 sigma=None, sigma_scale=1.0,
                 random_state=42, normalize=True):
        self.n_hidden     = n_hidden
        self.lam          = lam
        self.k            = k
        self.sigma        = sigma
        self.sigma_scale  = sigma_scale
        self.random_state = random_state
        self.normalize    = normalize

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('RW-IFRVFL is a binary classifier.')

        y_ = np.where(y == self.classes_[0], -1.0, +1.0)

        self.scaler_ = MinMaxScaler() if self.normalize else None
        Xn = self.scaler_.fit_transform(X) if self.normalize else X

        N, d = Xn.shape

        self.W_, self.b_ = init_random_layer(d, self.n_hidden, self.random_state)

        Z = augment_features(Xn, self.W_, self.b_)

        centroids = compute_class_centroids(Xn, y_)
        if self.sigma is not None:
            self.sigma_ = float(self.sigma)
        else:
            sigma_bar   = mean_intraclass_distance(centroids)
            self.sigma_ = self.sigma_scale * (sigma_bar if sigma_bar > 0 else 1.0)

        nu           = compute_non_membership(Xn, y_, self.k)
        mu           = compute_membership(Xn, y_, nu, centroids, self.sigma_)
        pi, w        = compute_ifs_weights(mu, nu)

        self.mu_, self.nu_, self.pi_, self.weights_ = mu, nu, pi, w

        self.beta_ = solve_output_weights(Z, y_, w, self.lam)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X)
        Xn = self.scaler_.transform(X) if self.scaler_ is not None else X
        return augment_features(Xn, self.W_, self.b_) @ self.beta_

    def predict(self, X):
        scores = self.decision_function(X)
        internal = np.sign(scores)
        internal[internal == 0] = 1
        return np.where(internal == -1, self.classes_[0], self.classes_[1])

    def predict_proba(self, X):
        """Soft estimates via sigmoid of the decision score."""
        p1 = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p1, p1])

    def evaluate(self, X, y):
        """Return accuracy, F1, and AUC (the three paper metrics)."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        return {
            'accuracy': accuracy_score(y, y_pred),
            'f1'      : f1_score(y, y_pred, pos_label=self.classes_[1], zero_division=0),
            'auc'     : roc_auc_score(y, y_prob),
        }


print('RWIFRVFL class ready.')