import time
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class TRKMClassifier(BaseEstimator, ClassifierMixin):
    """
    Memory-efficient Scikit-learn compatible TRKM classifier.

    The mathematical formulation is kept unchanged.
    Kernel matrices are computed in batches to avoid
    large memory allocations on datasets such as Adult.
    """

    def __init__(
        self,
        eta1=1.0,
        eta2=1.0,
        lambd1=1.0,
        lambd2=1.0,
        mew=1.0,
        batch_size=1000
    ):
        self.eta1 = eta1
        self.eta2 = eta2
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.mew = mew
        self.batch_size = batch_size


    def _calculate_kA(self, A, C):

        A = np.asarray(A, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)

        n_A = A.shape[0]
        n_C = C.shape[0]

        K = np.empty(
            (n_A, n_C),
            dtype=np.float64
        )

        norm_A_sq = np.sum(A * A, axis=1)
        norm_C_sq = np.sum(C * C, axis=1)


        for start in range(0, n_A, self.batch_size):

            end = min(
                start + self.batch_size,
                n_A
            )

            A_batch = A[start:end]

            distance = (
                norm_A_sq[start:end, None]
                - 2.0 * np.dot(A_batch, C.T)
                + norm_C_sq[None, :]
            )

            distance = np.maximum(
                distance,
                0.0
            )

            K[start:end] = np.exp(
                -distance / (2.0 * self.mew)
            )

        return K


    def fit(self, X, y):

        start = time.time()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same number of samples."
            )

        if self.mew <= 0:
            raise ValueError(
                "mew must be greater than 0."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )


        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError(
                "TRKMClassifier supports binary classification only."
            )


        self._class_to_internal = {
            self.classes_[0]: -1,
            self.classes_[1]: 1
        }

        self._internal_to_class = {
            -1: self.classes_[0],
            1: self.classes_[1]
        }

        y_internal = np.array([
            self._class_to_internal[label]
            for label in y
        ])


        self.X_train_ = X
        self.y_train_ = y_internal

        A = X[y_internal == 1]
        B = X[y_internal == -1]

        if len(A) == 0 or len(B) == 0:
            raise ValueError(
                "Both classes must contain training samples."
            )

        self.A_ = A
        self.B_ = B

        p = A.shape[0]
        q = B.shape[0]


        print(
            f"TRKM: computing K(A,B), shape=({p}, {q})"
        )

        k_A_B = self._calculate_kA(A, B)

        eta_inv1 = 1.0 / self.eta1

        e1 = np.ones(
            (p, 1),
            dtype=np.float64
        )

        e2 = np.ones(
            (q, 1),
            dtype=np.float64
        )

        Rh11 = (
            e1
            + eta_inv1 * np.dot(k_A_B, e2)
        )

        one_array = q * np.ones(
            (Rh11.shape[1], 1),
            dtype=np.float64
        )

        Rh1 = np.vstack(
            (
                Rh11,
                one_array
            )
        )


        print(
            f"TRKM: computing K(A,A), shape=({p}, {p})"
        )

        k_A_A = self._calculate_kA(A, A)

        Rh21 = np.hstack(
            (
                eta_inv1 * k_A_A
                + self.lambd1 * np.ones(
                    (p, p),
                    dtype=np.float64
                ),
                e1
            )
        )

        e1_transposed = e1.T

        zero_array = np.zeros(
            (1, 1),
            dtype=np.float64
        )

        Rh22 = np.hstack(
            (
                e1_transposed,
                zero_array
            )
        )

        Rh2 = np.vstack(
            (
                Rh21,
                Rh22
            )
        )

        Rh2 += 0.05 * np.eye(
            Rh2.shape[1],
            dtype=np.float64
        )


        H1_b1 = np.linalg.solve(
            Rh2,
            Rh1
        )

        self.h1_ = H1_b1[:-1].ravel()
        self.b1_ = float(
            H1_b1[-1, 0]
        )

        del k_A_B
        del k_A_A
        del Rh11
        del Rh21
        del Rh22
        del Rh1
        del Rh2


        print(
            f"TRKM: computing K(B,A), shape=({q}, {p})"
        )

        k_B_A = self._calculate_kA(B, A)

        eta_inv2 = 1.0 / self.eta2

        Rt11 = (
            e2
            + eta_inv2 * np.dot(k_B_A, e1)
        )

        one_array1 = p * np.ones(
            (Rt11.shape[1], 1),
            dtype=np.float64
        )

        Rt1 = -np.vstack(
            (
                Rt11,
                one_array1
            )
        )


        print(
            f"TRKM: computing K(B,B), shape=({q}, {q})"
        )

        k_B_B = self._calculate_kA(B, B)

        Rt21 = np.hstack(
            (
                eta_inv2 * k_B_B
                + self.lambd2 * np.ones(
                    (q, q),
                    dtype=np.float64
                ),
                e2
            )
        )

        e2_transposed = e2.T

        zero_array1 = np.zeros(
            (1, 1),
            dtype=np.float64
        )

        Rt22 = np.hstack(
            (
                e2_transposed,
                zero_array1
            )
        )

        Rt2 = np.vstack(
            (
                Rt21,
                Rt22
            )
        )

        Rt2 += 0.05 * np.eye(
            Rt2.shape[1],
            dtype=np.float64
        )


        H2_b2 = np.linalg.solve(
            Rt2,
            Rt1
        )

        self.h2_ = H2_b2[:-1].ravel()
        self.b2_ = float(
            H2_b2[-1, 0]
        )


        self.n_features_in_ = X.shape[1]

        self.fit_time_ = time.time() - start

        return self


    def decision_function(self, X):

        X = np.asarray(
            X,
            dtype=np.float64
        )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        K_X_A = self._calculate_kA(
            X,
            self.A_
        )

        K_X_B = self._calculate_kA(
            X,
            self.B_
        )

        e1 = np.ones(
            self.A_.shape[0],
            dtype=np.float64
        )

        e2 = np.ones(
            self.B_.shape[0],
            dtype=np.float64
        )

        eta_inv1 = 1.0 / self.eta1
        eta_inv2 = 1.0 / self.eta2


        y1 = (
            eta_inv1
            * np.dot(K_X_A, self.h1_)
            - eta_inv1
            * np.dot(K_X_B, e2)
            + self.b1_
        )


        y2 = (
            eta_inv2
            * np.dot(K_X_B, self.h2_)
            + eta_inv2
            * np.dot(K_X_A, e1)
            + self.b2_
        )

        self._last_y1 = y1
        self._last_y2 = y2

        score = np.abs(y2) - np.abs(y1)

        return score


    def predict(self, X):

        scores = self.decision_function(X)

        internal_pred = np.where(
            scores >= 0,
            1,
            -1
        )

        return np.array([
            self._internal_to_class[p]
            for p in internal_pred
        ])


    def score(self, X, y):

        y_pred = self.predict(X)

        return accuracy_score(
            y,
            y_pred
        )