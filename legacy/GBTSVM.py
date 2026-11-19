import time
import numpy as np
from collections import Counter

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy import linalg
from cvxopt import matrix, solvers


class GranularBall:
    """Represents one granular ball."""

    def __init__(self, data):
        self.data = data[:, :]

        self.data_no_label = data[:, :-2]

        self.num, self.dim = self.data_no_label.shape

        self.center = self.data_no_label.mean(axis=0)

        self.label, self.purity, self.r = (
            self._get_label_purity_radius()
        )

    def _get_label_purity_radius(self):
        count = Counter(self.data[:, -2])

        label = max(count, key=count.get)
        purity = count[label] / self.num

        arr = self.data_no_label - self.center
        distances = np.sqrt(np.sum(arr ** 2, axis=1))

        radius = np.mean(distances)

        return label, purity, radius

    def split_2balls(self):
        """Split a granular ball using K-Means."""

        if self.num < 2:
            return self, None

        kmeans = KMeans(
            n_clusters=2,
            random_state=0,
            n_init=10
        )

        labels = kmeans.fit_predict(self.data_no_label)

        if np.sum(labels == 0) > 0 and np.sum(labels == 1) > 0:

            ball1 = GranularBall(
                self.data[labels == 0, :]
            )

            ball2 = GranularBall(
                self.data[labels == 1, :]
            )

            return ball1, ball2

        ball1 = GranularBall(self.data[:1, :])
        ball2 = GranularBall(self.data[1:, :])

        return ball1, ball2


class GBList:

    def __init__(self, data):
        self.data = data[:, :]
        self.granular_balls = [
            GranularBall(self.data)
        ]

    def init_granular_balls(self, purity=0.925, min_sample=1):

        i = 0

        while i < len(self.granular_balls):

            ball = self.granular_balls[i]

            if (
                ball.purity < purity
                and ball.num > min_sample
            ):

                ball1, ball2 = ball.split_2balls()

                self.granular_balls[i] = ball1

                if ball2 is not None:
                    self.granular_balls.append(ball2)

            i += 1

    def del_ball(self, purity=0.0, num_data=0):

        self.granular_balls = [
            ball
            for ball in self.granular_balls
            if (
                ball.purity >= purity
                and ball.num >= num_data
            )
        ]

    def get_centers(self):

        return np.array([
            ball.center
            for ball in self.granular_balls
        ])

    def get_radii(self):

        return np.array([
            ball.r
            for ball in self.granular_balls
        ])

    def get_labels(self):

        return np.array([
            ball.label
            for ball in self.granular_balls
        ])


class GBTSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible Granular-Ball Twin SVM classifier.

    Parameters
    ----------
    d1 : float
        Regularization parameter for first QP.

    d2 : float
        Regularization parameter for second QP.

    purity : float
        Purity threshold used for granular-ball generation.

    min_sample : int
        Minimum number of samples required for splitting.

    min_ball_size : int
        Minimum number of samples retained in a granular ball.

    eps1 : float
        Numerical regularization for first QP.

    eps2 : float
        Numerical regularization for second QP.
    """

    def __init__(
        self,
        d1=1e-5,
        d2=1e-5,
        purity=0.925,
        min_sample=1,
        min_ball_size=4,
        eps1=0.05,
        eps2=0.05
    ):

        self.d1 = d1
        self.d2 = d2
        self.purity = purity
        self.min_sample = min_sample
        self.min_ball_size = min_ball_size
        self.eps1 = eps1
        self.eps2 = eps2


    def _generate_balls(self, X, y):

        n_samples = X.shape[0]

        indices = np.arange(n_samples).reshape(-1, 1)

        data = np.hstack(
            (
                X,
                y.reshape(-1, 1),
                indices
            )
        )

        gb = GBList(data)

        gb.init_granular_balls(
            purity=self.purity,
            min_sample=self.min_sample
        )

        gb.del_ball(
            num_data=self.min_ball_size
        )

        centers = gb.get_centers()
        radii = gb.get_radii()
        labels = gb.get_labels()

        ball_data = np.hstack(
            (
                centers,
                radii.reshape(-1, 1),
                labels.reshape(-1, 1)
            )
        )

        return ball_data


    def _train_gbtsvm(self, train_data):

        A = train_data[
            train_data[:, -1] == 1,
            :-1
        ]

        B = train_data[
            train_data[:, -1] == -1,
            :-1
        ]

        if len(A) == 0 or len(B) == 0:
            raise ValueError(
                "Both classes must contain at least one granular ball."
            )

        C1 = A[:, :-1]
        C2 = B[:, :-1]

        R1 = train_data[
            train_data[:, -1] == 1,
            -1
        ]

        R2 = train_data[
            train_data[:, -1] == -1,
            -1
        ]

        m1 = A.shape[0]
        m2 = B.shape[0]

        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))

        H1 = np.hstack((C1, e1))
        G1 = np.hstack((C2, e2))


        HH1 = (
            H1.T @ H1
            + self.eps1 * np.eye(H1.shape[1])
        )

        HHG = linalg.solve(
            HH1,
            G1.T
        )

        kerH1 = G1 @ HHG

        kerH1 = (
            kerH1 + kerH1.T
        ) / 2

        m1 = kerH1.shape[0]

        e3 = np.ones(m1)

        R11 = -(e3 + R2)

        vlb = np.zeros((m1, 1))
        vub = self.d1 * np.ones((m1, 1))

        G = np.vstack(
            (
                np.eye(m1),
                -np.eye(m1)
            )
        )

        h = np.vstack(
            (
                vub,
                -vlb
            )
        )

        solvers.options["show_progress"] = False

        alpha1 = solvers.qp(
            matrix(kerH1, tc="d"),
            matrix(R11, tc="d"),
            matrix(G, tc="d"),
            matrix(h, tc="d")
        )

        alphasol1 = np.asarray(
            alpha1["x"]
        )

        z = -HHG @ alphasol1

        z = z.ravel()

        self.w1_ = z[:-1]
        self.b1_ = z[-1]


        QQ = G1.T @ G1

        QQ = (
            QQ
            + self.eps2 * np.eye(QQ.shape[1])
        )

        QQP = linalg.solve(
            QQ,
            H1.T
        )

        kerH2 = H1 @ QQP

        kerH2 = (
            kerH2 + kerH2.T
        ) / 2

        m2 = kerH2.shape[0]

        e4 = np.ones(m2)

        R22 = -(e4 + R1)

        vlb = np.zeros((m2, 1))
        vub = self.d2 * np.ones((m2, 1))

        cd = np.vstack(
            (
                np.eye(m2),
                -np.eye(m2)
            )
        )

        vcd = np.vstack(
            (
                vub,
                -vlb
            )
        )

        alpha2 = solvers.qp(
            matrix(kerH2, tc="d"),
            matrix(R22, tc="d"),
            matrix(cd, tc="d"),
            matrix(vcd, tc="d")
        )

        alphasol2 = np.asarray(
            alpha2["x"]
        )

        z = QQP @ alphasol2

        z = z.ravel()

        self.w2_ = z[:-1]
        self.b2_ = z[-1]


    def fit(self, X, y):

        start = time.time()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError(
                "GBTSVMClassifier supports binary classification only."
            )

        self.classes_ = classes

        self._class_to_internal = {
            classes[0]: -1,
            classes[1]: 1
        }

        self._internal_to_class = {
            -1: classes[0],
            1: classes[1]
        }

        y_internal = np.array([
            self._class_to_internal[label]
            for label in y
        ])

        ball_data = self._generate_balls(
            X,
            y_internal
        )

        if len(ball_data) < 2:
            raise ValueError(
                "Granular-ball generation produced fewer than "
                "two balls. Try reducing min_ball_size or purity."
            )

        self.ball_centers_ = ball_data[:, :-2]
        self.ball_radii_ = ball_data[:, -2]
        self.ball_labels_ = ball_data[:, -1]

        self._train_gbtsvm(ball_data)

        self.n_features_in_ = X.shape[1]

        self.fit_time_ = time.time() - start

        return self


    def decision_function(self, X):

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"got {X.shape[1]}."
            )

        y1 = X @ self.w1_ + self.b1_
        y2 = X @ self.w2_ + self.b2_

        score = np.where(
            np.abs(y1) <= np.abs(y2),
            y1,
            -y2
        )

        return score


    def predict(self, X):

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        y1 = X @ self.w1_ + self.b1_
        y2 = X @ self.w2_ + self.b2_

        internal_prediction = np.where(
            np.abs(y1) <= np.abs(y2),
            1,
            -1
        )

        return np.array([
            self._internal_to_class[p]
            for p in internal_prediction
        ])


    def score(self, X, y):

        y_pred = self.predict(X)

        return accuracy_score(y, y_pred)