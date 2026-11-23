"""Neuro-fuzzy RVFL (NF-RVFL) baseline."""
import numpy as np

from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted
)


class NeuroFuzzyRVFLClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_hidden=100,
        random_state=42,
        reg=1e-3
    ):
        self.n_hidden = n_hidden
        self.random_state = random_state
        self.reg = reg

    def _fuzzy_transform(self, X):

        X = np.asarray(X, dtype=float)

        return np.exp(
            -(
                (X - self.fuzzy_means_) ** 2
            ) /
            (
                2 * self.fuzzy_stds_ ** 2
            )
        )

    def _random_features(self, X):

        return np.tanh(
            np.dot(X, self.W_) + self.b_
        )

    def fit(self, X, y):

        X, y = check_X_y(
            X,
            y
        )

        self.n_features_in_ = X.shape[1]

        self.classes_ = np.unique(y)

        self.enc_ = OneHotEncoder(
            sparse_output=False,
            categories='auto'
        )

        Y = self.enc_.fit_transform(
            y.reshape(-1, 1)
        )

        self.fuzzy_means_ = np.mean(
            X,
            axis=0
        )

        self.fuzzy_stds_ = (
            np.std(X, axis=0) + 1e-6
        )

        rng = np.random.RandomState(
            self.random_state
        )

        self.W_ = rng.normal(
            size=(
                self.n_features_in_,
                self.n_hidden
            )
        )

        self.b_ = rng.normal(
            size=self.n_hidden
        )

        X_fuzzy = self._fuzzy_transform(X)

        X_random = self._random_features(X)

        X_concat = np.concatenate(
            [
                X,
                X_fuzzy,
                X_random
            ],
            axis=1
        )

        I = np.eye(
            X_concat.shape[1]
        )

        self.beta_ = np.linalg.pinv(
            X_concat.T @ X_concat
            + self.reg * I
        ) @ X_concat.T @ Y

        self.n_features_combined_ = (
            X_concat.shape[1]
        )

        return self

    def decision_function(self, X):

        check_is_fitted(
            self,
            [
                "beta_",
                "W_",
                "b_",
                "fuzzy_means_",
                "fuzzy_stds_"
            ]
        )

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but {self.__class__.__name__} "
                f"expects {self.n_features_in_} features."
            )

        X_fuzzy = self._fuzzy_transform(X)

        X_random = self._random_features(X)

        X_concat = np.concatenate(
            [
                X,
                X_fuzzy,
                X_random
            ],
            axis=1
        )

        return X_concat @ self.beta_

    def predict_proba(self, X):

        scores = self.decision_function(X)

        scores = (
            scores -
            np.max(
                scores,
                axis=1,
                keepdims=True
            )
        )

        exp_scores = np.exp(scores)

        probabilities = (
            exp_scores /
            np.sum(
                exp_scores,
                axis=1,
                keepdims=True
            )
        )

        return probabilities

    def predict(self, X):

        probabilities = self.predict_proba(X)

        indices = np.argmax(
            probabilities,
            axis=1
        )

        return self.classes_[indices]

    def score(self, X, y):

        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)

        return accuracy_score(
            y,
            y_pred
        )