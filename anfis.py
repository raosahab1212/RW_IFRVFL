"""ANFIS baseline: fuzzy c-means rules with hybrid (gradient and least-squares) learning."""
import numpy as np
import torch


def calculateMfParams(X, L):
    """
    Calculate Gaussian Membership Function parameters.

    Parameters
    ----------
    X : torch.Tensor
        Shape: (num_samples, num_features)
    L : int
        Number of membership functions per feature

    Returns
    -------
    dict
        Gaussian MF parameters
    """

    params = {}

    for i in range(X.shape[1]):

        feature = X[:, i]

        mean = torch.mean(feature)
        std = torch.std(feature)

        std = torch.clamp(std, min=1e-6)

        centers = [
            mean + (k - (L - 1) / 2) * std
            for k in range(L)
        ]

        sigmas = [
            std / (1 + abs(k - (L - 1) / 2))
            for k in range(L)
        ]

        params[f"Feature_{i}"] = {
            f"mu_{k+1}": {
                "c": centers[k].item(),
                "sigma": max(sigmas[k].item(), 1e-6)
            }
            for k in range(L)
        }

    return params


def normalizedWeights(membershipValues):
    """
    Calculate normalized fuzzy rule strengths.

    Parameters
    ----------
    membershipValues : torch.Tensor
        Shape: (samples, features, rules)

    Returns
    -------
    torch.Tensor
        Shape: (samples, rules)
    """

    weights = torch.prod(membershipValues, dim=1)

    weights_sum = torch.sum(weights, dim=1, keepdim=True)

    weights_sum = torch.clamp(weights_sum, min=1e-12)

    normalized_weights = weights / weights_sum

    return normalized_weights


def gaussian_membership(X, c, sigma):
    """
    Gaussian membership function.
    """

    X = torch.as_tensor(X, dtype=torch.float32)
    c = torch.as_tensor(c, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)

    sigma = torch.clamp(sigma, min=1e-6)

    result = torch.exp(
        -((X - c) ** 2) / (2 * sigma ** 2)
    )

    return result


def calculate_A(X, weights):

    X_copy = X.clone()

    ones = torch.ones(X.size(0), 1)

    newX = torch.cat((X_copy, ones), dim=1)

    A = []

    for i in range(newX.shape[0]):

        row_result = torch.cat([
            newX[i] * weights[i, j]
            for j in range(weights.shape[1])
        ])

        A.append(row_result)

    return torch.stack(A)


class HybridAlgorithmClassifier:

    def __init__(
        self,
        X,
        y,
        L,
        eta=0.01,
        num_iterations=100,
        num_cluster=None,
        fuzziness_index=2,
        a=1,
        epsilon=1e-6,
        lambda_reg=1e-4,
        lambda1=1e-2,
        lambda2=1e-1
    ):

        self.L = L
        self.eta = eta
        self.num_iterations = num_iterations
        self.num_cluster = num_cluster
        self.fuzziness_index = fuzziness_index
        self.a = a
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.lambda1 = lambda1
        self.lambda2 = lambda2


        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        y = np.asarray(y).reshape(-1)

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )


        self.classes_ = np.unique(y)

        self.n_classes = len(self.classes_)

        self.y_encoded = np.searchsorted(
            self.classes_,
            y
        )

        self.y = torch.tensor(
            self.y_encoded,
            dtype=torch.long
        )


        self.Y_onehot = torch.nn.functional.one_hot(
            self.y,
            num_classes=self.n_classes
        ).float()


        self.mf_params = calculateMfParams(
            self.X,
            self.L
        )

        self.D = None

        self.A = None
        self.weights = None


    def _membership_values(self, X, params=None):

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        if params is None:
            params = self.mf_params

        num_samples, num_features = X.shape

        membership = torch.zeros(
            num_samples,
            num_features,
            self.L
        )

        for i in range(num_samples):

            for j in range(num_features):

                feature_params = params[
                    f"Feature_{j}"
                ]

                for k, mu_params in enumerate(
                    feature_params.values()
                ):

                    c = mu_params["c"]
                    sigma = mu_params["sigma"]

                    membership[i, j, k] = (
                        gaussian_membership(
                            X[i, j],
                            c,
                            sigma
                        )
                    )

        return membership


    def _rule_weights(self, membership):

        return normalizedWeights(
            membership
        )


    def _build_A(self, X, weights):

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        return calculate_A(
            X,
            weights
        )


    def _solve_D(self, A):

        A_np = A.detach().numpy()

        Y_np = self.Y_onehot.detach().numpy()


        ATA = A_np.T @ A_np

        I = np.eye(
            ATA.shape[0]
        )

        D = np.linalg.solve(
            ATA + self.lambda_reg * I,
            A_np.T @ Y_np
        )

        return torch.tensor(
            D,
            dtype=torch.float32
        )


    def fit(self):


        membership = self._membership_values(
            self.X,
            self.mf_params
        )


        weights = self._rule_weights(
            membership
        )


        A = self._build_A(
            self.X,
            weights
        )

        self.A = A
        self.weights = weights


        self.D = self._solve_D(A)


        logits = A @ self.D

        self.train_logits = logits


        self.train_probabilities = torch.softmax(
            logits,
            dim=1
        )


        predicted_encoded = torch.argmax(
            self.train_probabilities,
            dim=1
        )

        self.y_train_pred = self.classes_[
            predicted_encoded.numpy()
        ]

        return self


    def forward(self):

        return self.fit()


    def predict_proba(self, X):

        if self.D is None:
            self.fit()

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        membership = self._membership_values(
            X,
            self.mf_params
        )

        weights = self._rule_weights(
            membership
        )

        A = self._build_A(
            X,
            weights
        )

        logits = A @ self.D

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        return probabilities.detach().numpy()


    def predict(self, X):

        probabilities = self.predict_proba(
            X
        )

        predicted_index = np.argmax(
            probabilities,
            axis=1
        )

        return self.classes_[
            predicted_index
        ]