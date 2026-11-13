from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.svm import LinearSVC
from ai_alignment_linear_model_extension.models.btl import BTLResult
from ai_alignment_linear_model_extension.models.election import Election, Alternative
from ai_alignment_linear_model_extension.preference_graph.marginal_matrix import MarginalMatrix
import warnings
from sklearn.exceptions import ConvergenceWarning
    
class HingeLinearModel:
    """
    Feature-based Bradley-Terry-Luce model.

    score(a) = theta^T x_a
    """
    def __init__(
        self,
        election:Election,
        marginal_matrix:MarginalMatrix,
        margin:float,
        ) -> None:
        self.election = election
        self.marginal_matrix = marginal_matrix
        self.margin = margin

        self.theta = np.zeros(
            self.dimension,
            dtype=float,
        )

        self._features = {
            alternative.id: alternative.features
            for alternative in self.alternatives
        }
        
    @property
    def num_voters(self) -> int:
        return self.election.num_voters
    
    @property 
    def dimension(self) -> int:
        return self.election.dimension
    
    @property
    def alternatives(self) -> tuple[Alternative, ...]:
        return self.election.alternatives
    
    # def probability(
    #     self,
    #     alt1: int | str,
    #     alt2: int | str,
    #     theta: np.ndarray,
    # ) -> float:
    #     score_difference = theta @ (
    #         self._features[alt1] - self._features[alt2]
    #     )

    #     return float(expit(score_difference))
    
    def hinge_loss(
        self,
        theta: np.ndarray,
    ) -> float:
        loss = 0.0

        for alt1 in self.marginal_matrix.alternative_ids:
            for alt2 in self.marginal_matrix.alternative_ids:

                if alt1 == alt2:
                    continue

                count = (
                    self.marginal_matrix.weight(alt1, alt2)
                    * self.num_voters
                )

                if count == 0:
                    continue
                
                linear_score_difference = theta @ (
                    self._features[alt1] - self._features[alt2]
                )

                loss += count * max(
                    0.0,
                    self.margin - linear_score_difference
                )

        return float(loss)

    def build_pairwise_data(self):
        X, y, weights = [], [], []

        for a in self.marginal_matrix.alternative_ids:
            for b in self.marginal_matrix.alternative_ids:
                if a == b:
                    continue
                
                difference = self._features[a] - self._features[b]
                count_ab = (
                    self.marginal_matrix.weight(a, b) * self.num_voters
                )
                count_ba = (
                    self.marginal_matrix.weight(b, a) * self.num_voters
                )

                if count_ab > 0:
                    X.append(difference)
                    y.append(1)
                    weights.append(count_ab)

                if count_ba > 0:
                    X.append(difference)
                    y.append(-1)
                    weights.append(count_ba)

        return (
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(weights, dtype=np.float64),
        )
    
    def fit(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6
        ) -> BTLResult:
        
        # def optimization_loss(
        #     theta: np.ndarray,
        # ) -> float:
        #     return self.hinge_loss(theta)
        # constraints = [
        #     {
        #         "type": "ineq",
        #         "fun": lambda theta: 1.0 - np.max(np.abs(theta)),
        #     }
        # ]
        
        # result = minimize(
        #     optimization_loss,
        #     x0=np.zeros(self.dimension),
        #     method="SLSQP",
        #     constraints=constraints,
        #     options={
        #         "maxiter": max_iter,
        #         "ftol": tol,
        #     },
        # )
        
        X_train, y_train, sample_weights = self.build_pairwise_data()
        W = np.sum(sample_weights)
        reg_strength = 1e-3
        if self.margin <= 0 or reg_strength <= 0 or W <= 0:
            raise ValueError("eta, reg_strength, and total weight must be positive.")

        C = 1.0 / (reg_strength * self.margin * W)

        classifier = LinearSVC(
            penalty="l2",
            loss="hinge",
            dual=True,
            C=C,
            fit_intercept=False,
            max_iter=max_iter,
            tol=tol,
            random_state=42,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)

            classifier.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
            )

        converged = not any(
            issubclass(w.category, ConvergenceWarning)
            for w in caught
        )
        
        self.theta = self.margin * np.asarray(
            classifier.coef_.ravel(),
            dtype=float,
        ).copy()
        
        scores = {
            alt.id: float(np.dot(self.theta, self._features[alt.id]))
            for alt in self.alternatives
        }

        margins = y_train * (X_train @ self.theta)

        hinge_loss = (
            np.sum(
                sample_weights * np.maximum(0.0, self.margin - margins)
            ) / W + reg_strength * np.sum(self.theta ** 2)
        )

        return BTLResult(
            scores=scores.copy(),
            log_likelihood=-float(hinge_loss),
            converged= converged,
            iterations=int(np.max(classifier.n_iter_)),
            learned_theta=self.theta,
        )
        
        
class HingePartialLinearModel:
    """
    Feature-based Bradley-Terry-Luce model trained jointly on
    multiple independent elections.

    score(a) = theta^T x_a

    A single shared theta is learned across all elections.
    """

    def __init__(
        self,
        elections: tuple[Election, ...],
        marginal_matrices: tuple[MarginalMatrix, ...],
        margin:float,
        tau: float = 1e-4
    ) -> None:
        if len(elections) != len(marginal_matrices):
            raise ValueError(
                "The number of elections must match the number "
                "of marginal matrices."
            )

        if not elections:
            raise ValueError(
                "At least one election is required."
            )

        dimensions = {
            election.dimension
            for election in elections
        }

        if len(dimensions) != 1:
            raise ValueError(
                "All elections must have the same feature dimension."
            )

        self.elections = elections
        self.marginal_matrices = marginal_matrices
        self.margin = margin
        self.theta = np.zeros(
            self.dimension,
            dtype=float,
        )
        self.tau = tau
        self._features = {}

        for election in elections:
            for alternative in election.alternatives:
                if alternative.id in self._features:
                    raise ValueError(
                        f"Duplicate alternative ID: {alternative.id}"
                    )
                self._features[alternative.id] = alternative.features

    @property
    def num_elections(self) -> int:
        return len(self.elections)

    @property
    def dimension(self) -> int:
        return self.elections[0].dimension
    
    def hinge_loss(
        self,
        theta: np.ndarray,
    ) -> float:
        loss = 0.0

        for election, marginal_matrix in zip(self.elections, self.marginal_matrices):
            for alt1 in marginal_matrix.alternative_ids:
                for alt2 in marginal_matrix.alternative_ids:

                    if alt1 == alt2:
                        continue

                    count = marginal_matrix.weight(alt1, alt2) * election.num_voters

                    if count == 0:
                        continue

                    linear_score_difference = theta @ (
                        self._features[alt1] - self._features[alt2]
                    )

                    loss += count * max(
                        0.0,
                        self.margin - linear_score_difference
                    )

        return float(loss)

    def build_pairwise_data(self):
        elections = self.elections
        marginal_matrices = self.marginal_matrices


        X, y, weights = [], [], []

        for election, matrix in zip(elections, marginal_matrices):
            ids = list(matrix.alternative_ids)

            for i, a in enumerate(ids):
                for b in ids[i + 1:]:

                    difference = self._features[a] - self._features[b]

                    count_ab = (
                        matrix.weight(a, b) * election.num_voters
                    )
                    count_ba = (
                        matrix.weight(b, a) * election.num_voters
                    )

                    if count_ab > 0:
                        X.append(difference)
                        y.append(1)
                        weights.append(count_ab)

                    if count_ba > 0:
                        X.append(difference)
                        y.append(-1)
                        weights.append(count_ba)

        return (
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(weights, dtype=np.float64),
        )


    def fit(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        ftol: float = 1e-10,
        gtol: float = 1e-6,
    ) -> BTLResult:

        # def optimization_loss(
        #     theta: np.ndarray,
        # ) -> float:
        #     return self.hinge_loss(theta)

        # constraints = [
        #     {
        #         "type": "ineq",
        #         "fun": lambda theta: (
        #             1.0 - np.max(np.abs(theta))
        #         ),
        #     }
        # ]

        # result = minimize(
        #     optimization_loss,
        #     x0=np.zeros(self.dimension),
        #     method="SLSQP",
        #     constraints=constraints,
        #     options={
        #         "maxiter": max_iter,
        #         "ftol": tol,
        #     },
        # )
        X, y, sample_weights = self.build_pairwise_data()
        W = np.sum(sample_weights)
        reg_strength = 1e-3
        def objective_and_gradient(theta):
            tau = self.tau
            scores = X @ theta

            # Residual hinge margins
            margins = self.margin - y * scores

            # Smooth hinge loss
            softplus_loss = tau * np.logaddexp(
                0.0, margins / tau
            )

            data_loss = np.dot(
                sample_weights, softplus_loss
            ) / W

            regularization = (
                0.5 * reg_strength * np.dot(theta, theta)
            )

            loss = data_loss + regularization

            # Analytical gradient
            coefficients = (
                -sample_weights
                * y
                * expit(margins / tau)
                / W
            )

            gradient = (
                X.T @ coefficients
                + reg_strength * theta
            )

            return loss, gradient


        if self.margin <= 0 or reg_strength <= 0 or W <= 0:
            raise ValueError("eta, reg_strength, and total weight must be positive.")
        initial_theta = np.zeros(self.dimension, dtype=np.float64)
        # C = 1.0 / (reg_strength * self.margin * W)
        
        # classifier = LinearSVC(
        #     penalty="l2",
        #     loss="hinge",
        #     dual=True,
        #     C=C,
        #     fit_intercept=False,
        #     max_iter=max_iter,
        #     tol=tol,
        #     random_state=42,
        # )
        result = minimize(
            fun=objective_and_gradient,
            x0=initial_theta,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": max_iter,
                "ftol": ftol,
                "gtol": gtol,
            },
        )
        # with warnings.catch_warnings(record=True) as caught:
        #     warnings.simplefilter("always", ConvergenceWarning)
        
        #     classifier.fit(
        #         X_train,
        #         y_train,
        #         sample_weight=sample_weights,
        #     )
        
        # converged = not any(
        #     issubclass(w.category, ConvergenceWarning)
        #     for w in caught
        # )
        
        # self.theta = self.margin * np.asarray(
        #     classifier.coef_.ravel(),
        #     dtype=float,
        # ).copy()
        
        # margins = y_train * (X_train @ self.theta)
        
        # hinge_loss = (
        #     np.sum(
        #         sample_weights * np.maximum(0.0, self.margin - margins)
        #     ) / W
        # )
        self.theta = result.x.copy()
        scores = {
            alternative_id: float(
                self.theta @ features
            )
            for alternative_id, features in self._features.items()
        }
        # Evaluate the original, non-smoothed hinge loss.
        margins = self.margin - y * (X @ self.theta)
        hinge_loss = np.dot(
            sample_weights,
            np.maximum(0.0, margins)
        ) / W

        regularization_loss = (
            0.5 * reg_strength * np.dot(self.theta, self.theta)
        )
        return_dict = {
            
        }
        # print(return_dict)
        return BTLResult(
            learned_theta = self.theta,
            converged = bool(result.success),
            log_likelihood = hinge_loss + regularization_loss,
            scores = scores.copy(),
            iterations = int(result.nit),
        )
