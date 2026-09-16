from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ai_alignment_linear_model_extension.models.btl import BTLResult
from ai_alignment_linear_model_extension.models.election import Election, Alternative
from ai_alignment_linear_model_extension.preference_graph.marginal_matrix import MarginalMatrix

    
class BTLLinearModel:
    """
    Feature-based Bradley-Terry-Luce model.

    score(a) = theta^T x_a
    """
    def __init__(
        self,
        election:Election,
        marginal_matrix:MarginalMatrix,
        ) -> None:
        self.election = election
        self.marginal_matrix = marginal_matrix
        
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
    
    def probability(
        self,
        alt1: int | str,
        alt2: int | str,
        theta: np.ndarray,
    ) -> float:
        score_difference = theta @ (
            self._features[alt1] - self._features[alt2]
        )

        return float(expit(score_difference))
    
    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        ll = 0.0

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

                probability = self.probability(
                    alt1,
                    alt2,
                    theta,
                )

                ll += count * np.log(probability) if probability > 0 else 0

        return float(ll)
    
    def fit(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6
        ) -> BTLResult:
        
        def negative_log_likelihood(
            theta: np.ndarray,
        ) -> float:
            return -self.log_likelihood(theta)

        constraints = [
            {
                "type": "ineq",
                "fun": lambda theta: 1.0 - np.max(np.abs(theta)),
            }
        ]

        result = minimize(
            negative_log_likelihood,
            x0=np.zeros(self.dimension),
            method="SLSQP",
            constraints=constraints,
            options={
                "maxiter": max_iter,
                "ftol": tol,
            },
        )

        self.theta = np.asarray(
            result.x,
            dtype=float,
        ).copy()
        
        scores = {
            alt.id: float(np.dot(self.theta, self._features[alt.id]))
            for alt in self.alternatives
        }

        return BTLResult(
            scores=scores.copy(),
            log_likelihood=-float(result.fun),
            converged=bool(result.success),
            iterations=int(result.nit),
        )
        



class BTLPartialLinearModel:
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

        self.theta = np.zeros(
            self.dimension,
            dtype=float,
        )

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

    def probability(
        self,
        alt1: int | str,
        alt2: int | str,
        theta: np.ndarray,
    ) -> float:
        score_difference = theta @ (
            self._features[alt1] -
            self._features[alt2]
        )

        return float(
            expit(score_difference)
        )

    def log_likelihood(
        self,
        theta: np.ndarray,
    ) -> float:
        ll = 0.0

        for election, marginal_matrix in zip(self.elections, self.marginal_matrices):
            for alt1 in marginal_matrix.alternative_ids:
                for alt2 in marginal_matrix.alternative_ids:

                    if alt1 == alt2:
                        continue

                    count = marginal_matrix.weight(alt1, alt2) * election.num_voters
                    
                    if count == 0:
                        continue

                    probability = self.probability(alt1=alt1, alt2=alt2, theta=theta)

                    ll += count * np.log(probability) if probability > 0 else 0

        return float(ll)

    def fit(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> BTLResult:

        def negative_log_likelihood(
            theta: np.ndarray,
        ) -> float:
            return -self.log_likelihood(theta)

        constraints = [
            {
                "type": "ineq",
                "fun": lambda theta: (
                    1.0 - np.max(np.abs(theta))
                ),
            }
        ]

        result = minimize(
            negative_log_likelihood,
            x0=np.zeros(self.dimension),
            method="SLSQP",
            constraints=constraints,
            options={
                "maxiter": max_iter,
                "ftol": tol,
            },
        )

        self.theta = np.asarray(
            result.x,
            dtype=float,
        ).copy()

        scores = {
            alternative_id: float(
                self.theta @ features
            )
            for alternative_id, features in self._features.items()
        }

        return BTLResult(
            scores=scores,
            log_likelihood=-float(result.fun),
            converged=bool(result.success),
            iterations=int(result.nit),
        )