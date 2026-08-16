from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy
from scipy.optimize import minimize

from ai_alignment_linear_model_extension.preference_graph.marginal_matrix import MarginalMatrix
from ai_alignment_linear_model_extension.models.alternative import Alternative

class BTLModel:
    """
    A class representing the Bradley-Terry-Luce (BTL) model for pairwise comparisons.
    The BTL model estimates the probability of one alternative being preferred over another
    based on their respective scores.

    Attributes:
    -----------
    scores: dict[int, float]
        A dictionary mapping AlternativeIDs to their estimated scores.
    """

    def __init__(self, scores: dict[int, float], marginal_matrix: MarginalMatrix, num_voters:int) -> None:
        self.scores = scores
        self.marginal_matrix = marginal_matrix
        self.num_voters = num_voters

    @property
    def alternative_ids(self) -> tuple[int, ...]:
        return self.marginal_matrix.alternative_ids

    @property
    def num_alternatives(self) -> int:
        """
        Returns the number of alternatives in the model.

        Returns:
        --------
        int
            The number of alternatives.
        """
        return len(self.alternative_ids)

    def probability(self, score1: float, score2: float) -> float:
        """
        Calculate the probability of alternative alt1 being preferred over alternative alt2.

        Parameters:
        -----------
        score1: float
            The score of the first alternative.
        score2: float
            The score of the second alternative.

        Returns:
        --------
        float
            The probability of alt1 being preferred over alt2.
        """
        prob = scipy.special.expit(score1 - score2)
        return prob

    def log_likelihood(self, scores: dict[int, float]) -> float:
        """
        Calculate the log-likelihood of the model given a list of pairwise comparisons.

        Parameters:
        -----------
        scores: dict[int, float]
            A dictionary mapping AlternativeIDs to their scores.

        Returns:
        --------
        float
            The log-likelihood of the model given the data.
        """
        ll = 0.0
        for i, alt1 in enumerate(self.alternative_ids):
            for j, alt2 in enumerate(self.alternative_ids):
                if i != j:
                    score1 = scores.get(alt1, 0.0)
                    score2 = scores.get(alt2, 0.0)
                    prob = self.probability(score1, score2)
                    count = self.marginal_matrix.weight(alt1, alt2) * self.num_voters  # Since alternative ids are not necessarily the index of matrix, we use the weight method to get the count of comparisons
                    ll += count * np.log(prob) if prob > 0 else 0
        return ll

    def fit(self, max_iter: int = 1000, tol: float = 1e-6, optimizer: str = "gradient_descent") -> BTLResult:
        """
        Fit the BTL model to the data using an iterative optimization algorithm.

        Parameters:
        -----------
        max_iter: int
            The maximum number of iterations for the optimization algorithm.
        tol: float
            The tolerance for convergence. If the change in log-likelihood is less than this value, the algorithm will stop.

        Returns:
        --------
        BTLResult
            An object containing the estimated scores, log-likelihood, convergence status, and number of iterations.
        """
        # Initialize scores to zero
        self.scores = {alt_id: 0.0 for alt_id in self.alternative_ids}

        def negative_log_likelihood(
            scores_array: np.ndarray,
        ) -> float:

            scores = dict(
                zip(self.alternative_ids, scores_array)
            )

            return -self.log_likelihood(scores)

        constraint = {
            "type": "eq",
            "fun": lambda scores: np.sum(scores),
        }

        result = minimize(
            negative_log_likelihood,
            x0=np.zeros(self.num_alternatives),
            method="SLSQP",
            constraints=constraint,
            options={
                "maxiter": max_iter,
                "ftol": tol,
            },
        )

        self.scores = {
            alt_id: float(score)
            for alt_id, score in zip(
                self.alternative_ids,
                result.x,
            )
        }

        return BTLResult(
            scores=self.scores.copy(),
            log_likelihood=float(result.fun),
            converged=bool(result.success),
            iterations=int(result.nit),
        )
        
        
class BTLHingeModel:
    """
    A class representing the Bradley-Terry-Luce (BTL) model for pairwise comparisons.
    The BTL model estimates the probability of one alternative being preferred over another
    based on their respective scores.

    Attributes:
    -----------
    scores: dict[int, float]
        A dictionary mapping AlternativeIDs to their estimated scores.
    """

    def __init__(self, scores: dict[int, float], marginal_matrix: MarginalMatrix, num_voters:int) -> None:
        self.scores = scores
        self.marginal_matrix = marginal_matrix
        self.num_voters = num_voters

    @property
    def alternative_ids(self) -> tuple[int, ...]:
        return self.marginal_matrix.alternative_ids

    @property
    def num_alternatives(self) -> int:
        """
        Returns the number of alternatives in the model.

        Returns:
        --------
        int
            The number of alternatives.
        """
        return len(self.alternative_ids)

    def probability(self, score1: float, score2: float) -> float:
        """
        Calculate the probability of alternative alt1 being preferred over alternative alt2.

        Parameters:
        -----------
        score1: float
            The score of the first alternative.
        score2: float
            The score of the second alternative.

        Returns:
        --------
        float
            The probability of alt1 being preferred over alt2.
        """
        prob = scipy.special.expit(score1 - score2)
        return prob

    def log_likelihood(self, scores: dict[int, float]) -> float:
        """
        Calculate the log-likelihood of the model given a list of pairwise comparisons.

        Parameters:
        -----------
        scores: dict[int, float]
            A dictionary mapping AlternativeIDs to their scores.

        Returns:
        --------
        float
            The log-likelihood of the model given the data.
        """
        ll = 0.0
        for i, alt1 in enumerate(self.alternative_ids):
            for j, alt2 in enumerate(self.alternative_ids):
                if i != j:
                    score1 = scores.get(alt1, 0.0)
                    score2 = scores.get(alt2, 0.0)
                    prob = self.probability(score1, score2)
                    count = self.marginal_matrix.weight(alt1, alt2) * self.num_voters  # Since alternative ids are not necessarily the index of matrix, we use the weight method to get the count of comparisons
                    ll += count * np.log(prob) if prob > 0 else 0
        return ll

    def fit(self, max_iter: int = 1000, tol: float = 1e-6, optimizer: str = "gradient_descent") -> BTLResult:
        """
        Fit the BTL model to the data using an iterative optimization algorithm.

        Parameters:
        -----------
        max_iter: int
            The maximum number of iterations for the optimization algorithm.
        tol: float
            The tolerance for convergence. If the change in log-likelihood is less than this value, the algorithm will stop.

        Returns:
        --------
        BTLResult
            An object containing the estimated scores, log-likelihood, convergence status, and number of iterations.
        """
        # Initialize scores to zero
        self.scores = {alt_id: 0.0 for alt_id in self.alternative_ids}

        def hinge_loss(
            scores: dict[int, float],
            margin: float = 1.0,
        ) -> float:

            loss = 0.0

            for alt1 in self.alternative_ids:
                for alt2 in self.alternative_ids:
                    if alt1 == alt2:
                        continue

                    score_diff = scores[alt1] - scores[alt2]

                    count = self.marginal_matrix.weight(alt1, alt2) * self.num_voters

                    loss += count * max(
                        0.0,
                        margin - score_diff,
                    )

            return float(loss)

        constraint = {
            "type": "eq",
            "fun": lambda scores: np.sum(scores),
        }

        result = minimize(
            hinge_loss,
            x0=np.zeros(self.num_alternatives),
            method="SLSQP",
            constraints=constraint,
            options={
                "maxiter": max_iter,
                "ftol": tol,
            },
        )

        self.scores = {
            alt_id: float(score)
            for alt_id, score in zip(
                self.alternative_ids,
                result.x,
            )
        }

        return BTLResult(
            scores=self.scores.copy(),
            log_likelihood=float(result.fun),
            converged=bool(result.success),
            iterations=int(result.nit),
        )

@dataclass(frozen=True, slots=True)
class BTLResult:
    """
    Result of running the BTL model on a given dataset.
    Parameters:
    -----------
    scores: dict[int, float]
        A dictionary mapping AlternativeIDs to their estimated scores.
    log_likelihood: float
        The log-likelihood of the model given the data.
    converged: bool
        Whether the optimization algorithm converged.
    iterations: int
        The number of iterations taken by the optimization algorithm.
    """
    scores: dict[int, float]
    log_likelihood: float
    converged: bool
    iterations: int

