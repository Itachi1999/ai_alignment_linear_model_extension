from __future__ import annotations

from dataclasses import dataclass


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

    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores


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

