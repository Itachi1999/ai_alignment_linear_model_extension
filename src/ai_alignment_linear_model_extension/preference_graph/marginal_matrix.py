# src/ai_alignment_linear_model_extension/preference_graph/marginal_matrix.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MarginalMatrix:
    """
    Represents the normalized pairwise preference matrix.

    Entry (a, b) equals

        w_{a ≻ b}

    i.e., the fraction of voters preferring alternative a over b.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:

        object.__setattr__(
            self,
            "matrix",
            np.asarray(self.matrix, dtype=float),
        )

        if self.matrix.ndim != 2:
            raise ValueError("Marginal matrix must be two-dimensional.")

        rows, cols = self.matrix.shape

        if rows != cols:
            raise ValueError("Marginal matrix must be square.")

    # Properties

    @property
    def num_alternatives(self) -> int:
        return self.matrix.shape[0]

    # Query methods

    def weight(self, a: int, b: int) -> float:
        """
        Return w_{a ≻ b}.
        """
        return float(self.matrix[a, b])

    def margin(self, a: int, b: int) -> float:
        """
        Return
            w_{a ≻ b} - w_{b ≻ a}.
        """
        return self.weight(a, b) - self.weight(b, a)

    def majority_prefers(self, a: int, b: int) -> bool:
        """
        True iff a is preferred to b by a strict majority.
        """
        # TODO: How to handle 0.5 != 0.5000000000000001 this case here?  SOLVED
        return np.isclose(self.weight(a, b), 0.5) is False and self.weight(a, b) > 0.5

    def is_tie(self, a: int, b: int) -> bool:
        """
        True iff exactly half the voters prefer each alternative.
        """
        # Is close is used to avoid floating point issues, e.g., 0.5 != 0.5000000000000001
        return np.isclose(self.weight(a, b), 0.5)

    # Convenience

    def __getitem__(self, index):
        return self.matrix[index]

    def __str__(self) -> str:
        return (
            f"MarginalMatrix(num_alternatives={self.num_alternatives})"
        )