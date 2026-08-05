# src/ai_alignment_linear_model_extension/preference_graph/marginal_matrix.py

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ai_alignment_linear_model_extension.models.election import Election


@dataclass(frozen=True, slots=True)
class MarginalMatrix:
    """
    Represents the normalized pairwise preference matrix.

    Entry (a, b) equals

        w_{a ≻ b}

    i.e., the fraction of voters preferring alternative a over b.
    """
    alternative_ids: tuple[int, ...]
    matrix: np.ndarray
    _id_to_index: dict[int, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

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

        if rows != len(self.alternative_ids):
            raise ValueError(
                "Matrix size does not match the number of alternatives."
            )

        if len(set(self.alternative_ids)) != len(self.alternative_ids):
            raise ValueError("Alternative IDs must be unique.")

        object.__setattr__(
            self,
            "_id_to_index",
            {
                alternative_id: index
                for index, alternative_id in enumerate(self.alternative_ids)
            },
        )


    # Properties

    @property
    def num_alternatives(self) -> int:
        return self.matrix.shape[0]

    # Query methods

    def weight(self, a: int, b: int) -> float:
        """
        Return w_{a ≻ b}.
        """
        ia = self._id_to_index[a]
        ib = self._id_to_index[b]

        return float(self.matrix[ia, ib])

    def margin(self, a: int, b: int) -> float:
        """
        Return
            w_{a ≻ b} - w_{b ≻ a}.
        """
        ia = self._id_to_index[a]
        ib = self._id_to_index[b]
        return self.weight(ia, ib) - self.weight(ib, ia)

    def majority_prefers(self, a: int, b: int) -> bool:
        """
        True iff a is preferred to b by a strict majority.
        """
        # TODO: How to handle 0.5 != 0.5000000000000001 this case here?  SOLVED
        ia = self._id_to_index[a]
        ib = self._id_to_index[b]
        # print(f"ia: {ia}, ib: {ib}, weight(a, b): {self.weight(ia, ib)}, weight(b, a): {self.weight(ib, ia)}")
        # print(f"Majority prefers {a} over {b}: {(not np.isclose(self.weight(ia, ib), 0.5)) and (self.weight(ia, ib) > 0.5)}, weight: {self.weight(ia, ib)}")
        return (not np.isclose(self.weight(ia, ib), 0.5)) and (self.weight(ia, ib) > 0.5)

    def is_tie(self, a: int, b: int) -> bool:
        """
        True iff exactly half the voters prefer each alternative.
        """
        # Is close is used to avoid floating point issues, e.g., 0.5 != 0.5000000000000001
        ia = self._id_to_index[a]
        ib = self._id_to_index[b]
        return np.isclose(self.weight(ia, ib), 0.5)

    # Convenience
    def as_numpy(self) -> np.ndarray:
        """
        Return a copy of the underlying matrix.
        """

        return self.matrix.copy()

    def __getitem__(self, index):
        """
        Return the weight of the edge from alternative a to alternative b.
        """
        a, b = index
        ia = self._id_to_index[a]
        ib = self._id_to_index[b]
        return self.matrix[ia, ib]

    def __str__(self) -> str:
        return (
            f"MarginalMatrix(num_alternatives={self.num_alternatives})"
        )



class MarginalMatrixBuilder:
    """
    Builds the normalized marginal matrix

        w_{a ≻ b}

    from an election.
    """

    def build(
        self,
        election: Election,
    ) -> MarginalMatrix:

        alternative_ids = tuple(
            alternative.id
            for alternative in election.alternatives
        )

        id_to_index = {
            alternative_id: index
            for index, alternative_id in enumerate(alternative_ids)
        }

        n = election.num_voters
        m = election.num_alternatives

        matrix = np.zeros((m, m), dtype=float)

        # Count pairwise victories

        for voter in election.voters:

            for i, a in enumerate(alternative_ids):

                for b in alternative_ids[i + 1:]:

                    ia = id_to_index[a]
                    ib = id_to_index[b]

                    if voter.prefers(a, b):

                        matrix[ia, ib] += 1

                    else:

                        matrix[ib, ia] += 1

        # Normalize

        matrix /= n

        return MarginalMatrix(
            alternative_ids=alternative_ids,
            matrix=matrix,
        )