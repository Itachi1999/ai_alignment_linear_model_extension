# src/ai_alignment_linear_model_extension/models/voter.py

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True, slots=True)
class Voter:
    """Represents a voter with a complete ranking of alternatives."""

    id: int | str
    ranking: tuple[int | str, ...]
    _rank_lookup: Dict[int | str, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if len(self.ranking) == 0:
            raise ValueError("Ranking cannot be empty.")

        if len(set(self.ranking)) != len(self.ranking):
            raise ValueError("Ranking contains duplicate alternatives.")

        object.__setattr__(
            self,
            "_rank_lookup",
            {
                alternative: position
                for position, alternative in enumerate(self.ranking)
            },
        )

    @property
    def num_alternatives(self) -> int:
        return len(self.ranking)

    def prefers(self, a: int, b: int) -> bool:
        """Return True iff this voter prefers a over b."""
        return self._rank_lookup[a] < self._rank_lookup[b]

    def rank_of(self, alternative: int) -> int:
        """Return the position of an alternative in the ranking (0 = best)."""
        return self._rank_lookup[alternative]

    def __str__(self) -> str:
        return (
            f"Voter(id={self.id}, ranking={self.ranking})"
        )