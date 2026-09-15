from __future__ import annotations
from collections import defaultdict

from ai_alignment_linear_model_extension.models.election import Election

class Borda:
    def __init__(self, election: Election) -> None:
        self._election = election

    def scores(self) -> dict[int | str, int]:
        """Return Borda scores as {alternative_id: score}."""
        scores = defaultdict(int) # gets a default value of 0 for any new key

        num_alternatives = len(self._election.alternatives)

        for voter in self._election.voters:
            for position, alternative_id in enumerate(voter.ranking):
                scores[alternative_id] += num_alternatives - 1 - position

        return dict(scores)