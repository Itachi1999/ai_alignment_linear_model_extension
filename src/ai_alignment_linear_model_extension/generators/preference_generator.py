# src/ai_alignment_linear_model_extension/generators/ranking_generator.py

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

from ai_alignment_linear_model_extension.models.voter import Voter


class PreferenceGenerator(ABC):
    """
    Abstract base class for generating voter preferences.
    """

    @abstractmethod
    def generate(
        self,
        num_voters: int,
        alternative_ids: tuple[int, ...],
    ) -> tuple[Voter, ...]:
        """
        Generate preferences for voters.

        Parameters
        ----------
        num_voters : int
            Number of voters.

        alternative_ids : tuple[int, ...]
            IDs of all alternatives.

        Returns
        -------
        tuple[Voter, ...]
            Generated voters.
        """
        pass 


class UniformPreferenceGenerator(PreferenceGenerator):
    """
    Generates independent uniformly random preferences.
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        num_voters: int,
        alternative_ids: tuple[int, ...],
    ) -> tuple[Voter, ...]:

        if num_voters <= 0:
            raise ValueError("Number of voters must be positive.")

        voters = []

        for voter_id in range(num_voters):

            ranking = tuple(
                self._rng.permutation(alternative_ids).tolist()
            )

            voters.append(
                Voter(
                    id=voter_id,
                    ranking=ranking,
                )
            )

        return tuple(voters)