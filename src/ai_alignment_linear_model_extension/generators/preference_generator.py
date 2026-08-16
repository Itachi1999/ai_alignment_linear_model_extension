# src/ai_alignment_linear_model_extension/generators/ranking_generator.py

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

from ai_alignment_linear_model_extension.models.voter import Voter
from ai_alignment_linear_model_extension.models.alternative import Alternative


class PreferenceGenerator(ABC):
    """
    Abstract base class for generating voter preferences.
    """

    @abstractmethod
    def generate(
        self,
        num_voters: int,
        alternatives: tuple[Alternative, ...],
    ) -> tuple[Voter, ...]:
        """
        Generate preferences for voters.

        Parameters
        ----------
        num_voters : int
            Number of voters.

        alternatives : tuple[Alternative, ...]
            Tuple of all alternatives.

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
        alternatives: tuple[Alternative, ...],
    ) -> tuple[Voter, ...]:

        if num_voters <= 0:
            raise ValueError("Number of voters must be positive.")

        voters = []
        alternative_ids = tuple(alt.id for alt in alternatives)

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
    


class LinearPreferenceGenerator(PreferenceGenerator):
    def __init__(
        self,
        theta_0: np.ndarray,
        noise_std: float,
        noisy_alternative_fraction: float,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= noisy_alternative_fraction<= 1.0:
            raise ValueError("noisy_voter_fraction must be in [0, 1].")

        self.theta_0 = theta_0
        self.noise_std = noise_std
        self.noisy_alternative_fraction = noisy_alternative_fraction
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        alternatives: tuple[Alternative, ...],
        num_voters: int,
    ) -> tuple[Voter, ...]:

        if num_voters <= 0:
            raise ValueError("Number of voters must be positive.")

        num_alternatives = len(alternatives)
        voters = []

        # Linear utility: <theta_0, x_a>
        utility = {
            alt.id : self.theta_0 @ alt.features
            for alt in alternatives
        }

        noisy_alternative_ids = set(
            self.rng.choice(
                tuple(utility.keys()),
                size=num_noisy,
                replace=False,
            )
        )
        for alt_id in noisy_alternative_ids:
            utility[alt_id] += 10
        # Sanity check for linearity
        num_noisy = int(
            round(num_alternatives * self.noisy_alternative_fraction)
        )

        for voter_id in range(num_voters):
            #TODO: Mallows Model
            for alt_id in noisy_alternative_ids:
                utility[alt_id] += self.rng.normal(
                    loc=0.0,
                    scale=self.noise_std,
                )
            ranking = tuple(sorted(utility, key=utility.get, reverse=True))
            voters.append(
                Voter(
                    id = voter_id,
                    ranking=ranking
                )
            )

        return tuple(voters)


class KLengthPOPreferenceGenerator(PreferenceGenerator):
    """
    Generates preference profiles containing a common k-length PO subsequence.

    The relative order of the k selected alternatives is identical for
    every voter, while the remaining alternatives are freely permuted
    and interleaved.
    """

    def __init__(
        self,
        theta: np.ndarray,
        k: int,
        seed: int | None = None,
    ) -> None:
        self._theta = theta
        self._k = k
        self._rng = np.random.default_rng(seed)

    def generate(self, num_voters: int, alternatives: tuple[Alternative, ...]) -> tuple[Voter, ...]:
        m = len(alternatives)
        if not 1 <= self._k <= m:
            raise ValueError(
                f"k must satisfy 1 <= k <= {m}."
            )

        if self._theta.shape[0] != alternatives[0].features.shape[0]:
            raise ValueError(
                "Theta dimension must match the feature dimension."
            )

        # Step 1: Compute utility under theta

        utilities = {
            alternative.id: float(
                self._theta @ alternative.features
            )
            for alternative in alternatives
        }

        # Step 2: Select and order the common k alternatives

        po_sequence = tuple(
            sorted(
                utilities,
                key=utilities.get,
                reverse=True,
            )[: self._k]
        )

        po_ids = set(po_sequence)

        remaining_ids = tuple(
            alternative.id
            for alternative in alternatives
            if alternative.id not in po_ids
        )

        # Step 3: Generate voters
        voters = []


        for voter_id in range(num_voters):

            # Random order of non-PO alternatives
            remaining = list(remaining_ids)
            self._rng.shuffle(remaining)

            # Choose positions occupied by PO sequence
            po_positions = sorted(
                self._rng.choice(
                    m,
                    size=self._k,
                    replace=False,
                )
            )

            ranking = [-1] * m

            # Preserve PO order
            for position, alternative_id in zip(
                po_positions,
                po_sequence,
            ):
                ranking[position] = alternative_id

            # Fill remaining positions
            remaining_iter = iter(remaining)

            for index in range(m):
                if ranking[index] == -1:
                    ranking[index] = next(remaining_iter)

            voters.append(
                Voter(
                    id=voter_id,
                    ranking=tuple(ranking)
                )
            )

        return tuple(voters)