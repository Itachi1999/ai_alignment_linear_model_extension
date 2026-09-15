from dataclasses import dataclass, field
import numpy as np
from typing import Dict

from ai_alignment_linear_model_extension.models.voter import Voter
from ai_alignment_linear_model_extension.models.alternative import Alternative

@dataclass(slots=True, frozen=True)
class Election:
    """Represents an election with a set of alternatives and voters. """

    alternatives: tuple[Alternative, ...]
    voters: tuple[Voter, ...]


    # Dctionary for quick lookup of alternatives and voters by their IDs. Otherwise, we would have to iterate through the lists to find a specific alternative or voter.
    _alternative_lookup: Dict[int | str, Alternative] = field(
        init=False,
        repr=False,
        compare=False,
    )

    _voter_lookup: Dict[int, Voter] = field(
        init=False,
        repr=False,
        compare=False,
    )

    # properties
    # @property
    # def num_voters(self)-> int: 
    #     return len(self.voters)

    # @property
    # def num_alternatives(self)-> int:
    #     return len(self.alternatives)

    # @property
    # def dimension(self) -> int:
    #     return self.alternatives[0].dimension



    # def __post_init__(self) ->None:
    #     if len(self.alternatives) == 0:
    #         raise ValueError("Election must have at least one alternative.")
    #     if len(self.voters) == 0:
    #         raise ValueError("Election must have at least one voter.")
    #     if any(alt.dimension != self.dimension for alt in self.alternatives):
    #         raise ValueError("All alternatives must have the same dimension.")
    #     if any(voter.num_alternatives != self.num_alternatives for voter in self.voters):
    #         raise ValueError("All voters must rank the same number of alternatives.")

    def __post_init__(self) -> None:

        if len(self.alternatives) == 0:
            raise ValueError("Election must contain at least one alternative.")

        if len(self.voters) == 0:
            raise ValueError("Election must contain at least one voter.")

        # Validate alternative IDs

        alternative_ids = [alt.id for alt in self.alternatives]

        if len(alternative_ids) != len(set(alternative_ids)):
            raise ValueError("Alternative IDs must be unique.")

        # Validate voter IDs

        voter_ids = [voter.id for voter in self.voters]

        if len(voter_ids) != len(set(voter_ids)):
            raise ValueError("Voter IDs must be unique.")

        # Validate feature dimensions

        dimension = self.alternatives[0].dimension

        for alternative in self.alternatives:
            if alternative.dimension != dimension:
                raise ValueError(
                    "All alternatives must have the same feature dimension."
                )

        # Validate rankings

        valid_ids = set(alternative_ids)

        for voter in self.voters:

            if len(voter.ranking) != len(self.alternatives):
                raise ValueError(
                    f"Voter {voter.id} has an invalid ranking length."
                )

            if set(voter.ranking) != valid_ids:
                raise ValueError(
                    f"Voter {voter.id} does not rank every alternative exactly once."
                )

        # Lookup dictionaries

        object.__setattr__(
            self,
            "_alternative_lookup",
            {alt.id: alt for alt in self.alternatives},
        )

        object.__setattr__(
            self,
            "_voter_lookup",
            {voter.id: voter for voter in self.voters},
        )

    # Properties

    @property
    def num_alternatives(self) -> int:
        return len(self.alternatives)

    @property
    def num_voters(self) -> int:
        return len(self.voters)

    @property
    def dimension(self) -> int:
        return self.alternatives[0].dimension

    # Convenience methods

    def get_alternative(self, alternative_id: int | str) -> Alternative:
        """Return the alternative with the given ID. """
        return self._alternative_lookup[alternative_id]

    def get_voter(self, voter_id: int | str) -> Voter:
        """Return the voter with the given ID. """
        return self._voter_lookup[voter_id]

    def feature_matrix(self) -> np.ndarray:
        """
        Returns an (m × d) feature matrix.
        """
        return np.vstack(
            [alternative.features for alternative in self.alternatives]
        ).copy()
        # Returns a copy of the feature matrix to prevent accidental modifications to the original data.

    def rankings(self):
        """
        Returns an (n × m) ranking matrix.
        """
        return [voter.ranking for voter in self.voters].copy()
        #Returns a copy of the ranking matrix to prevent accidental modifications to the original data.

    # String representation
    def __str__(self) -> str:
        return (
            f"Election(num_voters={self.num_voters}, num_alternatives={self.num_alternatives}, dimension={self.dimension})"
        )                
        
