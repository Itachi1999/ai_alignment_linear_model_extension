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
    _alternative_lookup: Dict[int, Alternative] = field(
        init=False,
        repr=False,
        compare=False,
    )

    _voter_lookup: Dict[int, Voter] = field(
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def num_voters(self)-> int: 
        return len(self.voters)

    @property
    def num_alternatives(self)-> int:
        return len(self.alternatives)

    @property
    def dimension(self) -> int:
        return self.alternatives[0].dimension


    def __post_init__(self) ->None:
        if len(self.alternatives) == 0:
            raise ValueError("Election must have at least one alternative.")
        if len(self.voters) == 0:
            raise ValueError("Election must have at least one voter.")
        if any(alt.dimension != self.dimension for alt in self.alternatives):
            raise ValueError("All alternatives must have the same dimension.")
        if any(voter.num_alternatives != self.num_alternatives for voter in self.voters):
            raise ValueError("All voters must rank the same number of alternatives.")

                
        
