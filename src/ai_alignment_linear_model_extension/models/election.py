from dataclasses import dataclass
import numpy as np

from ai_alignment_linear_model_extension.models.voter import Voter
from ai_alignment_linear_model_extension.models.alternative import Alternative

@dataclass(slots=True, frozen=True)
class Election:
    alternatives: tuple[Alternative, ...]
    voters: tuple[Voter, ...]

    @property
    def num_voters(self)-> int: 
        return len(self.voters)

    @property
    def num_alternatives(self)-> int:
        return len(self.alternatives)

    @property
    def dimension(self) -> int:
        return self.alternatives[0].dimension
