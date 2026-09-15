from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class PreferenceEdgeType(Enum):
    """Types of edges in a preference graph."""

    MAJORITY = auto()
    UNANIMOUS = auto()

@dataclass(frozen=True, slots=True)
class MajorityEdge:
    """
    Directed edge in a preference graph.
    """

    source: int | str
    target: int | str
    weight: float
    edge_type: PreferenceEdgeType = PreferenceEdgeType.MAJORITY

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("Self-loops are not allowed.")

        # This check that weight is a postive number and greater than 1/2 is only for majority graph, for other graphs, we can have negative weights. So, we will not check for negative weights here.
        if self.weight <= 0.5:
            raise ValueError("Weight must be greater than 0.5 for majority graph.")

    def __str__(self):
        return f"Edge from {self.source} to {self.target}"