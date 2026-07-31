from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MajorityEdge:
    """
    Directed edge in a preference graph.
    """

    source: int
    target: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("Self-loops are not allowed.")

        # This check that weight is a postive number and greater than 1/2 is only for majority graph, for other graphs, we can have negative weights. So, we will not check for negative weights here.
        if self.weight <= 0.5:
            raise ValueError("Weight must be greater than 0.5 for majority graph.")