from __future__ import annotations
from dataclasses import dataclass

from ai_alignment_linear_model_extension.preference_graph.edge import MajorityEdge as Edge


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Result of evaluating the learned linear model.
    """
    num_constraints: int
    num_violations: int
    
    num_po_constraints: int
    num_pmc_constraints: int
    
    num_po_violations: int
    num_pmc_violations: int
    
    violation_percentage: float
    po_violation_percentage: float
    pmc_violation_percentage: float

    violated_edges: tuple[Edge, ...]