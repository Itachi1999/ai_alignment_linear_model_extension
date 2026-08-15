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

    # Epsilon support
    epsilon_support: tuple[int, ...] | None = None # These are having default None because they are not always computed in the evaluation (like in the case of BTL model), but can be added later if needed.
    num_nonzero_epsilon: int | None = None
    epsilon_support_percentage: float | None = None
    # epsilon_sparsity: float = property(lambda self: 1 - self.epsilon_support_percentage)

    # z support
    z_support_percentage: float | None = None

    objective_value: float | None = None
    loss_value: float | None = None

    @property
    def epsilon_sparsity(self) -> float | None:
        return 100.0 - self.epsilon_support_percentage if self.epsilon_support_percentage is not None else None

    @property 
    def z_sparsity(self) -> float | None:
        return 100.0 - self.z_support_percentage if self.z_support_percentage is not None else None

