from __future__ import annotations

import numpy as np
import scipy

from ai_alignment_linear_model_extension.models.alternative import Alternative
from ai_alignment_linear_model_extension.evaluation.evaluation_result import (
    EvaluationResult,
)
# from ai_alignment_linear_model_extension.models.election import Election
from ai_alignment_linear_model_extension.optimization.loss_function_implementation import TrainableLossResult
from ai_alignment_linear_model_extension.preference_graph.edge import (
    PreferenceEdgeType,
)
from ai_alignment_linear_model_extension.preference_graph.preference_graph import (
    PreferenceGraph,
)


class LP3LossModelEvaluator:
    """Evaluator for the BTL model.
    This class evaluates the BTL model by comparing the predicted probabilities of pairwise comparisons with the actual outcomes in the preference graph.
    """

    def __init__(
        self,
        tolerance: float = 1e-9,
        eta: float = 1e-2
    ) -> None:
        self._tolerance = tolerance
        self.eta = eta

    def evaluate(
        self,
        graph: PreferenceGraph,
        result: TrainableLossResult,
        testing: bool = False,
        alternatives: tuple[Alternative, ...] | None = None
    ) -> EvaluationResult:

        violated_edges: list = []
        po_violations = 0
        pmc_violations = 0
        epsilon_l1_norm = np.sum(np.abs(list(result.epsilon.values())))
        features = {}
        if alternatives is not None:
            features = {
                alternative.id: alternative.features
                for alternative in alternatives
            }
        
        score1 = 0.0
        score2 = 0.0
        for edge in graph.edges:
            if testing:
                score1 = result.theta @ features[edge.source]
                score2 = result.theta @ features[edge.target]
            else:
                score1 = result.scores.get(edge.source, 0.0)
                score2 = result.scores.get(edge.target, 0.0)
            prob = scipy.special.expit(score1 - score2)
            # if np.isclose(prob, 0.5) or (prob < 0.5):
            score_difference = score1 - score2
            if score_difference < 0:
                violated_edges.append(edge)
                if edge.edge_type == PreferenceEdgeType.UNANIMOUS:
                    po_violations += 1
                elif edge.edge_type == PreferenceEdgeType.MAJORITY:
                    pmc_violations += 1

        num_constraints = len(graph.edges)
        num_violations = len(violated_edges)

        violation_percentage = 100 * num_violations / num_constraints if num_constraints > 0 else 0.0
        po_violation_percentage = (
            100 * po_violations / graph.num_po_edges if graph.num_po_edges > 0 else 0.0
        )
        pmc_violation_percentage = (
            100 * pmc_violations / graph.num_pmc_edges if graph.num_pmc_edges > 0 else 0.0
        )

        return EvaluationResult(
            num_constraints=num_constraints,
            num_violations=num_violations,
            num_po_constraints=graph.num_po_edges,
            num_pmc_constraints=graph.num_pmc_edges,
            num_po_violations=po_violations,
            num_pmc_violations=pmc_violations,
            violation_percentage=violation_percentage,
            po_violation_percentage=po_violation_percentage,
            pmc_violation_percentage=pmc_violation_percentage,
            violated_edges=tuple(violated_edges),
            epsilon_l1_norm=epsilon_l1_norm,
            loss_value=result.objective
        )