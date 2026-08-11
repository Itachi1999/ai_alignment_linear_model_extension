from __future__ import annotations

import numpy as np

from ai_alignment_linear_model_extension.evaluation.evaluation_result import (
    EvaluationResult,
)
from ai_alignment_linear_model_extension.models.election import Election
from ai_alignment_linear_model_extension.models.btl import BTLResult, BTLModel
from ai_alignment_linear_model_extension.preference_graph.edge import (
    PreferenceEdgeType,
)
from ai_alignment_linear_model_extension.preference_graph.preference_graph import (
    PreferenceGraph,
)


class BTLModelEvaluator:
    """Evaluator for the BTL model.
    This class evaluates the BTL model by comparing the predicted probabilities of pairwise comparisons with the actual outcomes in the preference graph.
    """

    def __init__(
        self,
        tolerance: float = 1e-9,
    ) -> None:
        self._tolerance = tolerance

    def evaluate(
        self,
        graph: PreferenceGraph,
        result: BTLResult,
    ) -> EvaluationResult:

        violated_edges: list = []

        po_violations = 0
        pmc_violations = 0

        for edge in graph.edges:

            score1 = result.scores.get(edge.source, 0.0)
            score2 = result.scores.get(edge.target, 0.0)

            prob = result.model.probability(score1, score2)

            if np.isclose(prob, 0.5, atol=self._tolerance) or prob < 0.5:
                violated_edges.append(edge)
                if edge.edge_type == PreferenceEdgeType.UNANIMOUS:
                    po_violations += 1
                elif edge.edge_type == PreferenceEdgeType.MAJORITY:
                    pmc_violations += 1

        num_constraints = len(graph.edges)
        num_violations = len(violated_edges)

        violation_percentage = (
            num_violations / num_constraints if num_constraints > 0 else 0.0
        )
        po_violation_percentage = (
            po_violations / graph.num_po_edges if graph.num_po_edges > 0 else 0.0
        )
        pmc_violation_percentage = (
            pmc_violations / graph.num_pmc_edges if graph.num_pmc_edges > 0 else 0.0
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
        )