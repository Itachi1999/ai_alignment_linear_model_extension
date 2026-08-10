from __future__ import annotations

import numpy as np

from ai_alignment_linear_model_extension.evaluation.evaluation_result import (
    EvaluationResult,
)
from ai_alignment_linear_model_extension.models.election import Election
from ai_alignment_linear_model_extension.optimization.lp_result import LPResult
from ai_alignment_linear_model_extension.preference_graph.edge import (
    PreferenceEdgeType,
)
from ai_alignment_linear_model_extension.preference_graph.preference_graph import (
    PreferenceGraph,
)


class LinearModelEvaluator:

    def __init__(
        self,
        tolerance: float = 1e-9,
    ) -> None:
        self._tolerance = tolerance

    def evaluate(
        self,
        election: Election,
        graph: PreferenceGraph,
        result: LPResult,
    ) -> EvaluationResult:

        features = {
            alternative.id: alternative.features
            for alternative in election.alternatives
        }

        violated_edges: list = []

        po_violations = 0
        pmc_violations = 0

        epsilon = result.epsilon

        epsilon_support = tuple(
            alternative.id
            for alternative in election.alternatives
            if abs(epsilon[alternative.id]) > self._tolerance
        )

        for edge in graph.edges:

            score = float(
                np.dot(
                    result.theta,
                    features[edge.source] - features[edge.target],
                )
            )

            if score < self._tolerance:

                violated_edges.append(edge)
                # print(f"Violated edge: {edge.source} -> {edge.target}, score: {score}, type: {edge.edge_type}")
                if edge.edge_type == PreferenceEdgeType.UNANIMOUS:
                    po_violations += 1
                else:
                    pmc_violations += 1

        total = len(graph.edges)
        total_violations = len(violated_edges)

        total_po = sum(
            edge.edge_type == PreferenceEdgeType.UNANIMOUS
            for edge in graph.edges
        )

        total_pmc = total - total_po

        return EvaluationResult(
            num_constraints=total,
            num_violations=total_violations,
            num_po_constraints=total_po,
            num_pmc_constraints=total_pmc,
            num_po_violations=po_violations,
            num_pmc_violations=pmc_violations,
            epsilon_support=epsilon_support,
            num_nonzero_epsilon=len(epsilon_support),
            epsilon_support_percentage=(
                len(epsilon_support) / len(election.alternatives)
                if election.alternatives else 0.0
            ),
            violation_percentage=100 * total_violations / total if total else 0.0,
            po_violation_percentage=100 * po_violations / total_po if total_po else 0.0,
            pmc_violation_percentage=100 * pmc_violations / total_pmc if total_pmc else 0.0,
            violated_edges=tuple(violated_edges),
        )