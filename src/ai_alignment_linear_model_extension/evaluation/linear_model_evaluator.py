from __future__ import annotations

import numpy as np
import logging

from ai_alignment_linear_model_extension.evaluation.evaluation_result import (
    EvaluationResult,
)
from ai_alignment_linear_model_extension.models.election import Election, Alternative
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
        graph: PreferenceGraph,
        result: LPResult,
        election: Election | None,
        alternatives: tuple[Alternative, ...]
    ) -> EvaluationResult:

        if election is None and alternatives is None:
            raise ValueError("At least one of alternatives or election must be given")

        if election is not None:
            alternatives = election.alternatives
        # else:
        #     alternatives = alternatives

        features = {
            alternative.id: alternative.features
            for alternative in alternatives
        }
        
        violated_edges: list = []

        po_violations = 0
        pmc_violations = 0

        epsilon = result.epsilon

        z = result.z

        z_support = None
        if z is not None:
            z_support = tuple(
                edge_key
                for edge_key, z_val in z.items()
                if z_val > self._tolerance
            )
            
        epsilon_l1_norm = sum(abs(epsilon_val) for epsilon_val in epsilon.values())

        epsilon_support = tuple(
            alternative.id
            for alternative in alternatives
            if abs(epsilon[alternative.id]) > 0.0
        )

        for edge in graph.edges:

            score = float(
                np.dot(
                    result.theta,
                    features[edge.source] - features[edge.target],
                )
            )
            # print(f"Score: {score}")
            epsilon_diff = (
                result.epsilon[edge.source]
                - result.epsilon[edge.target]
            )

            total_score = score + epsilon_diff

            logging.debug(
                f"Edge {edge.source} -> {edge.target} | theta= {result.theta} | linear score={score} | feature difference= {features[edge.source] - features[edge.target]} | epsilon_diff={epsilon_diff} | total= {total_score}"
            )
            if score < 0.0:

                violated_edges.append(edge)
                # print(f"Violated edge: {edge.source} -> {edge.target}, score: {score}, type: {edge.edge_type}")
                if edge.edge_type == PreferenceEdgeType.UNANIMOUS:
                    po_violations += 1
                else:
                    pmc_violations += 1

        total = len(graph.edges)
        total_violations = len(violated_edges)

        # total_po = sum(
        #     edge.edge_type == PreferenceEdgeType.UNANIMOUS
        #     for edge in graph.edges
        # )

        # total_pmc = total - total_po

        return EvaluationResult(
            num_constraints=total,
            num_violations=total_violations,
            num_po_constraints=graph.num_po_edges,
            num_pmc_constraints=graph.num_pmc_edges,
            num_po_violations=po_violations,
            num_pmc_violations=pmc_violations,
            epsilon_support=epsilon_support,
            num_nonzero_epsilon=len(epsilon_support),
            epsilon_support_percentage=(
                100 * len(epsilon_support) / len(alternatives)
                if alternatives else 0.0
            ),
            z_support_percentage = (
                100 * len(z_support) / total
                if z_support else 0.0
            ),
            violation_percentage=100 * total_violations / total if total else 0.0,
            po_violation_percentage=100 * po_violations / graph.num_po_edges if graph.num_po_edges else 0.0,
            pmc_violation_percentage=100 * pmc_violations / graph.num_pmc_edges if graph.num_pmc_edges else 0.0,
            violated_edges=tuple(violated_edges),
            objective_value=result.objective_value,
            epsilon_l1_norm=epsilon_l1_norm,
        )