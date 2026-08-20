from __future__ import annotations

import cvxpy as cp
import numpy as np

from ai_alignment_linear_model_extension.models.election import Election
from ai_alignment_linear_model_extension.preference_graph.preference_graph import (
    PreferenceGraph, PreferenceEdgeType
)
from ai_alignment_linear_model_extension.optimization.lp_result import (
    LPResult,
)


class KTMinLinearModelSolver:
    """
    Solves the new LP with pairwise inversion variables z_ab.
    """

    def __init__(
        self,
        eta: float = 1e-6,
        lambda_: float = 1.0,
        solver: str = cp.HIGHS,
    ) -> None:
        self._eta = eta
        self._lambda = lambda_
        self._solver = solver

    def solve(
        self,
        election: Election,
        graph: PreferenceGraph,
    ) -> LPResult:

        d = election.dimension

        features = {
            alternative.id: alternative.features
            for alternative in election.alternatives
        }

        features_list = [
            alternative.features
            for alternative in election.alternatives
        ]

        delta = max(
            np.linalg.norm(x_a - x_b)
            for i, x_a in enumerate(features_list)
            for x_b in features_list[i + 1:]
        )

        self._L = np.sqrt(d) * delta

        # Variables

        theta = cp.Variable(d, name="theta")

        epsilon = {
            alternative.id: cp.Variable(
                name=f"epsilon_{alternative.id}"
            )
            for alternative in election.alternatives
        }

        t = {
            alternative.id: cp.Variable(
                nonneg=True,
                name=f"t_{alternative.id}",
            )
            for alternative in election.alternatives
        }

        z = {
            (edge.source, edge.target): cp.Variable(
                # boolean=True,
                nonneg = True,
                name=f"z_{edge.source}_{edge.target}",
            )
            for edge in graph.edges
        }

        # Constraints

        constraints: list[cp.Constraint] = []

        for edge in graph.edges:
            # if edge.edge_type == PreferenceEdgeType.UNANIMOUS:
            a = edge.source
            b = edge.target

            x_a = features[a]
            x_b = features[b]

            linear_score = theta @ (x_a - x_b)

            # Original preference constraint
            constraints.append(
                linear_score + epsilon[a] - epsilon[b] >= self._eta
            )

            # Pairwise inversion constraint
            constraints.append(
                linear_score + z[(a, b)]
                >= self._eta
            )

        # |epsilon_a| <= t_a
        for alternative in election.alternatives:
            a = alternative.id
            constraints.append(cp.abs(epsilon[a]) <= t[a])

        # ||theta||_inf <= 1
        constraints.append(
            cp.norm_inf(theta) <= 1
        )

        # Objective

        objective = cp.Minimize(
            cp.sum(list(t.values()))
            + self._lambda * cp.sum(list(z.values()))
        )

        problem = cp.Problem(
            objective,
            constraints,
        )

        # Solve
        problem.solve(
            solver=self._solver,
            highs_options={
                "primal_feasibility_tolerance": 1e-10,
                "dual_feasibility_tolerance": 1e-10,
            },
        )
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(
                f"LP solver failed with status '{problem.status}'."
            )
        # except Exception as e:
        #     return LPResult(
        #         theta=np.zeros(d),
        #         epsilon={
        #             alternative.id: float(0.0)
        #             for alternative in election.alternatives
        #         },
        #         z={
        #             edge_key: float(0.0)
        #             for edge_key, variable in z.items()
        #         },
        #         objective_value=float(0.0),
        #         status=str(problem.status),
        #     )
            

        # if problem.status not in (
        #     cp.OPTIMAL,
        #     cp.OPTIMAL_INACCURATE,
        # ):
        #     raise RuntimeError(
        #         f"Solver terminated with status "
        #         f"'{problem.status}'."
        #     )

        return LPResult(
            theta=np.asarray(theta.value).copy(),
            epsilon={
                alternative.id: float(epsilon[alternative.id].value)
                for alternative in election.alternatives
            },
            z={
                edge_key: float(variable.value)
                for edge_key, variable in z.items()
            },
            objective_value=float(problem.value),
            status=str(problem.status),
        )