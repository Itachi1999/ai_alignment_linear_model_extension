# optimization/linear_model_solver.py

from __future__ import annotations

import cvxpy as cp
import numpy as np

from ai_alignment_linear_model_extension.models.election import Election
from ai_alignment_linear_model_extension.preference_graph.preference_graph import (
    PreferenceGraph,
)
from ai_alignment_linear_model_extension.optimization.lp_result import (
    LPResult,
)


class LinearModelSolver:
    """
    Solves the linear preference learning LP.
    """

    def __init__(
        self,
        eta: float = 1e-6,
        solver: str = cp.HIGHS,
    ) -> None:
        """Constructor method of the LinearModelSolver class

        Args:
            eta (float, optional): Margin for the preference constraints. Defaults to 1e-6.
            solver (str, optional): The optimization solver to use. Defaults to "HIGHS".
        """
        self._eta = eta
        self._solver = solver

    def solve(
        self,
        election: Election,
        graph: PreferenceGraph,
    ) -> LPResult:

        d = election.dimension

        # --------------------------------------------------
        # Cache feature vectors
        # --------------------------------------------------

        features = {
            alternative.id: alternative.features
            for alternative in election.alternatives
        }

        # --------------------------------------------------
        # Decision variables
        # --------------------------------------------------

        theta = cp.Variable(
            shape=d,
            name="theta",
        )

        epsilon = {
            alternative.id: cp.Variable(
                name=f"epsilon_{alternative.id}"
            )
            for alternative in election.alternatives
        }

        t = {
            alternative.id: cp.Variable(
                nonneg=True,
                name=f"t_{alternative.id}"
            )
            for alternative in election.alternatives
        }

        # --------------------------------------------------
        # Constraints
        # --------------------------------------------------

        constraints: list[cp.Constraint] = []

        # Preference graph constraints

        for edge in graph.edges:

            x_a = features[edge.source]
            x_b = features[edge.target]

            constraints.append(
                (theta @ (x_a - x_b))
                + (epsilon[edge.source]
                - epsilon[edge.target])
                >= self._eta
            )

        # Absolute value constraints

        for alternative in election.alternatives:

            a = alternative.id

            # constraints.append(
            #     epsilon[a] <= t[a]
            # )

            # constraints.append(
            #     -epsilon[a] <= t[a]
            # )
            constraints.append(
                cp.abs(epsilon[a]) <= t[a]
            ) 

        constraints.append(
            cp.norm_inf(theta) <= 1
        )
            
        # --------------------------------------------------
        # Objective
        # --------------------------------------------------

        objective = cp.Minimize(
            cp.sum(list(t.values()))
        )

        # --------------------------------------------------
        # Solve
        # --------------------------------------------------

        problem = cp.Problem(
            objective,
            constraints,
        )

        problem.solve(
            solver=self._solver,
        )
        
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(
                f"LP solver failed with status '{problem.status}'."
            )
        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return LPResult(
            theta=theta.value.copy(),
            epsilon={
                alternative.id: float(epsilon[alternative.id].value)
                for alternative in election.alternatives
            },
            objective_value=float(problem.value),
            status=str(problem.status),
            # solver=str(self._solver),
        )
        # raise NotImplementedError