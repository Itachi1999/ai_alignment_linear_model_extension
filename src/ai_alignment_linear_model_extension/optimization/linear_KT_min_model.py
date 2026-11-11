from __future__ import annotations

from ai_alignment_linear_model_extension.models import election
import cvxpy as cp
import numpy as np
from scipy.sparse import coo_matrix

from ai_alignment_linear_model_extension.models.election import Election, Alternative
from ai_alignment_linear_model_extension.preference_graph.preference_graph import (
    PreferenceGraph, PreferenceEdgeType
)
from ai_alignment_linear_model_extension.optimization.lp_result import (
    LPResult,
)
from ai_alignment_linear_model_extension.models.borda import Borda


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
        graph: PreferenceGraph,
        election: Election | None = None,
        alternatives: tuple[Alternative, ...] | None = None, 
        dimension: int | None = None
    ) -> LPResult:

        if election is None and alternatives is None and dimension is None:
            raise ValueError("At least one of Election or Alternatives (along with dimension must be given)")
        
        if election is not None:
            d = election.dimension
            alternatives = election.alternatives
        else:
            d = dimension

        edges = list(graph.edges)
        alternative_ids = [alt.id for alt in alternatives]

        m = len(alternative_ids)
        E = len(edges)

        id_to_idx = {
            alt_id: i for i, alt_id in enumerate(alternative_ids)
        }

        X = np.vstack([
            alt.features for alt in alternatives
        ])

        source = np.array([
            id_to_idx[edge.source] for edge in edges
        ])

        target = np.array([
            id_to_idx[edge.target] for edge in edges
        ])


        # Pairwise feature differences: (E, d)
        X_diff = X[source] - X[target]

        # Sparse incidence matrix: (E, m)
        rows = np.repeat(np.arange(E), 2)
        cols = np.column_stack((source, target)).ravel()
        data = np.tile([1.0, -1.0], E)

        D = coo_matrix(
            (data, (rows, cols)),
            shape=(E, m),
        ).tocsr()

        theta = cp.Variable(d)
        epsilon = cp.Variable(m)
        t = cp.Variable(m, nonneg=True)
        z = cp.Variable(E, nonneg=True)

        constraints = [
            X_diff @ theta + D @ epsilon >= self._eta,
            z >= self._eta - X_diff @ theta,
            epsilon <= t,
            -epsilon <= t,
            theta <= 1,
            theta >= -1,
        ]

        objective = cp.Minimize(cp.sum(t) + self._lambda * cp.sum(z))

        problem = cp.Problem(
            objective,
            constraints,
        )
        
        # Solve
        problem.solve(
            solver=self._solver,
            highs_options={
                "solver": "ipm"
            },
        )
        
        epsilon_dict = {
            alt_id: float(epsilon.value[i])
            for i, alt_id in enumerate(alternative_ids)
        }

        t_dict = {
            alt_id: float(t.value[i])
            for i, alt_id in enumerate(alternative_ids)
        }

        z_dict = {
            (edge.source, edge.target): float(z.value[i])
            for i, edge in enumerate(edges)
        }
        print(f"LP compilation time: {problem.compilation_time}, status: {problem.status}, objective value: {problem.value}, LP time: {problem.solver_stats}")
        return LPResult(
            theta=np.asarray(theta.value).copy(),
            epsilon=epsilon_dict,
            z=z_dict,
            objective_value=float(problem.value),
            status=str(problem.status),
        )
        # features = {
        #     alternative.id: alternative.features
        #     for alternative in alternatives
        # }

        # features_list = [
        #     alternative.features
        #     for alternative in alternatives
        # ]

        # delta = max(
        #     np.linalg.norm(x_a - x_b)
        #     for i, x_a in enumerate(features_list)
        #     for x_b in features_list[i + 1:]
        # )

        # self._L = np.sqrt(d) * delta

        # # Variables

        # theta = cp.Variable(d, name="theta")

        # epsilon = {
        #     alternative.id: cp.Variable(
        #         name=f"epsilon_{alternative.id}"
        #     )
        #     for alternative in alternatives
        # }

        # t = {
        #     alternative.id: cp.Variable(
        #         nonneg=True,
        #         name=f"t_{alternative.id}",
        #     )
        #     for alternative in alternatives
        # }

        # z = {
        #     (edge.source, edge.target): cp.Variable(
        #         # boolean=True,
        #         nonneg = True,
        #         name=f"z_{edge.source}_{edge.target}",
        #     )
        #     for edge in graph.edges
        # }

        # # m = len(alternatives)
        # # E = len(graph.edges)

        # # Constraints

        # constraints: list[cp.Constraint] = []

        # for edge in graph.edges:
        #     # if edge.edge_type == PreferenceEdgeType.UNANIMOUS:
        #     #     eta0 = self._eta
        #     # else:
        #     #     eta0 = 1e-2 * self._eta 
        #     a = edge.source
        #     b = edge.target

        #     x_a = features[a]
        #     x_b = features[b]

        #     linear_score = theta @ (x_a - x_b)

        #     # Original preference constraint
        #     constraints.append(
        #         linear_score + epsilon[a] - epsilon[b] >= self._eta
        #     )

        #     # Pairwise inversion constraint
        #     constraints.append(
        #         linear_score + z[(a, b)]
        #         >= self._eta
        #     )
        # # edges = list(graph.edges)
        # # X_diff = np.vstack([
        # #     features[edge.source] - features[edge.target]
        # #     for edge in edges
        # # ])

        # # epsilon_diff = cp.hstack([
        # #     epsilon[edge.source] - epsilon[edge.target]
        # #     for edge in edges
        # # ])

        # # constraints.append(X_diff @ theta + epsilon_diff >= self._eta)

        # # z_vector = cp.hstack([ z[(edge.source, edge.target)] for edge in edges])
        # # constraints.append(X_diff @ theta + z_vector >= self._eta)

        # # |epsilon_a| <= t_a
        # for alternative in alternatives:
        #     a = alternative.id
        #     constraints.append(cp.abs(epsilon[a]) <= t[a])

        # # ||theta||_inf <= 1
        # constraints.append(
        #     cp.norm_inf(theta) <= 1
        # )

        # # This is done becuase the cp varaibles are in dictionary format
        # # epsilon_vector = cp.hstack([epsilon[alternative.id] for alternative in alternatives])
        # # t_vector = cp.hstack([t[alternative.id] for alternative in alternatives])

        # # constraints.append(cp.abs(epsilon_vector) <= t_vector)
        # # constraints.append(cp.norm_inf(theta) <= 1)

        # # Objective         
        # objective = cp.Minimize(
        #     cp.sum(list(t.values()))
        #     + self._lambda * cp.sum(list(z.values()))
        # )
        # # objective = cp.Minimize(
        # #     cp.sum(t_vector)
        # #     + self._lambda * cp.sum(z_vector)
        # # )
        # problem = cp.Problem(
        #     objective,
        #     constraints,
        # )

        # # Solve
        # problem.solve(
        #     solver=self._solver,
        #     highs_options={
        #         "solver": "ipm"
        #     },
        # )
        # if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        #     raise RuntimeError(
        #         f"LP solver failed with status '{problem.status}'."
        #     )
        # print(f"LP compilation time: {problem.compilation_time}, status: {problem.status}, objective value: {problem.value}, LP time: {problem.solver_stats}")
        # return LPResult(
        #     theta=np.asarray(theta.value).copy(),
        #     epsilon={
        #         alternative.id: float(epsilon[alternative.id].value)
        #         for alternative in alternatives
        #     },
        #     z={
        #         edge_key: float(variable.value)
        #         for edge_key, variable in z.items()
        #     },
        #     objective_value=float(problem.value),
        #     status=str(problem.status),
        # )
        
        



class WeightedKTMinLinearModelSolver:
    """
    Solves the new LP with pairwise inversion variables z_ab and the z_ab are weighted by w_ab.
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
            #     eta0 = self._eta
            # else:
            #     eta0 = 1e-2 * self._eta 
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
        y = {}
        for edge in graph.edges:
            a = edge.source
            b = edge.target
            y[(a, b)] = z[(a, b)] * edge.weight
            
        objective = cp.Minimize(
            cp.sum(list(t.values()))
            + self._lambda * cp.sum(list(y.values()))
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
        
        

class BordaKTMinLinearModelSolver:
    """
    Solves the new LP with pairwise inversion variables z_ab but only for the ones which satisfy borda score constraint.
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
            #     eta0 = self._eta
            # else:
            #     eta0 = 1e-2 * self._eta 
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
        
        borda = Borda(election)
        borda_scores = borda.scores()
        
        y = {}
        for edge in graph.edges:
            a = edge.source
            b = edge.target
            
            if borda_scores[a] < borda_scores[b]:
                y[(a, b)] = 0.0 * z[(a, b)]  # This effectively removes the z_ab variable from the objective for this edge
            else:   
                y[(a, b)] = z[(a, b)]

        # Objective         
        objective = cp.Minimize(
            cp.sum(list(t.values()))
            + self._lambda * cp.sum(list(y.values()))
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