from dataclasses import dataclass
from typing import Any
from ai_alignment_linear_model_extension.models.alternative import Alternative
from ai_alignment_linear_model_extension.models.election import Election
from ai_alignment_linear_model_extension.preference_graph.preference_graph import PreferenceGraph
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True, slots=True)
class TrainableLossResult:
    scores: dict[Any, float]
    theta: np.ndarray
    epsilon: dict[Any, float]
    objective: float
    original_objective: float
    converged: bool
    iterations: int
    # solve_time: float
    status: str



class LP3TrainableLossModel:
    """
    Trainable linear reward model with candidate-specific corrections.

    Objective:

        J(theta, epsilon) =
            0.5 * ||epsilon||_2^2
            + beta * sum softplus_tau(
                h_ab(theta) - (epsilon_a - epsilon_b)
            )
            + lambda * sum softplus_tau(h_ab(theta))
            + 0.5 * kappa * ||theta||_2^2

    where:

        h_ab(theta) = eta - theta.T @ (x_a - x_b)

    A single shared theta is learned across all elections.
    """

    def __init__(
        self,
        graph: PreferenceGraph,
        election: Election | None = None,
        alternatives: tuple[Alternative, ...] | None = None,
        dimension: int | None = None,
        eta: float = 1e-4,
        beta: float = 1.0,
        lambda_: float = 1.0,
        kappa: float = 1e-3,
        tau: float = 1e-4,
    ) -> None:

        # if not election:
        #     raise ValueError("At least one election is required.")

        if beta <= 0:
            raise ValueError("beta must be positive.")

        if lambda_ < 0:
            raise ValueError("lambda must be non-negative.")

        if kappa <= 0 or tau <= 0:
            raise ValueError("kappa and tau must be positive.")

        self.election = election
        self.graph = graph
        self.eta = eta
        self.beta = beta
        self.lambda_ = lambda_
        self.kappa = kappa
        self.tau = tau

        # Collect all alternative features.
        self._features = {}

        if alternatives is None:
            alternatives = election.alternatives
        self.alternatives = alternatives

        for alternative in alternatives:
            if alternative.id in self._features:
                raise ValueError(
                    f"Duplicate alternative ID: {alternative.id}"
                )

            self._features[alternative.id] = np.asarray(
                alternative.features,
                dtype=np.float64,
            )

        self._alternative_ids = tuple(self._features.keys())

        self._id_to_idx = {
            alternative_id: i
            for i, alternative_id in enumerate(self._alternative_ids)
        }

        if election is not None:
            self._dimension = self.election.dimension
        else:
            self._dimension = dimension

        # Extract preference edges.
        edges = list(graph.edges)

        if not edges:
            raise ValueError("Preference graph contains no edges.")

        self._source_idx = np.asarray([
            self._id_to_idx[edge.source]
            for edge in edges
        ], dtype=int)

        self._target_idx = np.asarray([
            self._id_to_idx[edge.target]
            for edge in edges
        ], dtype=int)

        # Pairwise feature difference matrix.
        self._edge_features = np.ascontiguousarray(
            np.vstack([
                self._features[edge.source]
                - self._features[edge.target]
                for edge in edges
            ]),
            dtype=np.float64,
        )

        if self._edge_features.shape[1] != self._dimension:
            raise ValueError("Inconsistent feature dimensions.")
        if self._edge_features.shape[0] != len(edges):
            raise ValueError("Inconsistent number of edges.")

        self.theta = np.zeros(self._dimension, dtype=np.float64)

    @property
    def dimension(self) -> int:
        return self._dimension

    # @property
    # def num_elections(self) -> int:
    #     return len(self.elections)

    @property
    def num_alternatives(self) -> int:
        return len(self._alternative_ids)

    @property
    def num_edges(self) -> int:
        return len(self._source_idx)

    def _objective_and_gradient(
        self,
        params: np.ndarray,
    ) -> tuple[float, np.ndarray]:

        d = self.dimension
        E = self.graph.num_edges

        theta = params[:d]
        epsilon = params[d:]

        beta = self.beta
        lam = self.lambda_
        kappa = self.kappa
        tau = self.tau

        source = self._source_idx
        target = self._target_idx
        A = self._edge_features

        # h_ab(theta) = eta - theta.T @ (x_a - x_b)
        h = self.eta - A @ theta

        # Corrected residual shortfall.
        z = h - epsilon[source] + epsilon[target]

        # Softplus approximations.
        loss_axiom = tau * np.logaddexp(0.0, z / tau)
        loss_deployment = tau * np.logaddexp(0.0, h / tau)

        # Individual objective components.
        correction = 0.5 * np.dot(epsilon, epsilon)

        axiom_violation = beta * np.sum(loss_axiom) / E

        deployment_violation = lam * np.sum(loss_deployment) / E

        theta_regularization = (
            0.5 * kappa * np.dot(theta, theta)
        )

        objective = (
            correction
            + axiom_violation
            + deployment_violation
            + theta_regularization
        )

        # Derivatives of softplus.
        p_axiom = expit(z / tau)
        p_deployment = expit(h / tau)

        # Gradient with respect to theta.
        gradient_theta = (
            -A.T @ (beta * p_axiom + lam * p_deployment) / E
            + kappa * theta 
        )

        # Gradient with respect to epsilon.
        gradient_epsilon = (
            epsilon
            - np.bincount(
                source,
                weights=beta * p_axiom / E,
                minlength=self.num_alternatives,
            )
            + np.bincount(
                target,
                weights=beta * p_axiom / E,
                minlength=self.num_alternatives,
            )
        )

        gradient = np.concatenate([
            gradient_theta,
            gradient_epsilon,
        ])

        return float(objective), gradient

    def fit(
        self,
        max_iter: int = 2000,
        ftol: float = 1e-10,
        gtol: float = 1e-6,
        maxcor: int = 20,
    ) -> TrainableLossResult:

        d = self.dimension
        m = self.num_alternatives
        E = self.graph.num_edges
        # Initialize both theta and epsilon to zero.
        initial_params = np.zeros(d + m, dtype=np.float64)

        result = minimize(
            fun=self._objective_and_gradient,
            x0=initial_params,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": max_iter,
                "ftol": ftol,
                "gtol": gtol,
                "maxcor": maxcor,
            },
        )


        self.theta = result.x[:d].copy()
        epsilon = result.x[d:].copy()

        # Recover candidate-specific corrections.
        epsilon_dict = {
            alternative_id: float(epsilon[i])
            for i, alternative_id in enumerate(self._alternative_ids)
        }

        # Deployed linear reward scores.
        scores = {
            alternative_id: float(
                self.theta @ self._features[alternative_id]
            )
            for alternative_id in self._alternative_ids
        }

        # Evaluate the original, non-smoothed objective.
        h = self.eta - self._edge_features @ self.theta

        z = (
            h
            - epsilon[self._source_idx]
            + epsilon[self._target_idx]
        )

        original_objective = (
            0.5 * np.dot(epsilon, epsilon)
            + self.beta * np.maximum(0.0, z).sum() / E
            + self.lambda_ * np.maximum(0.0, h).sum() / E
            + 0.5 * self.kappa * np.dot(self.theta, self.theta)
        )

        return TrainableLossResult(
            scores=scores,
            theta=self.theta.copy(),
            epsilon=epsilon_dict,
            objective=float(result.fun),
            original_objective=float(original_objective),
            converged=bool(result.success),
            iterations=int(result.nit),
            status=str(result.success),
        )