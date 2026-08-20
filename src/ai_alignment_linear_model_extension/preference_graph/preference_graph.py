from __future__ import annotations
from copy import deepcopy
import networkx as nx
import numpy as np

from ai_alignment_linear_model_extension.preference_graph.marginal_matrix import MarginalMatrix, MarginalMatrixBuilder
from ai_alignment_linear_model_extension.models.election import Election
from ai_alignment_linear_model_extension.preference_graph.edge import MajorityEdge as Edge, PreferenceEdgeType


class PreferenceGraph:
    """
    Thin wrapper around a NetworkX directed graph.

    This class exposes only the operations required by the
    preference-learning pipeline while hiding the underlying
    graph library.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    # Basic graph construction

    def add_vertex(self, vertex: int) -> None:
        """Add a vertex."""
        self._graph.add_node(vertex)

    def add_vertices(self, vertices: list[int] | tuple[int, ...]) -> None:
        """Add multiple vertices."""
        self._graph.add_nodes_from(vertices)

    def add_edge(self, edge: Edge) -> None:
        """Add a weighted directed edge."""
        self._graph.add_edge(
            edge.source,
            edge.target,
            weight=edge.weight,
            edge_type=edge.edge_type,
        )

    def in_degree(self, vertex: int) -> int:
        return self._graph.in_degree(vertex)

    def out_degree(self, vertex: int) -> int:
        return self._graph.out_degree(vertex)

    # Properties

    @property
    def vertices(self) -> tuple[int, ...]:
        return tuple(self._graph.nodes)

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(
            Edge(
                source=u,
                target=v,
                weight=data.get("weight"),
                edge_type=data.get("edge_type", PreferenceEdgeType.MAJORITY),
            )
            for u, v, data in self._graph.edges(data=True)
        )

    @property
    def num_vertices(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def num_po_edges(self) -> int:
        return sum(
            1 for _, _, data in self._graph.edges(data=True)
            if data.get("edge_type") == PreferenceEdgeType.UNANIMOUS
        )

    @property
    def num_pmc_edges(self) -> int:
        return sum(
            1 for _, _, data in self._graph.edges(data=True)
            if data.get("edge_type") == PreferenceEdgeType.MAJORITY
        )

    # Graph algorithms

    def is_dag(self) -> bool:
        """Return True iff the graph is acyclic."""
        return nx.is_directed_acyclic_graph(self._graph)

    def topological_order(self) -> tuple[int, ...]:
        """Return a topological ordering."""
        return tuple(nx.topological_sort(self._graph))

    def has_edge(self, source: int, target: int) -> bool:
        return self._graph.has_edge(source, target)

    def successors(self, vertex: int) -> tuple[int, ...]:
        return tuple(self._graph.successors(vertex))

    def predecessors(self, vertex: int) -> tuple[int, ...]:
        return tuple(self._graph.predecessors(vertex))

    # Utilities

    def copy(self) -> "PreferenceGraph":
        new_graph = PreferenceGraph()
        new_graph._graph = deepcopy(self._graph)
        return new_graph

    def to_networkx(self) -> nx.DiGraph:
        """
        Return the underlying NetworkX graph.

        Use this only when an algorithm from NetworkX is needed.
        """
        return self._graph

    def __str__(self) -> str:
        return (
            f"""Preference Graph(
                Number of Vertices = {self._graph.number_of_nodes},
                Number of Edges = {self._graph.number_of_edges},
                Edges = {self._graph.edges}
            )"""
        )



class PreferenceGraphBuilder:
    """
    Builds the preference graph from an election.
    """

    def __init__(self):
        self._marginal_matrix_builder = MarginalMatrixBuilder()

    def build(
        self,
        election: Election,
    ) -> tuple[MarginalMatrix, PreferenceGraph]:

        marginal_matrix = self._marginal_matrix_builder.build(election)
        # print("Marginal Matrix:", marginal_matrix.matrix)
        return marginal_matrix, self._build_pmc_graph(marginal_matrix)

    def _build_pmc_graph(
        self,
        marginal_matrix: MarginalMatrix,
    ) -> PreferenceGraph:

        graph = PreferenceGraph()

        graph.add_vertices(
            marginal_matrix.alternative_ids
        )

        alternative_ids = marginal_matrix.alternative_ids

        for i, a in enumerate(alternative_ids):
            for b in alternative_ids[i + 1:]:

                if marginal_matrix.majority_prefers(a, b):
                    wab = marginal_matrix.weight(a, b)
                    # print(f"Weight of edge {a} -> {b}: {wab}")

                    edge_type = PreferenceEdgeType.UNANIMOUS if np.isclose(wab, 1.0) else PreferenceEdgeType.MAJORITY

                    graph.add_edge(
                        Edge(
                            source=a,
                            target=b,
                            weight=wab,
                            edge_type=edge_type,
                        )
                    )

                elif marginal_matrix.majority_prefers(b, a):
                    wba = marginal_matrix.weight(b, a)
                    # print(f"Weight of edge {b} -> {a}: {wba}")
                    edge_type = PreferenceEdgeType.UNANIMOUS if np.isclose(wba, 1.0) else PreferenceEdgeType.MAJORITY

                    graph.add_edge(
                        Edge(
                            source=b,
                            target=a,
                            weight=wba,
                            edge_type=edge_type,
                        )
                    )

                else:
                    # ties -> no edge
                    pass
        return graph