from __future__ import annotations
from copy import deepcopy
import networkx as nx
from ai_alignment_linear_model_extension.preference_graph.edge import MajorityEdge as Edge


class PreferenceGraph:
    """
    Thin wrapper around a NetworkX directed graph.

    This class exposes only the operations required by the
    preference-learning pipeline while hiding the underlying
    graph library.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    # -----------------------------------------------------
    # Basic graph construction
    # -----------------------------------------------------

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
        )

    # -----------------------------------------------------
    # Properties
    # -----------------------------------------------------

    @property
    def vertices(self) -> tuple[int, ...]:
        return tuple(self._graph.nodes)

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(
            Edge(
                source=u,
                target=v,
                weight=data.get("weight", 1.0),
            )
            for u, v, data in self._graph.edges(data=True)
        )

    @property
    def num_vertices(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    # -----------------------------------------------------
    # Graph algorithms
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Utilities
    # -----------------------------------------------------

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
            f"PreferenceGraph("
            f"num_vertices={self.num_vertices}, "
            f"num_edges={self.num_edges})"
        )