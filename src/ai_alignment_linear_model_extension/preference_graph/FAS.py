from __future__ import annotations
from dataclasses import dataclass

from ai_alignment_linear_model_extension.preference_graph.preference_graph import (
    PreferenceGraph,
)
from ai_alignment_linear_model_extension.preference_graph.edge import MajorityEdge as Edge



@dataclass(frozen=True, slots=True)
class FeedbackArcSetResult:
    """
    Result of the Feedback Arc Set computation.
    """

    dag: PreferenceGraph
    removed_edges: tuple[Edge, ...]
    ordering: tuple[int, ...]


class FeedbackArcSetSolver:
    """
    Computes a Feedback Arc Set using the indegree ordering.
    """

    def solve(
        self,
        graph: PreferenceGraph,
    ) -> FeedbackArcSetResult:

        # First, compute the indegree ordering of the vertices.
        ordering = self._compute_ordering(graph)
        
        # Check if graph is already a DAG
        if graph.is_dag():
            return FeedbackArcSetResult(dag=graph, removed_edges=(), ordering=ordering)

        # If the graph has cycles, remove all backward edges.
        else:
            return self._remove_backward_edges(
                graph,
                ordering,
            )

    def _compute_ordering(
        self,
        graph: PreferenceGraph,
    ) -> tuple[int, ...]:
        """
        Return the indegree ordering.
        """
        return tuple(sorted(graph.vertices, key=graph.in_degree))

    def _remove_backward_edges(
        self,
        graph: PreferenceGraph,
        ordering: tuple[int, ...],
    ) -> FeedbackArcSetResult:
        """
        Remove all backward edges.
        """
        position = {
            vertex: index
            for index, vertex in enumerate(ordering)
        }

        dag = PreferenceGraph()

        dag.add_vertices(graph.vertices)
        removed_edges = []
        for edge in graph.edges:

            if position[edge.source] < position[edge.target]:
                dag.add_edge(edge)
            else:
                # This is a backward edge, so we skip it.
                removed_edges.append(edge)

        return FeedbackArcSetResult(dag=dag, removed_edges=tuple(removed_edges), ordering=ordering)

