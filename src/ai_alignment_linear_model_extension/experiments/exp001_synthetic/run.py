from ai_alignment_linear_model_extension.generators.feature_generator import GaussianFeatureGenerator
from ai_alignment_linear_model_extension.generators.preference_generator import UniformPreferenceGenerator
from ai_alignment_linear_model_extension.generators.election_generator import  ElectionGenerator
from ai_alignment_linear_model_extension.preference_graph.preference_graph import PreferenceGraphBuilder
from ai_alignment_linear_model_extension.preference_graph.FAS import FeedbackArcSetSolver
from ai_alignment_linear_model_extension.optimization.extension_linear_model import LinearModelSolver
from ai_alignment_linear_model_extension.evaluation.linear_model_evaluator import LinearModelEvaluator


def main() -> None:

    # --------------------------------------------------
    # Generate synthetic election
    # --------------------------------------------------

    generator = ElectionGenerator(
        feature_generator=GaussianFeatureGenerator(),
        preference_generator=UniformPreferenceGenerator(),
    )

    election = generator.generate(
        num_alternatives=4,
        num_voters=11,
        dimension=2,
    )

    # --------------------------------------------------
    # Build preference graph
    # --------------------------------------------------

    graph = PreferenceGraphBuilder().build(election)

    # --------------------------------------------------
    # Remove cycles
    # --------------------------------------------------

    fas_result = FeedbackArcSetSolver().solve(graph)

    dag = fas_result.dag

    # --------------------------------------------------
    # Solve LP
    # --------------------------------------------------

    lp_result = LinearModelSolver().solve(
        election,
        dag,
    )

    # --------------------------------------------------
    # Evaluate θ only
    # --------------------------------------------------

    evaluation = LinearModelEvaluator().evaluate(
        election,
        dag,
        lp_result,
    )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("=" * 50)
    print("Synthetic Election Summary")
    print("=" * 60)
    print("Election")
    print("=" * 60)
    print(election)
    print("Feature Matrix:")
    print(election.feature_matrix())
    print("Preference Matrix:")
    print(election.rankings())
    print("\n" + "=" * 60)
    print("Preference Graph")
    print("=" * 60)
    print(graph.edges)

    print("\nEdges:")
    for edge in graph.edges:
        print(edge)

    print("\n" + "=" * 60)
    print("Feedback Arc Set")
    print("=" * 60)

    print("Ordering:")
    print(fas_result.ordering)

    print("\nRemoved Edges:")
    for edge in fas_result.removed_edges:
        print(edge)

    print("\nDAG Edges:")
    for edge in dag.edges:
        print(edge)
    print("Objective :", lp_result.objective_value)

    print("Theta:")
    print(lp_result.theta)

    print()

    print("Violations")
    print(f"Total : {evaluation.num_violations}")
    print(f"PO    : {evaluation.num_po_violations}")
    print(f"PMC   : {evaluation.num_pmc_violations}")

    print(
        f"Violation % : "
        f"{evaluation.violation_percentage:.2f}%"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()