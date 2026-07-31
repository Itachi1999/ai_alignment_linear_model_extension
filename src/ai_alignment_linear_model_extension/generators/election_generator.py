# src/ai_alignment_linear_model_extension/generators/election_generator.py

from __future__ import annotations

from ai_alignment_linear_model_extension.generators.feature_generator import (
    FeatureGenerator,
)
from ai_alignment_linear_model_extension.generators.preference_generator import (
    PreferenceGenerator,
)
from ai_alignment_linear_model_extension.models.election import Election


class ElectionGenerator:
    """
    Generates complete Election objects by combining a feature generator
    and a ranking generator.
    """

    def __init__(
        self,
        feature_generator: FeatureGenerator,
        preference_generator: PreferenceGenerator,
    ) -> None:
        self._feature_generator = feature_generator
        self._preference_generator = preference_generator

    def generate(
        self,
        num_alternatives: int,
        num_voters: int,
        dimension: int,
    ) -> Election:
        """
        Generate a complete election.
        """

        # Generate alternatives
        alternatives = self._feature_generator.generate(
            num_alternatives=num_alternatives,
            dimension=dimension,
        )

        # Extract alternative IDs
        alternative_ids = tuple(
            alternative.id
            for alternative in alternatives
        )

        # Generate voters
        voters = self._preference_generator.generate(
            num_voters=num_voters,
            alternative_ids=alternative_ids,
        )

        # Construct validated election
        return Election(
            alternatives=alternatives,
            voters=voters,
        )