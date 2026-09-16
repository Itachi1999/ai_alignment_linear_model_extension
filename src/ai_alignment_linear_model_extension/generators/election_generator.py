from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

from ai_alignment_linear_model_extension.generators.feature_generator import (
    FeatureGenerator,
)
from ai_alignment_linear_model_extension.generators.preference_generator import (
    PreferenceGenerator,
)
from ai_alignment_linear_model_extension.models.election import Election, Alternative, Voter


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
            alternatives=alternatives,
        )

        # Construct validated election
        return Election(
            alternatives=alternatives,
            voters=voters,
        )

# class SOCElectionGenerator(ElectionGenerator):
#     """"
#     Generates a complete election from SOC data.
#     """

#     def __init__(
#         self,
#         feature_generator: FeatureGenerator,
#         preference_generator: PreferenceGenerator,
#         soc_data_path: str,
#     ) -> None:
#         super().__init__(feature_generator, preference_generator)


class RealElectionGenerator(ElectionGenerator):
    def __init__(
        self,
        d: int,
    ) -> None:
        self._d = d

    def generate(self, json_path: Path, is_pca: bool= True) -> Election:
        self._json_path = json_path
        with self._json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        alternative_vectors = data["alternative_feature_vectors"]

        alternative_ids = list(alternative_vectors.keys())

        X = np.asarray(
            [alternative_vectors[alternative_id] for alternative_id in alternative_ids],
            dtype=np.float64,
        )

        if self._d > X.shape[1]:
            raise ValueError(
                f"d={self._d} is invalid for this election. "
                f"Maximum PCA dimension is {min(X.shape)} "
                f"(m={X.shape[0]}, original dimension={X.shape[1]})."
            )

        if is_pca:
            X_reduced = PCA(n_components=self._d).fit_transform(X)

            alternatives = tuple(
                Alternative(
                    id=alternative_id,
                    dimension=self._d,
                    features=X_reduced[i],
                )
                for i, alternative_id in enumerate(alternative_ids)
            )
        else:
            alternatives = tuple(
                Alternative(
                    id=alternative_id,
                    dimension=self._d,
                    features=alternative_vectors[alternative_id],
                )
                for i, alternative_id in enumerate(alternative_ids)
            )

        voters = tuple(
            Voter(
                id=i,
                ranking=tuple(ranking),
            )
            for i, ranking in enumerate(data["preference_matrix"])
        )

        return Election(
            alternatives=alternatives,
            voters=voters,
        )