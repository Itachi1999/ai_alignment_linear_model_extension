from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from ai_alignment_linear_model_extension.models.alternative import Alternative
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class FeatureGenerator(ABC):
    """
    Abstract base class for generating alternative feature vectors.
    """

    @abstractmethod
    def generate(
        self,
        num_alternatives: int,
        dimension: int,
    ) -> tuple[Alternative, ...]:
        """
        Generate feature vectors for the alternatives.

        Parameters
        ----------
        num_alternatives : int
            Number of alternatives.

        dimension : int
            Feature dimension.

        Returns
        -------
        tuple[Alternative, ...]
            Generated alternatives.
        """
        pass


class GaussianFeatureGenerator(FeatureGenerator):
    """
    Generates feature vectors for alternatives using a Gaussian distribution.
    """
    def __init__(self, mean: float = 0.0, std_dev: float = 1.0, seed: int = 42) -> None:
        """
        Initialize the Gaussian feature generator.

        Parameters
        ----------
        mean : float, optional
            Mean of the Gaussian distribution (default is 0.0).
        std_dev : float, optional
            Standard deviation of the Gaussian distribution (default is 1.0).
        seed : int, optional
            Random seed for reproducibility (default is 42).
        """
        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive.")
        self.mean = mean
        self.std_dev = std_dev
        self._rng = np.random.default_rng(seed)  # Use numpy's Generator for reproducibility

    def generate(
        self,
        num_alternatives: int,
        dimension: int,
    ) -> tuple[Alternative, ...]:
        """
        Generate feature vectors for the alternatives.

        Parameters
        ----------
        num_alternatives : int
            Number of alternatives.

        dimension : int
            Feature dimension.

        Returns
        -------
        tuple[Alternative, ...]
            Generated alternatives.
        """
        alternatives = []
        if num_alternatives <= 0:
            raise ValueError("Number of alternatives must be positive.")

        if dimension <= 0:
            raise ValueError("Dimension must be positive.")
        
        for alt in range(num_alternatives):
            features = 10 * self._rng.normal(loc=self.mean, scale=self.std_dev, size=dimension)
            alternatives.append(Alternative(id=alt, dimension=dimension, features=features))
        # features = np.random.normal(size=(num_alternatives, dimension))
        return tuple(alternatives)
    

class EmbeddingFeatureGenerator(FeatureGenerator):
    """
    Generates feature vectors for alternatives using a provided embedding model and texts.
    """
    def __init__(self, model_name:str = "sentence-transformers/all-mpnet-base-v2") -> None:
        """
        Initialize the embedding feature generator.

        Parameters
        ----------
        model_name : str
            Name of the sentence transformer model to use.
        """
        self.model = SentenceTransformer(model_name)

    def generate(
        self,
        num_alternatives: int,
        dimension: int,
        alternatives_text: list[str] = None
    ) -> tuple[Alternative, ...]:
        """
        Generate feature vectors for the alternatives.

        Parameters
        ----------
        num_alternatives : int
            Number of alternatives.

        dimension : int
            Feature dimension.

        alternatives_text : list[str], optional
            Texts for the alternatives.

        Returns
        -------
        tuple[Alternative, ...]
            Generated alternatives.
        """
        if num_alternatives <= 0:
            raise ValueError("Number of alternatives must be positive.")

        if dimension <= 0:
            raise ValueError("Dimension must be positive.")

        alternatives = []
        if alternatives_text is not None:
            features_raw = self.model.encode(alternatives_text, normalize_embeddings=True)
            if features_raw.shape[1] > dimension:
                pca = PCA(n_components=dimension)
                features = pca.fit_transform(features_raw)
            else:
                features = features_raw
        if features.shape[0] != num_alternatives:
            raise ValueError("Number of alternatives does not match the number of provided texts.")
        for alt in range(features.shape[0]):
            alternatives.append(Alternative(id=alt, dimension=dimension, features=features[alt]))

        return tuple(alternatives)