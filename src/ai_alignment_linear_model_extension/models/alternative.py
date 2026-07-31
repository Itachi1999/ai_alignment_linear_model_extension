from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class Alternative:
    """Represents an alternative with an associated feature vector of specified dimension.
    Alternatives are used in the context of linear models for AI alignment where each alternatives or candidates are LLM responses, where each alternative is characterized by a unique identifier, a dimension indicating the size of its feature vector, and the feature vector itself.
    """

    id: int
    dimension: int
    features: np.ndarray

    def __post_init__(self) -> None:
        # Ensure features is a NumPy array
        object.__setattr__(self, "features", np.asarray(self.features, dtype=float))

        # Validate dimensionality
        if self.features.ndim != 1:
            raise ValueError(
                f"Feature vector must be one-dimensional, got shape {self.features.shape}."
            )

        if self.dimension <= 0:
            raise ValueError("Dimension must be positive.")

        if len(self.features) != self.dimension:
            raise ValueError(
                f"Expected feature vector of dimension {self.dimension},got {len(self.features)}."
            )

    def __str__(self) -> str:
        return (
            f"Alternative(id={self.id}, "
            f"dimension={self.dimension}, "
            f"features={self.features.tolist()})"
        )