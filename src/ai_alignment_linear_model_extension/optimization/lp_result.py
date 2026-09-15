from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class LPResult:
    """
    Result of solving the linear programming problem of the AI alignment model.
    """
    theta: np.ndarray
    epsilon: dict[int | str, float]
    objective_value: float
    status: str
    z: dict[tuple[int | str, ...], float] | None = None