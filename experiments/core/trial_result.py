from __future__ import annotations

from dataclasses import dataclass

# from ai_alignment_linear_model_extension.evaluation
from ai_alignment_linear_model_extension.optimization.utills import ModelType
from ai_alignment_linear_model_extension.evaluation.evaluation_result import EvaluationResult
from experiments.core.timer import TimerResult



@dataclass(frozen=True, slots=True)
class TrialResult:
    trial: int
    seed: int
    evaluation_result: dict[ModelType, EvaluationResult]
    fas_size: int
    timers: tuple[TimerResult, ...]
    error: str | None = None
    success: bool = True
    # lp_result: LPResult | None = None