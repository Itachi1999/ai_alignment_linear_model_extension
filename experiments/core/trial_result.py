from __future__ import annotations

from dataclasses import dataclass

from ai_alignment_linear_model_extension.optimization.lp_result import LPResult
from ai_alignment_linear_model_extension.evaluation.evaluation_result import (
    EvaluationResult,
)

from experiments.core.timer import TimerResult


@dataclass(frozen=True, slots=True)
class TrialResult:
    trial: int
    seed: int

    lp_result: LPResult
    evaluation_result: EvaluationResult

    fas_size: int

    timers: tuple[TimerResult, ...]