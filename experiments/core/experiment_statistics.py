from __future__ import annotations

from dataclasses import dataclass

from experiments.core.trial_result import TrialResult



@dataclass(frozen=True, slots=True)
class ExperimentStatistics:
    num_trials: int

    mean_objective: float
    std_objective: float

    mean_violation: float
    std_violation: float

    mean_po_violation: float
    std_po_violation: float

    mean_pmc_violation: float
    std_pmc_violation: float

    mean_fas_size: float
    std_fas_size: float

    mean_epsilon_support_percentage: float
    std_epsilon_support_percentage: float

    mean_epsilon_sparsity: float
    std_epsilon_sparsity: float

    success_rate: float


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    trials: tuple[TrialResult, ...]
    statistics: ExperimentStatistics