from __future__ import annotations

from dataclasses import dataclass

from experiments.core.trial_result import TrialResult, ModelType



@dataclass(frozen=True, slots=True)
class ExperimentStatistics:
    num_trials: int

    mean_violation_percentage: float
    std_violation_percentage: float

    mean_po_violation_percentage: float
    std_po_violation_percentage: float

    mean_pmc_violation_percentage: float
    std_pmc_violation_percentage: float

    mean_fas_size: float
    std_fas_size: float

    success_rate: float

    mean_epsilon_support_percentage: float | None = None
    std_epsilon_support_percentage: float | None = None

    mean_epsilon_sparsity: float | None = None
    std_epsilon_sparsity: float | None = None
    
    mean_z_support_percentage: float | None = None
    std_z_support_percentage: float | None = None

    mean_loss: float | None = None
    std_loss: float | None = None

    mean_objective: float | None = None
    std_objective: float | None = None
    

@dataclass(frozen=True, slots=True)
class ExperimentResult:
    trials: tuple[TrialResult, ...]
    statistics: dict[ModelType, ExperimentStatistics]