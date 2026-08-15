from __future__ import annotations

from dataclasses import dataclass
from matplotlib.figure import Figure

from experiments.core.experiment_statistics import ExperimentResult

@dataclass(frozen=True, slots=True)
class PlotRecord:
    name: str
    figure: Figure
    
    
@dataclass(frozen=True, slots=True)
class ParameterSweepResult:
    """
    Collection of experiment results indexed by a swept parameter.
    """

    parameter_name: str
    results: dict[float, ExperimentResult]

    @property
    def parameter_values(self) -> tuple[float, ...]:
        return tuple(sorted(self.results.keys()))

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, parameter_value: float) -> ExperimentResult:
        return self.results[parameter_value]