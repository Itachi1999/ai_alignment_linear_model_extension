from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from experiments.core.experiment_statistics import ExperimentResult
from experiments.core.trial_result import TrialResult
# from experiments.core.parameter_sweep_result import ParameterSweepResult


class ModelType(Enum):
    EPSILON_LP = auto()
    KT_MIN_LP = auto()
    BTL = auto()

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


@dataclass(frozen=True, slots=True)
class PlotRecord:
    """
    Stores a generated figure together with its name.
    """

    name: str
    figure: Figure


class ViolationPlots:
    """
    Visualization manager for LP/BTL experiment results.

    Supports:
        - single-experiment plots
        - parameter-sweep plots
        - model comparisons
        - PO/PMC/total violations
        - epsilon sparsity
        - epsilon/z support sizes
        - automatic figure collection
    """

    def __init__(
        self,
        result: ExperimentResult | ParameterSweepResult,
    ) -> None:
        self._result = result
        self._plots: list[PlotRecord] = []

    # =========================================================
    # Public API
    # =========================================================

    @property
    def plots(self) -> tuple[PlotRecord, ...]:
        return tuple(self._plots)

    @property
    def figures(self) -> tuple[Figure, ...]:
        return tuple(
            plot.figure
            for plot in self._plots
        )

    def clear(self) -> None:
        """
        Close and remove all generated figures.
        """

        for plot in self._plots:
            plt.close(plot.figure)

        self._plots.clear()

    def create_all(
        self,
        models: tuple[ModelName, ...] = (
            "old_lp",
            "new_lp",
            "btl",
        ),
    ) -> tuple[PlotRecord, ...]:

        if isinstance(self._result, ParameterSweepResult):

            self.plot_violation_vs_parameter(
                metric="violation_percentage",
                models=models,
            )

            self.plot_violation_vs_parameter(
                metric="po_violation_percentage",
                models=models,
            )

            self.plot_violation_vs_parameter(
                metric="pmc_violation_percentage",
                models=models,
            )

            self.plot_support_vs_parameter()

        else:

            self.plot_violation_rates(
                models=models,
            )

            self.plot_violation_distribution(
                models=models,
            )

            self.plot_po_vs_pmc(
                models=models,
            )

            self.plot_support_sizes()

            self.plot_epsilon_sparsity()

        return self.plots

    # =========================================================
    # Single experiment
    # =========================================================

    def plot_violation_rates(
        self,
        models: tuple[ModelName, ...] = (
            "old_lp",
            "new_lp",
            "btl",
        ),
    ) -> Figure:
        """
        Plot mean PO, PMC and total violation rates
        for several models in one figure.
        """

        metrics = (
            (
                "PO",
                "po_violation_percentage",
            ),
            (
                "PMC",
                "pmc_violation_percentage",
            ),
            (
                "Total",
                "violation_percentage",
            ),
        )

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        x = np.arange(len(models))
        width = 0.8 / len(metrics)

        for metric_index, (label, metric) in enumerate(
            metrics
        ):

            means = [
                self._mean_metric(
                    self._successful_trials(),
                    model,
                    metric,
                )
                for model in models
            ]

            axis.bar(
                x + metric_index * width,
                means,
                width=width,
                label=label,
            )

        axis.set_xticks(
            x + width * (len(metrics) - 1) / 2
        )

        axis.set_xticklabels(
            [self._model_label(model) for model in models]
        )

        axis.set_ylabel(
            "Violation Rate (%)"
        )

        axis.set_title(
            "Violation Rate Comparison"
        )

        axis.legend()

        figure.tight_layout()

        return self._store(
            "violation_rates",
            figure,
        )

    def plot_violation_distribution(
        self,
        models: tuple[ModelType, ...] = (
            ModelType.EPSILON_LP,
            ModelType.KT_MIN_LP,
            ModelType.BTL,
        ),
    ) -> Figure:

        trials = self._successful_trials()

        figure, axis = plt.subplots(figsize=(9, 5))

        for model in models:
            values = self._metric_values(
                trials,
                model,
                "violation_percentage",
            )

            axis.plot(
                range(1, len(values) + 1),
                values,
                label=self._model_label(model),
            )

        axis.set_xlabel("Trial")
        axis.set_ylabel("Violation Rate (%)")
        axis.set_title("Violation Rate Across Trials")
        axis.legend()

        figure.tight_layout()

        return self._store(
            "violation_across_trials",
            figure,
        )

    def plot_po_vs_pmc(
        self,
        models: tuple[ModelType, ...] = (
            ModelType.EPSILON_LP,
            ModelType.KT_MIN_LP,
            ModelType.BTL,
        ),
    ) -> Figure:
        """
        Compare mean PO and PMC violation rates.
        """

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        x = np.arange(2)

        for model in models:

            trials = self._successful_trials()

            po = self._mean_metric(
                trials,
                model,
                "po_violation_percentage",
            )

            pmc = self._mean_metric(
                trials,
                model,
                "pmc_violation_percentage",
            )

            axis.plot(
                x,
                [po, pmc],
                marker="o",
                label=self._model_label(model),
            )

        axis.set_xticks(x)
        axis.set_xticklabels(
            ["PO", "PMC"]
        )

        axis.set_ylabel(
            "Violation Rate (%)"
        )

        axis.set_title(
            "PO vs PMC Violations"
        )

        axis.legend()

        figure.tight_layout()

        return self._store(
            "po_vs_pmc",
            figure,
        )

    def plot_support_sizes(self) -> Figure:

        trials = self._successful_trials()

        figure, axis = plt.subplots(figsize=(9, 5))

        old_epsilon = [
            len(trial.evaluation_result.epsilon_support)
            for trial in trials
        ]

        new_epsilon = [
            len(trial.new_lp_evaluation_result.epsilon_support)
            for trial in trials
        ]

        z_support = [
            sum(
                abs(value) > 1e-9
                for value in trial.new_lp_result.z.values()
            )
            for trial in trials
        ]

        axis.plot(
            range(1, len(trials) + 1),
            old_epsilon,
            label="Old LP ε",
        )

        axis.plot(
            range(1, len(trials) + 1),
            new_epsilon,
            label="New LP ε",
        )

        axis.plot(
            range(1, len(trials) + 1),
            z_support,
            label="New LP z",
        )

        axis.set_xlabel("Trial")
        axis.set_ylabel("Support Size")
        axis.set_title("ε and z Support Across Trials")
        axis.legend()

        figure.tight_layout()

        return self._store(
            "support_across_trials",
            figure,
        )

    def plot_epsilon_sparsity(self) -> Figure:
        """
        Compare epsilon sparsity between old and new LP.
        """

        trials = self._successful_trials()

        old_values = [
            trial.evaluation_result.epsilon_sparsity
            for trial in trials
        ]

        new_values = [
            trial.new_lp_evaluation_result.epsilon_sparsity
            for trial in trials
        ]

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        axis.boxplot(
            [old_values, new_values],
            tick_labels=[
                "Old LP",
                "New LP",
            ],
        )

        axis.set_ylabel(
            "ε Sparsity"
        )

        axis.set_title(
            "ε Sparsity Comparison"
        )

        figure.tight_layout()

        return self._store(
            "epsilon_sparsity",
            figure,
        )

    # =========================================================
    # Parameter sweep
    # =========================================================

    def plot_violation_vs_parameter(
        self,
        metric: str,
        models: tuple[ModelType, ...] = (
            ModelType.EPSILON_LP,
            ModelType.KT_MIN_LP,
            ModelType.BTL
        ),
        error: Literal[
            "std",
            "sem",
        ] = "std",
    ) -> Figure:
        """
        Plot one metric against the swept parameter.

        All requested models appear on the same figure.
        """

        self._require_sweep()

        assert isinstance(
            self._result,
            ParameterSweepResult,
        )

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        for model in models:

            x_values: list[float] = []
            means: list[float] = []
            errors: list[float] = []

            for parameter, experiment_result in sorted(
                self._result.results.items()
            ):

                trials = [
                    trial
                    for trial in experiment_result.trials
                    if trial.success
                ]

                values = self._metric_values(
                    trials,
                    model,
                    metric,
                )

                if not values:
                    continue

                x_values.append(parameter)
                means.append(
                    float(np.mean(values))
                )

                errors.append(
                    self._error(values, error)
                )

            if not x_values:
                continue

            axis.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="o",
                capsize=4,
                label=self._model_label(model),
            )

        axis.set_xlabel(
            self._result.parameter_name
        )

        axis.set_ylabel(
            self._metric_label(metric)
        )

        axis.set_title(
            f"{self._metric_label(metric)} "
            f"vs {self._result.parameter_name}"
        )

        axis.legend()

        figure.tight_layout()

        return self._store(
            f"{metric}_vs_parameter",
            figure,
        )

    def plot_support_vs_parameter(
        self,
        error: Literal[
            "std",
            "sem",
        ] = "std",
    ) -> Figure:
        """
        Plot epsilon/z support size against the swept parameter.
        """

        self._require_sweep()

        assert isinstance(
            self._result,
            ParameterSweepResult,
        )

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        series = {
            "Old LP ε": lambda trial: len(
                trial.evaluation_result.epsilon_support
            ),
            "New LP ε": lambda trial: len(
                trial.new_lp_evaluation_result.epsilon_support
            ),
            "New LP z": lambda trial: sum(
                abs(value) > 1e-9
                for value in trial.new_lp_result.z.values()
            ),
        }

        for label, extractor in series.items():

            x_values: list[float] = []
            means: list[float] = []
            errors: list[float] = []

            for parameter, experiment_result in sorted(
                self._result.results.items()
            ):

                trials = [
                    trial
                    for trial in experiment_result.trials
                    if trial.success
                ]

                values = [
                    extractor(trial)
                    for trial in trials
                ]

                if not values:
                    continue

                x_values.append(parameter)
                means.append(
                    float(np.mean(values))
                )

                errors.append(
                    self._error(values, error)
                )

            axis.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="o",
                capsize=4,
                label=label,
            )

        axis.set_xlabel(
            self._result.parameter_name
        )

        axis.set_ylabel(
            "Support Size"
        )

        axis.set_title(
            f"Support Size vs "
            f"{self._result.parameter_name}"
        )

        axis.legend()

        figure.tight_layout()

        return self._store(
            "support_vs_parameter",
            figure,
        )

    # =========================================================
    # Internal helpers
    # =========================================================

    def _store(
        self,
        name: str,
        figure: Figure,
    ) -> Figure:
        self._plots.append(
            PlotRecord(
                name=name,
                figure=figure,
            )
        )
        return figure

    def _successful_trials(
        self,
    ) -> list[TrialResult]:

        if not isinstance(
            self._result,
            ExperimentResult,
        ):
            raise TypeError(
                "This plot requires a single ExperimentResult."
            )

        return [
            trial
            for trial in self._result.trials
            if trial.success
        ]

    @staticmethod
    def _metric_values(
        trials: list[TrialResult],
        model: ModelType,
        metric: str,
    ) -> list[float]:

        evaluation = ViolationPlots._evaluation_accessor(
            model
        )

        values: list[float] = []

        for trial in trials:

            result = evaluation(trial)

            value = getattr(
                result,
                metric,
                None,
            )

            if value is not None:
                values.append(
                    float(value)
                )

        return values

    @staticmethod
    def _mean_metric(
        trials: list[TrialResult],
        model: ModelType,
        metric: str,
    ) -> float:

        values = ViolationPlots._metric_values(
            trials,
            model,
            metric,
        )

        return (
            float(np.mean(values))
            if values
            else 0.0
        )

    @staticmethod
    def _evaluation_accessor(
        model: ModelType,
    ) -> Callable[[TrialResult], object]:

        if model == ModelType.EPSILON_LP:
            return lambda trial: (
                trial.evaluation_result
            )

        if model == ModelType.KT_MIN_LP:
            return lambda trial: (
                trial.new_lp_evaluation_result
            )

        if model == ModelType.BTL:
            return lambda trial: (
                trial.btl_evaluation_result
            )

        raise ValueError(
            f"Unknown model: {model}"
        )

    @staticmethod
    def _model_label(
        model: ModelType,
    ) -> str:

        return {
            ModelType.EPSILON_LP: "Old LP",
            ModelType.KT_MIN_LP: "New LP",
            ModelType.BTL: "BTL",
        }[model]

    @staticmethod
    def _metric_label(
        metric: str,
    ) -> str:

        labels = {
            "violation_percentage":
                "Violation Rate (%)",
            "po_violation_percentage":
                "PO Violation Rate (%)",
            "pmc_violation_percentage":
                "PMC Violation Rate (%)",
            "epsilon_sparsity":
                "ε Sparsity",
        }

        return labels.get(
            metric,
            metric.replace(
                "_",
                " ",
            ).title(),
        )

    @staticmethod
    def _error(
        values: list[float],
        error: Literal[
            "std",
            "sem",
        ],
    ) -> float:

        if len(values) <= 1:
            return 0.0

        standard_deviation = float(
            np.std(
                values,
                ddof=1,
            )
        )

        if error == "std":
            return standard_deviation

        return standard_deviation / np.sqrt(
            len(values)
        )

    def _require_sweep(self) -> None:

        if not isinstance(
            self._result,
            ParameterSweepResult,
        ):
            raise TypeError(
                "This plot requires a ParameterSweepResult."
            )