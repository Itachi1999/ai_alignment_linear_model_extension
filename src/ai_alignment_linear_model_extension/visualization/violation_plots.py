from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from experiments.core.experiment_statistics import ExperimentResult
from experiments.core.trial_result import TrialResult
from ai_alignment_linear_model_extension.visualization.plot_data import PlotRecord, ParameterSweepResult
# from experiments.core.parameter_sweep_result import ParameterSweepResult


class ModelType(Enum):
    EPSILON_LP = auto()
    KT_MIN_LP = auto()
    BTL = auto()
    
    @property
    def label(self) -> str:
        return {
            ModelType.EPSILON_LP: "Old LP",
            ModelType.KT_MIN_LP: "New LP",
            ModelType.BTL: "BTL",
        }[self]

Metric = Literal[
    "violation_percentage",
    "po_violation_percentage",
    "pmc_violation_percentage",
]


class ViolationPlots:
    """Create and manage violation-related figures."""

    def __init__(
        self,
        results: dict[ModelType, ExperimentResult] | ParameterSweepResult,
        ) -> None:
        self._results = results
        self._plots: list[PlotRecord] = []

    # =========================================================
    # Public properties
    # =========================================================

    @property
    def plots(self) -> tuple[PlotRecord, ...]:
        return tuple(self._plots)

    @property
    def figures(self) -> tuple[Figure, ...]:
        return tuple(plot.figure for plot in self._plots)

    # =========================================================
    # Plot management
    # =========================================================

    def clear(self) -> None:
        for plot in self._plots:
            plt.close(plot.figure)

        self._plots.clear()

    def create_all(self) -> tuple[PlotRecord, ...]:
        if isinstance(self._results, ParameterSweepResult):
            self.plot_violation_vs_parameter(
                "violation_percentage"
            )
            self.plot_violation_vs_parameter(
                "po_violation_percentage"
            )
            self.plot_violation_vs_parameter(
                "pmc_violation_percentage"
            )
            self.plot_support_vs_parameter()
        else:
            self.plot_violation_rates()
            self.plot_violation_across_trials()
            self.plot_po_vs_pmc()
            self.plot_epsilon_support_across_trials()
            self.plot_z_support_across_trials()
            self.plot_epsilon_sparsity()

        return self.plots

    # =========================================================
    # Single experiment
    # =========================================================

    def plot_violation_rates(self) -> Figure:
        """Mean PO, PMC and total violation rate by model."""

        models = tuple(self._require_single().keys())

        metrics = (
            ("PO", "po_violation_percentage"),
            ("PMC", "pmc_violation_percentage"),
            ("Total", "violation_percentage"),
        )

        figure, axis = plt.subplots(figsize=(9, 5))

        x = np.arange(len(models))
        width = 0.8 / len(metrics)

        for index, (label, metric) in enumerate(metrics):
            values = [
                self._mean_metric(
                    self._require_single()[model],
                    metric,
                )
                for model in models
            ]

            axis.bar(
                x + index * width,
                values,
                width,
                label=label,
            )

        axis.set_xticks(
            x + width * (len(metrics) - 1) / 2
        )
        axis.set_xticklabels(
            [model.label for model in models]
        )
        axis.set_ylabel("Violation Rate (%)")
        axis.set_title("Violation Rate Comparison")
        axis.legend()

        figure.tight_layout()

        return self._store("violation_rates", figure)

    def plot_violation_across_trials(self) -> Figure:
        """Total violation rate across trials for each model."""

        results = self._require_single()

        figure, axis = plt.subplots(figsize=(9, 5))

        for model, result in results.items():
            trials = self._successful_trials(result)

            values = [
                trial.evaluation_result.violation_percentage
                for trial in trials
            ]

            axis.plot(
                range(1, len(values) + 1),
                values,
                label=model.label,
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

    def plot_po_vs_pmc(self) -> Figure:
        """Mean PO and PMC violation rate by model."""

        results = self._require_single()

        figure, axis = plt.subplots(figsize=(9, 5))

        x = np.arange(2)

        for model, result in results.items():
            po = self._mean_metric(
                result,
                "po_violation_percentage",
            )
            pmc = self._mean_metric(
                result,
                "pmc_violation_percentage",
            )

            axis.plot(
                x,
                [po, pmc],
                marker="o",
                label=model.label,
            )

        axis.set_xticks(x)
        axis.set_xticklabels(("PO", "PMC"))
        axis.set_ylabel("Violation Rate (%)")
        axis.set_title("PO vs PMC Violations")
        axis.legend()

        figure.tight_layout()

        return self._store("po_vs_pmc", figure)

    def plot_epsilon_support_across_trials(self) -> Figure:
        """Epsilon support size across trials for LP models."""

        results = self._require_single()

        figure, axis = plt.subplots(figsize=(9, 5))

        for model in (
            ModelType.EPSILON_LP,
            ModelType.KT_MIN_LP,
        ):
            result = results.get(model)
            if result is None:
                continue

            trials = self._successful_trials(result)

            values = [
                len(trial.evaluation_result.epsilon_support)
                for trial in trials
                if trial.evaluation_result.epsilon_support is not None
            ]

            axis.plot(
                range(1, len(values) + 1),
                values,
                label=model.label,
            )

        axis.set_xlabel("Trial")
        axis.set_ylabel("ε Support Size")
        axis.set_title("ε Support Across Trials")
        axis.legend()

        figure.tight_layout()

        return self._store(
            "epsilon_support_across_trials",
            figure,
        )

    def plot_z_support_across_trials(self) -> Figure:
        """z support size across trials for the new LP."""

        results = self._require_single()
        result = results.get(ModelType.NEW_LP)

        if result is None:
            raise ValueError("New LP result is required.")

        trials = self._successful_trials(result)

        values = [
            self._z_support_size(trial)
            for trial in trials
        ]

        figure, axis = plt.subplots(figsize=(9, 5))

        axis.plot(
            range(1, len(values) + 1),
            values,
            label="New LP z",
        )

        axis.set_xlabel("Trial")
        axis.set_ylabel("z Support Size")
        axis.set_title("z Support Across Trials")
        axis.legend()

        figure.tight_layout()

        return self._store(
            "z_support_across_trials",
            figure,
        )

    def plot_epsilon_sparsity(self) -> Figure:
        """Epsilon sparsity across trials for LP models."""

        results = self._require_single()

        figure, axis = plt.subplots(figsize=(9, 5))

        for model in (
            ModelType.EPSILON_LP,
            ModelType.KT_MIN_LP,
        ):
            result = results.get(model)

            if result is None:
                continue

            trials = self._successful_trials(result)

            values = [
                trial.evaluation_result.epsilon_sparsity
                for trial in trials
            ]

            axis.plot(
                range(1, len(values) + 1),
                values,
                label=model.label,
            )

        axis.set_xlabel("Trial")
        axis.set_ylabel("ε Sparsity")
        axis.set_title("ε Sparsity Across Trials")
        axis.legend()

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
        metric: Metric,
        error: Literal["std", "sem"] = "std",
    ) -> Figure:
        """Plot all models against the swept parameter."""

        sweep = self._require_sweep()

        figure, axis = plt.subplots(figsize=(9, 5))

        for model in ModelType:
            x_values: list[float] = []
            means: list[float] = []
            errors: list[float] = []

            for parameter, model_results in sorted(
                sweep.results.items()
            ):
                result = model_results.get(model)

                if result is None:
                    continue

                values = self._metric_values(
                    result,
                    metric,
                )

                if not values:
                    continue

                x_values.append(parameter)
                means.append(float(np.mean(values)))
                errors.append(
                    self._calculate_error(
                        values,
                        error,
                    )
                )

            if x_values:
                axis.errorbar(
                    x_values,
                    means,
                    yerr=errors,
                    marker="o",
                    capsize=4,
                    label=model.label,
                )

        axis.set_xlabel(sweep.parameter_name)
        axis.set_ylabel(self._metric_label(metric))
        axis.set_title(
            f"{self._metric_label(metric)} "
            f"vs {sweep.parameter_name}"
        )
        axis.legend()

        figure.tight_layout()

        return self._store(
            f"{metric}_vs_parameter",
            figure,
        )

    def plot_support_vs_parameter(
        self,
        error: Literal["std", "sem"] = "std",
    ) -> Figure:
        """Plot LP epsilon and new-LP z support against a parameter."""

        sweep = self._require_sweep()

        figure, axis = plt.subplots(figsize=(9, 5))

        series = {
            "Old LP ε": ModelType.EPSILON_LP,
            "New LP ε": ModelType.KT_MIN_LP,
        }

        for label, model in series.items():
            x_values: list[float] = []
            means: list[float] = []
            errors: list[float] = []

            for parameter, model_results in sorted(
                sweep.results.items()
            ):
                result = model_results.get(model)

                if result is None:
                    continue

                trials = self._successful_trials(result)

                values = [
                    len(trial.evaluation_result.epsilon_support)
                    for trial in trials
                    if trial.evaluation_result.epsilon_support
                    is not None
                ]

                if not values:
                    continue

                x_values.append(parameter)
                means.append(float(np.mean(values)))
                errors.append(
                    self._calculate_error(values, error)
                )

            axis.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="o",
                capsize=4,
                label=label,
            )

        # z support for new LP
        x_values = []
        means = []
        errors = []

        for parameter, model_results in sorted(
            sweep.results.items()
        ):
            result = model_results.get(
                ModelType.KT_MIN_LP
            )

            if result is None:
                continue

            trials = self._successful_trials(result)

            values = [
                self._z_support_size(trial)
                for trial in trials
            ]

            if not values:
                continue

            x_values.append(parameter)
            means.append(float(np.mean(values)))
            errors.append(
                self._calculate_error(values, error)
            )

        axis.errorbar(
            x_values,
            means,
            yerr=errors,
            marker="o",
            capsize=4,
            label="New LP z",
        )

        axis.set_xlabel(sweep.parameter_name)
        axis.set_ylabel("Support Size")
        axis.set_title(
            f"Support Size vs {sweep.parameter_name}"
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

    def _require_single(
        self,
    ) -> dict[ModelType, ExperimentResult]:

        if not isinstance(
            self._results,
            dict,
        ):
            raise TypeError(
                "This plot requires model-wise ExperimentResult."
            )

        return self._results

    def _require_sweep(
        self,
    ) -> ParameterSweepResult:

        if not isinstance(
            self._results,
            ParameterSweepResult,
        ):
            raise TypeError(
                "This plot requires ParameterSweepResult."
            )

        return self._results

    @staticmethod
    def _successful_trials(
        result: ExperimentResult,
    ) -> list[TrialResult]:

        return [
            trial
            for trial in result.trials
            if trial.success
        ]

    @staticmethod
    def _metric_values(
        result: ExperimentResult,
        metric: Metric,
    ) -> list[float]:

        values: list[float] = []

        for trial in result.trials:
            if not trial.success:
                continue

            value = getattr(
                trial.evaluation_result,
                metric,
                None,
            )

            if value is not None:
                values.append(float(value))

        return values

    @classmethod
    def _mean_metric(
        cls,
        result: ExperimentResult,
        metric: Metric,
    ) -> float:

        values = cls._metric_values(
            result,
            metric,
        )

        return (
            float(np.mean(values))
            if values
            else 0.0
        )

    @staticmethod
    def _z_support_size(
        trial: TrialResult,
    ) -> int:
        return trial.evaluation_result.z_support_percentage 

    @staticmethod
    def _calculate_error(
        values: list[float],
        error: Literal["std", "sem"],
    ) -> float:

        if len(values) <= 1:
            return 0.0

        std = float(
            np.std(values, ddof=1)
        )

        if error == "std":
            return std

        return std / np.sqrt(len(values))

    @staticmethod
    def _metric_label(
        metric: Metric,
    ) -> str:

        return {
            "violation_percentage": "Violation Rate (%)",
            "po_violation_percentage": "PO Violation Rate (%)",
            "pmc_violation_percentage": "PMC Violation Rate (%)",
        }[metric]