from __future__ import annotations

# from dataclasses import dataclass
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns

from experiments.core.experiment_statistics import ExperimentResult
from experiments.core.trial_result import TrialResult
from ai_alignment_linear_model_extension.visualization.plot_data import PlotRecord, ParameterSweepResult
from ai_alignment_linear_model_extension.visualization.utils import setup_plot, style_axes
from ai_alignment_linear_model_extension.optimization.utills import ModelType
# from experiments.core.parameter_sweep_result import ParameterSweepResult


Metric = Literal[
    "violation_percentage",
    "po_violation_percentage",
    "pmc_violation_percentage",
]


class ViolationPlots:
    """Create and manage violation-related figures."""

    def __init__(
        self,
        results: ExperimentResult | ParameterSweepResult,
        ) -> None:
        setup_plot()
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

    # Plot management

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
            if ModelType.KT_MIN_LP in list(self._results.statistics.keys()):
                self.plot_z_support_across_trials()
            self.plot_epsilon_sparsity()

        return self.plots

    # Single experiment

    def plot_violation_rates(self) -> Figure:

        rows = []
        result = self._require_single()
        for model, stat in result.statistics.items():
            for trial in result.trials:
                if not trial.success:
                    continue

                evaluation = trial.evaluation_result[model]

                rows.extend([
                    {
                        "model": model.label,
                        "metric": "PO",
                        "value": evaluation.po_violation_percentage,
                    },
                    {
                        "model": model.label,
                        "metric": "PMC",
                        "value": evaluation.pmc_violation_percentage,
                    },
                    {
                        "model": model.label,
                        "metric": "Total",
                        "value": evaluation.violation_percentage,
                    },
                ])

        dataframe = pd.DataFrame(rows)

        figure, axis = plt.subplots(figsize=(7, 4.5))

        sns.barplot(
            data=dataframe,
            x="model",
            y="value",
            hue="metric",
            errorbar=("ci", 95),
            ax=axis,
        )

        axis.set_xlabel("Model")
        axis.set_ylabel("Violation Rate (%)")
        axis.set_ylim(0, 100)

        style_axes(axis)
        figure.tight_layout()

        return self._store("violation_rates", figure)

    def plot_violation_across_trials(self) -> Figure:
        """Total violation rate across trials for each model."""

        result = self._require_single()
        rows = []
        for model, stat in result.statistics.items():
            trials = self._successful_trials(result)

            for i, trial in enumerate(trials):          
                evaluation = trial.evaluation_result[model]           
                rows.extend([
                    {
                        "model": model.label,
                        "trial": i + 1,
                        "value": evaluation.violation_percentage,
                    }
                ])
        
        df = pd.DataFrame(rows)
        
        figure, axis = plt.subplots(figsize=(7, 4.5))
        
        sns.lineplot(
            data=df,
            x="trial",
            y="value",
            hue="model",
            estimator="mean",
            errorbar=None,
            ax=axis,
        )
        
        axis.set(
            xlabel = "Trial",
            ylabel = "Violation Rate (%)",
            title = "Violation Rate Across Trials",
            ylim = (0, 100),
        )
        axis.legend(
            title="Model",
            frameon=False,
        )

        style_axes(axis=axis)
        figure.tight_layout()

        return self._store(
            "violation_across_trials",
            figure,
        )

    def plot_po_vs_pmc(self) -> Figure:
        """Mean PO and PMC violation rate by model."""

        result = self._require_single()

        figure, axis = plt.subplots(figsize=(9, 5))

        x = np.arange(2)

        for model, stat in result.statistics.items():
            po = stat.mean_po_violation_percentage
            pmc = stat.mean_pmc_violation_percentage

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
        rows = []
        result = self._require_single()
        
        for model, stat in result.statistics.items():
            trials = self._successful_trials(result)
            
            for i, trial in enumerate(trials):          
                evaluation = trial.evaluation_result[model]           
                rows.extend([
                    {
                        "model": model.label,
                        "trial": i + 1,
                        "value": evaluation.epsilon_support_percentage,
                    }
                ])

        df = pd.DataFrame(rows)
        
        figure, axis = plt.subplots(figsize=(7, 4.5))
        
        sns.lineplot(
            data=df,
            x="trial",
            y="value",
            hue="model",
            estimator="mean",
            errorbar=None,
            ax=axis,
        )
        
        axis.set(
            xlabel = "Trial",
            ylabel = "ε Support Percentage (%)",
            title = "ε Support Across Trials",
            ylim = (0, 100)
        )
        
        axis.legend(
            title="Model",
            frameon=False,
        )
        style_axes(axis)
        figure.tight_layout()

        return self._store(
            "epsilon_support_across_trials",
            figure,
        )

    def plot_z_support_across_trials(self) -> Figure:
        """z support size across trials for the new LP."""

        rows = []
        result = self._require_single()
        
        for model, stat in result.statistics.items():
            trials = self._successful_trials(result)
            if model == ModelType.KT_MIN_LP:
                for i, trial in enumerate(trials):          
                    evaluation = trial.evaluation_result[model]           
                    rows.extend([
                        {
                            "model": model.label,
                            "trial": i + 1,
                            "value": evaluation.z_support_percentage,
                        }
                    ])
        
        df = pd.DataFrame(rows)
        
        figure, axis = plt.subplots(figsize=(7, 4.5))
        
        sns.lineplot(
            data=df,
            x="trial",
            y="value",
            hue="model",
            estimator="mean",
            errorbar=None,
            ax=axis,
        )
        
        axis.set(
            xlabel = "Trial",
            ylabel = "z Support Percentage (%)",
            title = "z Support Across Trials",
            ylim = (0, 100)
        )
        
        axis.legend(
            title="Model",
            frameon=False,
        )
        style_axes(axis)
        figure.tight_layout()

        return self._store(
            "z_support_across_trials",
            figure,
        )

    def plot_epsilon_sparsity(self) -> Figure:
        """Epsilon sparsity across trials for LP models."""

        rows = []
        result = self._require_single()
        
        for model, stat in result.statistics.items():
            trials = self._successful_trials(result)
        
            for i, trial in enumerate(trials):          
                evaluation = trial.evaluation_result[model]           
                rows.extend([
                    {
                        "model": model.label,
                        "trial": i + 1,
                        "value": evaluation.epsilon_sparsity,
                    }
                ])
        
        df = pd.DataFrame(rows)
        
        figure, axis = plt.subplots(figsize=(7, 4.5))
        
        sns.lineplot(
            data=df,
            x="trial",
            y="value",
            hue="model",
            estimator="mean",
            errorbar=None,
            ax=axis,
        )
        
        axis.set(
            xlabel = "Trial",
            ylabel = "ε Sparsity Percentage (%)",
            title = "ε Sparsity Across Trials",
            ylim = (0, 100)
        )
        
        axis.legend(
            title="Model",
            frameon=False,
        )
        style_axes(axis)
        figure.tight_layout()

        return self._store(
            "epsilon_sparsity_across_trials",
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

    
    def plot_mallows_summary(
        self,
        sweep: ParameterSweepResult,
        *,
        num_voters: int,
        num_alternatives: int,
        feature_dimension: int,
        confidence_level: float = 0.95,
    ) -> Figure:

        sns.set_theme(
            context="paper",
            style="whitegrid",
            font_scale=1.05,
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(7.2, 5.8),
            constrained_layout=True,
            gridspec_kw={
                "hspace": 0.10,
                "wspace": 0.10,
            }
        )

        specs = [
            (axes[0, 0], "total", "Total violations", "Violation rate (%)"),
            (axes[0, 1], "po", "Pareto Optimality", "Violation rate (%)"),
            (axes[1, 0], "pmc",
            "Pairwise Majority Consistency", "Violation rate (%)"),
        ]

        for ax, metric, title, ylabel in specs:

            df = self._sweep_dataframe(
                sweep,
                metric,
            )

            sns.lineplot(
                data=df,
                x="dispersion",
                y="value",
                hue="model",
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                errorbar=("ci", confidence_level * 100),
                ax=ax,
            )

            ax.set_title(title)
            ax.set_xlabel(r"Mallows dispersion $\phi$")
            ax.set_ylabel(ylabel)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        # epsilon panel
        epsilon_df = self._epsilon_dataframe(sweep)

        ax = axes[1, 1]

        sns.lineplot(
            data=epsilon_df,
            x="dispersion",
            y="value",
            hue="model",
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            errorbar=("ci", confidence_level * 100),
            ax=ax,
        )

        ax.set_title(r"Candidate-level repair")
        ax.set_xlabel(r"Mallows dispersion $\phi$")
        ax.set_ylabel(r"$\|\epsilon\|_1$")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Shared legend
        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
        )

        return fig
    
    # =========================================================
    # Internal helpers
    # =========================================================
    def _epsilon_dataframe(
        self,
        sweep: ParameterSweepResult,
    ) -> pd.DataFrame:

        rows = []

        for phi, experiment_result in sweep.results.items():

            for trial in experiment_result.trials:

                if not trial.success:
                    continue

                if trial.evaluation_result is None:
                    continue

                for model_type in (
                    ModelType.EPSILON_LP,
                    ModelType.KT_MIN_LP,
                    ModelType.WEIGHTED_KT_MIN_LP,
                    ModelType.BORDA_KT_MIN_LP,
                ):
                    if model_type not in trial.evaluation_result:
                        continue
                    result = trial.evaluation_result.get(model_type)

                    if result is None or result.epsilon_l1_norm is None:
                        continue

                    epsilon_l1 = result.epsilon_l1_norm

                    rows.append(
                        {
                            "dispersion": phi,
                            "trial": trial.trial,
                            "model": model_type.label,
                            "value": epsilon_l1,
                        }
                    )

        return pd.DataFrame(rows)
    
    
    def _sweep_dataframe(
        self,
        sweep: ParameterSweepResult,
        metric: str,
    ) -> pd.DataFrame:

        rows = []

        for phi, experiment_result in sweep.results.items():

            for trial in experiment_result.trials:

                if not trial.success:
                    continue

                if trial.evaluation_result is None:
                    continue

                for model_type, evaluation in (
                    trial.evaluation_result.items()
                ):

                    if metric == "total":
                        value = evaluation.violation_percentage

                    elif metric == "po":
                        value = evaluation.po_violation_percentage

                    elif metric == "pmc":
                        value = evaluation.pmc_violation_percentage

                    else:
                        raise ValueError(
                            f"Unknown metric: {metric}"
                        )

                    rows.append(
                        {
                            "dispersion": phi,
                            "trial": trial.trial,
                            "model": model_type.label,
                            "value": value,
                        }
                    )

        return pd.DataFrame(rows)


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
    ) -> ExperimentResult:

        if not isinstance(
            self._results,
            ExperimentResult,
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