from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm.rich import trange
from dataclasses import asdict
import csv, json
from matplotlib.figure import Figure

from experiments.core.experiment_statistics import ExperimentResult
from experiments.core.trial_result import TrialResult
from ai_alignment_linear_model_extension.visualization.plot_data import ParameterSweepResult, PlotRecord


class BaseExperiment(ABC):

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def run_trial(
        self,
        trial: int,
        seed: int,
    ) -> TrialResult:
        ...

    @abstractmethod
    def summarize(
        self,
        trials: list[TrialResult],
    ) -> ExperimentResult:
        ...


class ExperimentRunner:

    def run(
        self,
        experiment: BaseExperiment,
        num_trials: int,
        seed: int,
    ) -> ExperimentResult:

        if num_trials >= 2:
            logging.info(
                "Starting experiment: %d trials, seed=%d",
                num_trials,
                seed,
            )
        experiment.setup()

        trials: list[TrialResult] = []
        if num_trials <= 1:
            try:
                trial_result = experiment.run_trial(
                    trial=0,
                    seed=seed,
                )
            except Exception as e:
                logging.exception(
                    "Error occurred while running trial %d: %s", trial, e
                )
                trial_result = TrialResult(
                trial=0,
                seed=seed,
                success=False,
                error=str(e),
                evaluation_result={},
                fas_size=0,
                timers=(),
            )
            trials.append(trial_result)

        else:
            for trial in trange(num_trials):
                seed = seed + trial
                logging.debug(
                    "Starting trial %d with seed %d",
                    trial,
                    seed,
                )
                try:
                    trial_result = experiment.run_trial(
                        trial=trial,
                        seed=seed,
                    )
                except Exception as e:
                    logging.exception(
                        "Error occurred while running trial %d: %s", trial, e
                    )
                    trial_result = TrialResult(
                    trial=trial,
                    seed=seed,
                    success=False,
                    error=str(e),
                    evaluation_result={},
                    fas_size=0,
                    timers=(),
                )
                trials.append(trial_result)
            logging.info("Trials completed successfully.")  
        return experiment.summarize(trials)



class ExperimentLogger:

    def save(
        self,
        result: ExperimentResult | ParameterSweepResult,
        config: DictConfig,
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config(config, output_dir)

        if isinstance(result, ParameterSweepResult):
            self._save_sweep_summary(result, output_dir)
            self._save_sweep_trials(result, output_dir)
        else:
            self._save_summary(result, output_dir)
            self._save_trials(result, output_dir)

    def _save_config(
        self,
        config: DictConfig,
        output_dir: Path,
    ) -> None:

        OmegaConf.save(
            config,
            output_dir / "config.yaml",
        )

    def _save_summary(
        self,
        result: ExperimentResult,
        output_dir: Path,
    ) -> None:
        json_result = {}
        json_result.update({
            model.label: asdict(stat)
            for model, stat in result.statistics.items()
        })
        with (output_dir / "summary.json").open("w") as file:
            json.dump(
                json_result,
                file,
                indent=4,
            )

    def _save_trials(
        self,
        result: ExperimentResult,
        output_dir: Path,
    ) -> None:

        rows = []
        for model, _ in result.statistics.items():
            for trial in result.trials:
                evaluation = trial.evaluation_result[model]
                rows.append({
                    "model": model.label,
                    "trial": trial.trial,
                    "seed": trial.seed,

                    "objective": evaluation.objective_value,
                    "loss_value": evaluation.loss_value,
                    "fas_size": trial.fas_size,
                    
                    "violation_percentage": evaluation.violation_percentage,
                    "po_violation_percentage": evaluation.po_violation_percentage,
                    "pmc_violation_percentage": evaluation.pmc_violation_percentage,

                    "num_violations": evaluation.num_violations,
                    "num_po_violations": evaluation.num_po_violations,
                    "num_pmc_violations": evaluation.num_pmc_violations,

                    "epsilon_l1_norm": evaluation.epsilon_l1_norm,
                    "epsilon_support_percentage": evaluation.epsilon_support_percentage,
                    "epsilon_sparsity": evaluation.epsilon_sparsity,

                    **{
                        f"time_{timer.name}": timer.elapsed_time
                        for timer in trial.timers
                    },
                })

        pd.DataFrame(rows).to_csv(
            output_dir / "trials.csv",
            index=False,
        )
        
    def _save_sweep_summary(
        self,
        result: ParameterSweepResult,
        output_dir: Path,
    ) -> None:
        json_result = {}

        for phi, experiment_result in result.results.items():
            json_result[str(phi)] = {
                model.label: asdict(stat)
                for model, stat in experiment_result.statistics.items()
            }

        with (output_dir / "summary.json").open("w") as file:
            json.dump(
                json_result,
                file,
                indent=4,
            )
            
            
    def _save_sweep_trials(
        self,
        result: ParameterSweepResult,
        output_dir: Path,
    ) -> None:
        rows = []

        for phi, experiment_result in result.results.items():
            for model, _ in experiment_result.statistics.items():
                for trial in experiment_result.trials:
                    evaluation = trial.evaluation_result[model]

                    rows.append({
                        "phi": phi,
                        "model": model.label,
                        "trial": trial.trial,
                        "seed": trial.seed,
                        "objective": evaluation.objective_value,
                        "loss_value": evaluation.loss_value,
                        "fas_size": trial.fas_size,
                        "violation_percentage": evaluation.violation_percentage,
                        "po_violation_percentage": evaluation.po_violation_percentage,
                        "pmc_violation_percentage": evaluation.pmc_violation_percentage,
                        "num_violations": evaluation.num_violations,
                        "num_po_violations": evaluation.num_po_violations,
                        "num_pmc_violations": evaluation.num_pmc_violations,
                        "epsilon_l1_norm": evaluation.epsilon_l1_norm,
                        "epsilon_support_percentage": (
                            evaluation.epsilon_support_percentage
                        ),
                        "epsilon_sparsity": evaluation.epsilon_sparsity,
                        **{
                            f"time_{timer.name}": timer.elapsed_time
                            for timer in trial.timers
                        },
                    })

        pd.DataFrame(rows).to_csv(
            output_dir / "trials.csv",
            index=False,
        )

    def setup_logging(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger()

        logger.setLevel(logging.DEBUG)

        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            output_dir / "experiment.log",
            mode="w",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    def save_figure(
        self,
        figure: Figure,
        filename: str,
        output_dir: Path,
        dpi: int | None = 300,
    ) -> None:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        figure.savefig(
            figures_dir / filename,
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(figure)
        
    def save_plots(
        self,
        plots: tuple[PlotRecord, ...],
        output_dir: Path,
    ) -> None:
        for plot in plots:
            self.save_figure(
                figure=plot.figure,
                filename=f"{plot.name}.png",
                output_dir=output_dir,
            )
