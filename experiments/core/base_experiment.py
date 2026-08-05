from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm.rich import trange
from dataclasses import asdict
import csv, json

from experiments.core.experiment_statistics import ExperimentResult
from experiments.core.trial_result import TrialResult


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

        experiment.setup()

        trials: list[TrialResult] = []

        for trial in trange(num_trials):

            trials.append(
                experiment.run_trial(
                    trial=trial,
                    seed=seed + trial,
                )
            )

        return experiment.summarize(trials)


class ExperimentLogger:

    def save_summary(
        self,
        result: ExperimentResult,
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)

        summary = asdict(result.statistics)

        with (output_dir / "summary.json").open("w") as file:
            json.dump(summary, file, indent=4)

    def save_trials(
        self,
        result: ExperimentResult,
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "trials.csv").open(
            "w",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "trial",
                    "seed",
                    "objective",
                    "fas_size",
                    "violations",
                    "po_violations",
                    "pmc_violations",
                ]
            )

            for trial in result.trials:

                writer.writerow(
                    [
                        trial.trial,
                        trial.seed,
                        trial.lp_result.objective_value,
                        trial.fas_size,
                        trial.evaluation_result.num_violations,
                        trial.evaluation_result.num_po_violations,
                        trial.evaluation_result.num_pmc_violations,
                    ]
                )

    def save_config(
        self,
        cfg: DictConfig,
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(
            config=cfg,
            f=output_dir / "config.yaml",
        )

    def save_log(
        self,
        message: str,
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "log.txt").open("a") as file:
            file.write(message + "\n")

    def save_table(
        self,
        table: pd.DataFrame,
        filename: str,
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)

        table.to_csv(
            output_dir / filename,
            index=False,
        )


    def save_figure(
        self,
        figure: plt.Figure,
        filename: str,
        output_dir: Path,
    ) -> None:

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        figure.savefig(
            figures_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)