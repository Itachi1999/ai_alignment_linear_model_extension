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

        logging.info(
            "Starting experiment: %d trials, seed=%d",
            num_trials,
            seed,
        )
        experiment.setup()

        trials: list[TrialResult] = []

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
                evaluation_result=None,
                fas_size=None,
                timers=(),
            )
            trials.append(trial_result)
        logging.info("Trials completed successfully.")  
        return experiment.summarize(trials)



class ExperimentLogger:

    def save(
        self,
        result: ExperimentResult,
        config: DictConfig,
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_config(config, output_dir)
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

        with (output_dir / "summary.json").open("w") as file:
            json.dump(
                asdict(result.statistics),
                file,
                indent=4,
            )

    def _save_trials(
        self,
        result: ExperimentResult,
        output_dir: Path,
    ) -> None:

        rows = []

        for trial in result.trials:

            evaluation = trial.evaluation_result

            rows.append({
                "trial": trial.trial,
                "seed": trial.seed,

                "objective": trial.evaluation_result.objective_value,
                "loss_value": trial.evaluation_result.loss_value,
                
                "fas_size": trial.fas_size,

                "num_violations": evaluation.num_violations,
                "num_po_violations": evaluation.num_po_violations,
                "num_pmc_violations": evaluation.num_pmc_violations,

                "epsilon_support_percentage":
                    evaluation.epsilon_support_percentage,

                "epsilon_sparsity":
                    evaluation.epsilon_sparsity,

                **{
                    f"time_{timer.name}":
                        timer.elapsed_time
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


# class ExperimentLogger:

#     def save_summary(
#         self,
#         result: ExperimentResult,
#         output_dir: Path,
#     ) -> None:

#         output_dir.mkdir(parents=True, exist_ok=True)

#         summary = asdict(result.statistics)

#         with (output_dir / "summary.json").open("w") as file:
#             json.dump(summary, file, indent=4)

#     def save_trials(
#         self,
#         result: ExperimentResult,
#         output_dir: Path,
#     ) -> None:

#         output_dir.mkdir(parents=True, exist_ok=True)

#         with (output_dir / "trials.csv").open(
#             "w",
#             newline="",
#         ) as file:

#             writer = csv.writer(file)

#             writer.writerow(
#                 [
#                     "trial",
#                     "seed",
#                     "objective",
#                     "fas_size",
#                     "violations",
#                     "po_violations",
#                     "pmc_violations",
#                     "epsilon_support_percentage",
#                     "epsilon_sparsity",
#                 ]
#             )

#             for trial in result.trials:

#                 writer.writerow(
#                     [
#                         trial.trial,
#                         trial.seed,
#                         trial.lp_result.objective_value,
#                         trial.fas_size,
#                         trial.evaluation_result.num_violations,
#                         trial.evaluation_result.num_po_violations,
#                         trial.evaluation_result.num_pmc_violations,
#                         trial.evaluation_result.epsilon_support_percentage,
#                         trial.evaluation_result.epsilon_sparsity,
#                     ]
#                 )

#     def save_config(
#         self,
#         cfg: DictConfig,
#         output_dir: Path,
#     ) -> None:

#         output_dir.mkdir(parents=True, exist_ok=True)

#         OmegaConf.save(
#             config=cfg,
#             f=output_dir / "config.yaml",
#         )

#     def save_log(
#         self,
#         message: str,
#         output_dir: Path,
#     ) -> None:

#         output_dir.mkdir(parents=True, exist_ok=True)

#         with (output_dir / "log.txt").open("a") as file:
#             file.write(message + "\n")

#     def save_table(
#         self,
#         table: pd.DataFrame,
#         filename: str,
#         output_dir: Path,
#     ) -> None:

#         output_dir.mkdir(parents=True, exist_ok=True)

#         table.to_csv(
#             output_dir / filename,
#             index=False,
#         )


#     def save_figure(
#         self,
#         figure: plt.Figure,
#         filename: str,
#         output_dir: Path,
#     ) -> None:

#         figures_dir = output_dir / "figures"
#         figures_dir.mkdir(parents=True, exist_ok=True)

#         figure.savefig(
#             figures_dir / filename,
#             dpi=300,
#             bbox_inches="tight",
#         )

#         plt.close(figure)