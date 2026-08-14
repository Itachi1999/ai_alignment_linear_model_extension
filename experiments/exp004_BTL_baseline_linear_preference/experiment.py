from __future__ import annotations

import logging
import logging
from pathlib import Path
from statistics import mean, stdev, fmean
import hydra
from omegaconf import DictConfig
import numpy as np
from omegaconf import OmegaConf

# Experiment imports
from experiments.core.base_experiment import (
    BaseExperiment, ExperimentRunner, ExperimentLogger
)
from experiments.core.experiment_statistics import (
    ExperimentResult, TrialResult, ExperimentStatistics
)
from experiments.core.timer import Timer, TimerResult

# Main Module Imports
from ai_alignment_linear_model_extension.generators.feature_generator import GaussianFeatureGenerator
from ai_alignment_linear_model_extension.generators.preference_generator import LinearPreferenceGenerator
from ai_alignment_linear_model_extension.generators.election_generator import  ElectionGenerator
from ai_alignment_linear_model_extension.preference_graph.preference_graph import PreferenceGraphBuilder
from ai_alignment_linear_model_extension.preference_graph.FAS import FeedbackArcSetSolver
from ai_alignment_linear_model_extension.models.btl import BTLModel, BTLResult
from ai_alignment_linear_model_extension.evaluation.BTL_model_evaluator import BTLModelEvaluator

class BTLBaselineEvaluation(BaseExperiment):
    """ 
    This class implements a synthetic experiment to test the expressibility of LP generated theta against the LP generated theta and epsilon.
    """
    def __init__(self, cfg):
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def setup(self) -> None:
        self._data_generator = ElectionGenerator(
            feature_generator=GaussianFeatureGenerator(
                mean=self._cfg.data.mean,
                std_dev=self._cfg.data.std,
                seed=self._cfg.seed,
            ),
            preference_generator=LinearPreferenceGenerator(
                seed=self._cfg.seed,
                theta_0=np.random.normal(loc=self._cfg.data.mean, scale=self._cfg.data.std, size=self._cfg.data.dimension),
                noise_std=self._cfg.data.noise_std,
                noisy_alternative_fraction=self._cfg.data.noisy_alternative_fraction,
            )
        )

        self._graph_builder = PreferenceGraphBuilder()

        self._fas_solver = FeedbackArcSetSolver()

        self._graph_builder = PreferenceGraphBuilder()

        self._fas_solver = FeedbackArcSetSolver()

        self._evaluator = BTLModelEvaluator()

    def run_trial(
        self,
        trial: int,
        seed: int,
    ) -> TrialResult:
        rng = np.random.default_rng(seed)

        timers: list[TimerResult] = []

        with Timer("Data Generation") as timer:
            election = self._data_generator.generate(
                num_alternatives=self._cfg.data.num_alternatives,
                num_voters=self._cfg.data.num_voters,
                dimension=self._cfg.data.dimension
            )
        timers.append(timer.result)

        with Timer("Graph Construction") as timer:
            marginal_matrix, graph = self._graph_builder.build(election)
        timers.append(timer.result)

        with Timer("Feedback Arc Set") as timer:
            fas_result = self._fas_solver.solve(graph)
        timers.append(timer.result)

        with Timer("BTL Model Fitting") as timer:
            blt_model = BTLModel(
                marginal_matrix=marginal_matrix,
                scores = {alt.id: 0.0 for alt in election.alternatives},
                num_voters=self._cfg.data.num_voters,
            )
            btl_result = blt_model.fit(max_iter=self._cfg.optimization.max_iterations, tol=self._cfg.optimization.tolerance)
        timers.append(timer.result)

        with Timer("Evaluation") as timer:
            evaluation = self._evaluator.evaluate(
                fas_result.dag,
                btl_result,
            )
        timers.append(timer.result)

        return TrialResult(
            trial=trial,
            seed=seed,
            evaluation_result=evaluation,
            fas_size=len(fas_result.removed_edges),
            timers=tuple(timers),
        )

    def summarize(
        self,
        trials: list[TrialResult],
    ) -> ExperimentResult:

        # For this experiment, we will summarize the loss values, violation percentages, and other relevant metrics across all trials.
        losses = [
            trial.evaluation_result.loss_value
            for trial in trials
        ]

        violations = [
            trial.evaluation_result.violation_percentage
            for trial in trials
        ]

        po_violations = [
            trial.evaluation_result.po_violation_percentage
            for trial in trials
        ]

        pmc_violations = [
            trial.evaluation_result.pmc_violation_percentage
            for trial in trials
        ]

        fas_sizes = [
            trial.fas_size
            for trial in trials
        ]

        # epsilon_support_percentages = [
        #     trial.evaluation_result.epsilon_support_percentage
        #     for trial in trials
        # ]

        # epsilon_sparsities = [
        #     trial.evaluation_result.epsilon_sparsity
        #     for trial in trials
        # ]
        successful_trials = [
            trial for trial in trials
            if trial.success
        ]

        statistics = ExperimentStatistics(
            num_trials=len(trials),

            mean_loss=fmean(losses),
            std_loss=stdev(losses),

            mean_violation_percentage=mean(violations),
            std_violation_percentage=stdev(violations),

            mean_po_violation_percentage=mean(po_violations),
            std_po_violation_percentage=stdev(po_violations),

            mean_pmc_violation_percentage=mean(pmc_violations),
            std_pmc_violation_percentage=stdev(pmc_violations),

            mean_fas_size=mean(fas_sizes),
            std_fas_size=stdev(fas_sizes),

            # mean_epsilon_support_percentage=mean(epsilon_support_percentages),
            # std_epsilon_support_percentage=stdev(epsilon_support_percentages),

            # mean_epsilon_sparsity=mean(epsilon_sparsities),
            # std_epsilon_sparsity=stdev(epsilon_sparsities),

            # success_rate = (
            #     sum(
            #         trial.evaluation_result.num_violations == 0
            #         for trial in trials
            #     )
            #     / len(trials)
            # ),
            success_rate = (
                len(successful_trials) / len(trials)
                if trials else 0.0
            ),
        )

        return ExperimentResult(
            trials=tuple(trials),
            statistics=statistics,
        )

@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    exp_cfg = cfg.experiment
    print(f"Running experiment: {exp_cfg.name}")
    # print(f"Configuration:\n{cfg.pretty()}")
    experiment = BTLBaselineEvaluation(exp_cfg)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    logger = ExperimentLogger()
    logger.setup_logging(output_dir)
    logging.info("Starting %s", exp_cfg.name)

    runner = ExperimentRunner()
    result = runner.run(
        experiment=experiment,
        num_trials=exp_cfg.num_trials,
        seed=exp_cfg.seed,
    )

    logger.save(
        result=result,
        config=cfg,
        output_dir=output_dir,
    )

    logging.info("Experiment completed.")


    # logger.save_summary(result, output_dir)
    # logger.save_trials(result, output_dir)
    # logger.save_config(exp_cfg, output_dir)


if __name__ == "__main__":
    main()
