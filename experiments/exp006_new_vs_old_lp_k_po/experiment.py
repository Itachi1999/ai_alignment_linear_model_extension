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
from ai_alignment_linear_model_extension.generators.preference_generator import KLengthPOPreferenceGenerator
from ai_alignment_linear_model_extension.generators.election_generator import  ElectionGenerator
from ai_alignment_linear_model_extension.preference_graph.preference_graph import PreferenceGraphBuilder
from ai_alignment_linear_model_extension.preference_graph.FAS import FeedbackArcSetSolver
from ai_alignment_linear_model_extension.optimization.extension_linear_model import LinearModelSolver
from ai_alignment_linear_model_extension.optimization.linear_KT_min_model import KTMinLinearModelSolver
from ai_alignment_linear_model_extension.evaluation.linear_model_evaluator import LinearModelEvaluator
from ai_alignment_linear_model_extension.visualization.violation_plots import (
    ViolationPlots, ModelType
)

class OldVsNewLPExperiment(BaseExperiment):
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
            preference_generator=KLengthPOPreferenceGenerator(
                seed=self._cfg.seed,
                theta=np.random.normal(loc=self._cfg.data.mean, scale=self._cfg.data.std, size=self._cfg.data.dimension),
                k=self._cfg.data.k
            ),
        )

        self._graph_builder = PreferenceGraphBuilder()

        self._fas_solver = FeedbackArcSetSolver()

        # Optimization parameter is unused in the LinearModelSolver, but we keep it for consistency with other experiments.
        self._old_lp_solver = LinearModelSolver(
            eta=self._cfg.optimization.eta,
            solver=self._cfg.optimization.solver,
        )   
        self._new_lp_solver = KTMinLinearModelSolver(
            eta = self._cfg.optimization.eta,
            lambda_= self._cfg.optimization.lambda_,
            solver=self._cfg.optimization.solver,
        )

        self._graph_builder = PreferenceGraphBuilder()

        self._fas_solver = FeedbackArcSetSolver()

        self._evaluator = LinearModelEvaluator()

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
            _, graph = self._graph_builder.build(election)
        timers.append(timer.result)

        with Timer("Feedback Arc Set") as timer:
            fas_result = self._fas_solver.solve(graph)
        timers.append(timer.result)

        with Timer("Old LP Solve") as timer:
            old_lp_result = self._old_lp_solver.solve(
                election,
                fas_result.dag,
            )
        timers.append(timer.result)
        
        with Timer("New LP Solve") as timer:
            new_lp_result = self._new_lp_solver.solve(
                election,
                fas_result.dag,
            )
        timers.append(timer.result)

        with Timer("Old LP Evaluation") as timer:
            old_evaluation = self._evaluator.evaluate(
                election,
                fas_result.dag,
                old_lp_result,
            )
        timers.append(timer.result)
        
        with Timer("New LP Evaluation") as timer:
            new_evaluation = self._evaluator.evaluate(
                election,
                fas_result.dag,
                new_lp_result,
            )
        timers.append(timer.result)

        return TrialResult(
            trial=trial,
            seed=seed,
            # lp_result=lp_result,
            evaluation_result={
                ModelType.EPSILON_LP: old_evaluation,
                ModelType.KT_MIN_LP: new_evaluation
            },
            fas_size=len(fas_result.removed_edges),
            timers=tuple(timers),
        )

    def summarize(
        self,
        trials: list[TrialResult],
    ) -> ExperimentResult:
        statistics:dict[ModelType, ExperimentStatistics] = {}
        for model in [ModelType.EPSILON_LP, ModelType.KT_MIN_LP]:
            objectives = [
                trial.evaluation_result[model].objective_value
                for trial in trials
            ]

            violations = [
                trial.evaluation_result[model].violation_percentage
                for trial in trials
            ]

            po_violations = [
                trial.evaluation_result[model].po_violation_percentage
                for trial in trials
            ]

            pmc_violations = [
                trial.evaluation_result[model].pmc_violation_percentage
                for trial in trials
            ]

            fas_sizes = [
                trial.fas_size
                for trial in trials
            ]

            epsilon_support_percentages = [
                trial.evaluation_result[model].epsilon_support_percentage
                for trial in trials
            ]

            epsilon_sparsities = [
                trial.evaluation_result[model].epsilon_sparsity
                for trial in trials
            ]
            
            z_support_percentages = [
                trial.evaluation_result[model].z_support_percentage
                for trial in trials    
            ]
            
            successful_trials = [
                trial for trial in trials
                if trial.success
            ]

            statistics[model] =  ExperimentStatistics(
                num_trials=len(trials),

                mean_objective=fmean(objectives),
                std_objective=stdev(objectives),

                mean_violation_percentage=mean(violations),
                std_violation_percentage=stdev(violations),

                mean_po_violation_percentage=mean(po_violations),
                std_po_violation_percentage=stdev(po_violations),

                mean_pmc_violation_percentage=mean(pmc_violations),
                std_pmc_violation_percentage=stdev(pmc_violations),

                mean_fas_size=mean(fas_sizes),
                std_fas_size=stdev(fas_sizes),
                
                success_rate = (
                    len(successful_trials) / len(trials)
                    if trials else 0.0
                ),

                mean_epsilon_support_percentage=mean(epsilon_support_percentages),
                std_epsilon_support_percentage=stdev(epsilon_support_percentages),

                mean_epsilon_sparsity=mean(epsilon_sparsities),
                std_epsilon_sparsity=stdev(epsilon_sparsities),
                mean_z_support_percentage=mean(z_support_percentages), 
                std_z_support_percentage=stdev(z_support_percentages),    
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
    experiment = OldVsNewLPExperiment(exp_cfg)

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
    
    plots = ViolationPlots(result)
    plots.create_all()
    
    logger.save_plots(
        plots.plots,
        output_dir,
    )
    plots.clear()

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