from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean, stdev, fmean
from ai_alignment_linear_model_extension.visualization.plot_data import ParameterSweepResult
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
from ai_alignment_linear_model_extension.preference_graph.preference_graph import PreferenceGraphBuilder, PreferenceEdgeType
from ai_alignment_linear_model_extension.preference_graph.FAS import FeedbackArcSetSolver
from ai_alignment_linear_model_extension.optimization.extension_linear_model import LinearModelSolver
from ai_alignment_linear_model_extension.optimization.linear_KT_min_model import KTMinLinearModelSolver
from ai_alignment_linear_model_extension.optimization.btl_linear import BTLLinearModel
from ai_alignment_linear_model_extension.optimization.hinge_linear import HingeLinearModel
from ai_alignment_linear_model_extension.evaluation.linear_model_evaluator import LinearModelEvaluator
from ai_alignment_linear_model_extension.evaluation.BTL_model_evaluator import BTLModelEvaluator
from ai_alignment_linear_model_extension.visualization.violation_plots import (
    ViolationPlots, ModelType, ParameterNameMapping
)

class AllModelsComparisonMallows(BaseExperiment):
    """ 
    This class implements a synthetic experiment to test the expressibility of LP generated theta against the LP generated theta and epsilon.
    """
    def __init__(self, cfg, dispersion: float):
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._dispersion = dispersion
        self.voter_thetas: list[np.ndarray] = []

    def setup(self) -> None:
        self._theta_0 = np.random.normal(loc=self._cfg.data.mean, scale=self._cfg.data.std, size=self._cfg.data.dimension)
        feature_generator = GaussianFeatureGenerator(
            mean=self._cfg.data.mean,
            std_dev=self._cfg.data.std,
            seed=self._cfg.seed,
        )
        preference_generator = LinearPreferenceGenerator(theta_0= self._theta_0)
        self.voter_thetas = preference_generator.generate_thetas(num_voters = self._cfg.data.training_num_voters, noise_std = self._dispersion, seed = self._cfg.seed)
        self._data_generator = ElectionGenerator(feature_generator=feature_generator, preference_generator=preference_generator)
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
        self._evaluator = LinearModelEvaluator()
        self._btl_evaluator = BTLModelEvaluator(eta = self._cfg.optimization.eta)

    def run_trial(
        self,
        trial: int,
        seed: int,
    ) -> tuple[TrialResult, TrialResult]:
        rng = np.random.default_rng(seed)

        timers: list[TimerResult] = []

        with Timer("Training Data Generation") as timer:
            training_election = self._data_generator.generate(
                num_alternatives=self._cfg.data.training_num_alternatives,
                num_voters=self._cfg.data.training_num_voters,
                dimension=self._cfg.data.dimension
            )
        timers.append(timer.result)
        logging.debug(f"training election preference profile: {training_election.rankings}")

        with Timer("Training Graph Construction") as timer:
            training_marginal_matrix, training_graph = self._graph_builder.build(training_election)
        timers.append(timer.result)
        logging.debug(f"Training Marginal Matrix: {training_marginal_matrix}")
        logging.debug(f"Training preference graph: {training_graph}")
        with Timer("Feedback Arc Set") as timer:
            training_fas_result = self._fas_solver.solve(training_graph)
        timers.append(timer.result)
        logging.debug(f"FAS Ordering {training_fas_result.ordering}")
        logging.debug(f"Removed Edges: {training_fas_result.removed_edges}")

        # with Timer("LP2 Solve") as timer:
        #     old_lp_result = self._old_lp_solver.solve(
        #         election=training_election,
        #         graph=training_fas_result.dag,
        #     )
        # timers.append(timer.result)
        # logging.info(f"LP2 Solve time: {timer.result.elapsed_time:.6f} seconds")
        
        with Timer("LP3 Solve") as timer:
            new_lp_result = self._new_lp_solver.solve(
                election=training_election,
                graph=training_fas_result.dag,
            ) 
        timers.append(timer.result)
        logging.info(f"LP3 Solve time: {timer.result.elapsed_time:.6f} seconds")
        
        with Timer("BTL Model Fitting") as timer:
            blt_model = BTLLinearModel(
                election=training_election,
                marginal_matrix=training_marginal_matrix,
            )
            btl_linear_result = blt_model.fit(max_iter=self._cfg.optimization.max_iterations)
        timers.append(timer.result)
        logging.info(f"BTL Model Fitting time: {timer.result.elapsed_time:.6f} seconds, Converged: {btl_linear_result.converged}, Iterations: {btl_linear_result.iterations}, loss: {btl_linear_result.log_likelihood:.6f}")
        
        # with Timer("Hinge Model Fitting") as timer:
        #     hinge_model = HingeLinearModel(
        #         election=training_election,
        #         marginal_matrix=training_marginal_matrix,
        #         margin=self._cfg.optimization.eta,
        #     )
        #     hinge_result = hinge_model.fit(max_iter=self._cfg.optimization.max_iterations)
        # timers.append(timer.result)
        # logging.info(f"Hinge Model Fitting time: {timer.result.elapsed_time:.6f} seconds, Converged: {hinge_result.converged}, Iterations: {hinge_result.iterations}, loss: {hinge_result.log_likelihood:.6f}")        


        # with Timer("Training Old LP Evaluation") as timer:
        #     training_old_evaluation = self._evaluator.evaluate(
        #         election=training_election,
        #         graph=training_fas_result.dag,
        #         result=old_lp_result,
        #     )
        # timers.append(timer.result)
        
        with Timer("Training New LP Evaluation") as timer:
            training_new_evaluation = self._evaluator.evaluate(
                election=training_election,
                graph=training_fas_result.dag,
                result=new_lp_result,
            )
        timers.append(timer.result)

        with Timer("Training BTL Evaluation") as timer:
            training_btl_evaluation = self._btl_evaluator.evaluate(
                graph=training_fas_result.dag,
                result=btl_linear_result,
            )
        timers.append(timer.result)
        
        # with Timer("Training Hinge Evaluation") as timer:
        #     training_hinge_evaluation = self._btl_evaluator.evaluate(
        #         graph=training_fas_result.dag,
        #         result=hinge_result,
        #     )
        # timers.append(timer.result)
        

        training_trial_result = TrialResult(
            trial=trial,
            seed=seed,
            # lp_result=lp_result,
            evaluation_result={
                # ModelType.EPSILON_LP: training_old_evaluation,
                ModelType.KT_MIN_LP: training_new_evaluation,
                ModelType.BTL_LINEAR: training_btl_evaluation,
                # ModelType.LINEAR_HINGE: training_hinge_evaluation,
            },
            fas_size=len(training_fas_result.removed_edges),
            timers=tuple(timers),
        ) 

        with Timer("Testing Data Generation") as timer:
            testing_election = self._data_generator.generate(
                num_alternatives=self._cfg.data.testing_num_alternatives,
                num_voters=self._cfg.data.testing_num_voters,
                dimension=self._cfg.data.dimension
            )
        timers.append(timer.result)
        logging.debug(f"testing election preference profile: {testing_election.rankings}")
        
        with Timer("Testing Graph Construction") as timer:
            testing_marginal_matrix, testing_graph = self._graph_builder.build(testing_election)
        timers.append(timer.result)
        logging.debug(f"Testing Marginal Matrix: {testing_marginal_matrix}")
        logging.debug(f"Testing preference graph: {testing_graph}")
        with Timer("Feedback Arc Set") as timer:
            testing_fas_result = self._fas_solver.solve(testing_graph)
        timers.append(timer.result)
        logging.debug(f"FAS Ordering {testing_fas_result.ordering}")
        logging.debug(f"Removed Edges: {testing_fas_result.removed_edges}")
        
        # with Timer("Testing Old LP Evaluation") as timer:
        #     testing_old_evaluation = self._evaluator.evaluate(
        #         election=testing_election,
        #         graph=testing_fas_result.dag,
        #         result=old_lp_result,
        #         testing=True,
        #     )
        # timers.append(timer.result)
        
        with Timer("Testing New LP Evaluation") as timer:
            testing_new_evaluation = self._evaluator.evaluate(
                election=testing_election,
                graph=testing_fas_result.dag,
                result=new_lp_result,
                testing=True,
            )
        timers.append(timer.result)

        with Timer("Testing BTL Evaluation") as timer:
            testing_btl_evaluation = self._btl_evaluator.evaluate(
                graph=testing_fas_result.dag,
                result=btl_linear_result,
                testing=True,
                alternatives=testing_election.alternatives
            )
        timers.append(timer.result)
        
        # with Timer("Testing Hinge Evaluation") as timer:
        #     testing_hinge_evaluation = self._btl_evaluator.evaluate(
        #         graph=testing_fas_result.dag,
        #         result=hinge_result,
        #         testing=True,
        #         alternatives=testing_election.alternatives
        #     )
        # timers.append(timer.result)
        
        testing_trial_result = TrialResult(
            trial=trial,
            seed=seed,
            # lp_result=lp_result,
            evaluation_result={
                # ModelType.EPSILON_LP: testing_old_evaluation,
                ModelType.KT_MIN_LP: testing_new_evaluation,
                ModelType.BTL_LINEAR: testing_btl_evaluation,
                # ModelType.LINEAR_HINGE: testing_hinge_evaluation,
            },
            fas_size=len(testing_fas_result.removed_edges),
            timers=tuple(timers),
        )
        
        return training_trial_result, testing_trial_result

    def summarize(
            self,
            trials: list[TrialResult] | tuple[list[TrialResult], list[TrialResult]],
        ) -> ExperimentResult | tuple[ExperimentResult, ExperimentResult]:
        training_trials: list[TrialResult] = []
        testing_trials: list[TrialResult] = []
        
        for train_trial, test_trial in trials:
            training_trials.append(train_trial)
            testing_trials.append(test_trial)
        training_summary = self.extract_summary(training_trials)
        testing_summary = self.extract_summary(testing_trials)
        
        return training_summary, testing_summary
    
    def extract_summary(self, trials: list[TrialResult]) -> ExperimentResult:
        statistics:dict[ModelType, ExperimentStatistics] = {}
        if len(trials) <= 1:
            for model in [ModelType.KT_MIN_LP, ModelType.BTL_LINEAR]:
                only_trail_eval = trials[0].evaluation_result[model]
                statistics[model] =  ExperimentStatistics(
                    num_trials=len(trials),
                
                    mean_objective=only_trail_eval.objective_value if only_trail_eval.objective_value else None,
                    std_objective=0 if only_trail_eval.objective_value else None,
                
                    mean_loss=only_trail_eval.loss_value if only_trail_eval.loss_value else None,
                    std_loss=0 if only_trail_eval.loss_value else None,
                
                    mean_violation_percentage=only_trail_eval.violation_percentage,
                    std_violation_percentage=0,
                
                    mean_po_violation_percentage=only_trail_eval.po_violation_percentage,
                    std_po_violation_percentage=0,
                
                    mean_pmc_violation_percentage=only_trail_eval.pmc_violation_percentage,
                    std_pmc_violation_percentage=0,
                
                    mean_fas_size=trials[0].fas_size,
                    std_fas_size=0,
                
                    success_rate = (
                        trials[0].success / len(trials)
                        if trials else 0.0
                    ),
                
                    mean_epsilon_support_percentage=only_trail_eval.epsilon_support_percentage if only_trail_eval.epsilon_support_percentage else None,
                    std_epsilon_support_percentage=0 if only_trail_eval.epsilon_support_percentage else None,
                
                    mean_epsilon_sparsity=only_trail_eval.epsilon_sparsity if only_trail_eval.epsilon_sparsity else None,
                    std_epsilon_sparsity=0 if only_trail_eval.epsilon_sparsity else None,
                
                    mean_epsilon_l1_norm=only_trail_eval.epsilon_l1_norm if only_trail_eval.epsilon_l1_norm else None,
                    std_epsilon_l1_norm=0 if only_trail_eval.epsilon_l1_norm else None,
                
                    mean_z_support_percentage=only_trail_eval.z_support_percentage if only_trail_eval.z_support_percentage else None, 
                    std_z_support_percentage=0 if only_trail_eval.z_support_percentage else None,    
                    )
        else:
            for model in [ModelType.KT_MIN_LP, ModelType.BTL_LINEAR]:
                objectives = [
                    trial.evaluation_result[model].objective_value 
                    for trial in trials
                    if trial.evaluation_result[model].objective_value is not None
                ]

                loss_values = [
                    trial.evaluation_result[model].loss_value 
                    for trial in trials
                    if trial.evaluation_result[model].loss_value is not None
                ]

                violations = [
                    trial.evaluation_result[model].violation_percentage
                    for trial in trials
                    if trial.evaluation_result[model] is not None
                ]

                po_violations = [
                    trial.evaluation_result[model].po_violation_percentage
                    for trial in trials
                    if trial.evaluation_result[model] is not None

                ]

                pmc_violations = [
                    trial.evaluation_result[model].pmc_violation_percentage
                    for trial in trials
                    if trial.evaluation_result[model] is not None

                ]

                fas_sizes = [
                    trial.fas_size
                    for trial in trials
                    if trial.evaluation_result[model] is not None

                ]

                epsilon_support_percentages = [
                    trial.evaluation_result[model].epsilon_support_percentage
                    for trial in trials
                    if trial.evaluation_result[model].epsilon_support_percentage is not None

                ]

                epsilon_sparsities = [
                    trial.evaluation_result[model].epsilon_sparsity
                    for trial in trials
                    if trial.evaluation_result[model].epsilon_sparsity is not None

                ]

                epsilon_l1_norms = [
                    trial.evaluation_result[model].epsilon_l1_norm
                    for trial in trials
                    if trial.evaluation_result[model].epsilon_l1_norm is not None

                ]

                z_support_percentages = [
                    trial.evaluation_result[model].z_support_percentage
                    for trial in trials 
                    if trial.evaluation_result[model].z_support_percentage is not None

                ]

                successful_trials = [
                    trial for trial in trials
                    if trial.success
                    if trial.evaluation_result[model] is not None
                ]

                statistics[model] =  ExperimentStatistics(
                    num_trials=len(trials),

                    mean_objective=fmean(objectives) if len(objectives) else None,
                    std_objective=stdev(objectives) if len(objectives) else None,

                    mean_loss=fmean(loss_values) if len(loss_values) else None,
                    std_loss=stdev(loss_values) if len(loss_values) else None,

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

                    mean_epsilon_support_percentage=mean(epsilon_support_percentages) if len(epsilon_support_percentages) else None,
                    std_epsilon_support_percentage=stdev(epsilon_support_percentages) if len(epsilon_support_percentages) else None,

                    mean_epsilon_sparsity=mean(epsilon_sparsities) if len(epsilon_sparsities) else None,
                    std_epsilon_sparsity=stdev(epsilon_sparsities) if len(epsilon_sparsities) else None,

                    mean_epsilon_l1_norm=mean(epsilon_l1_norms) if len(epsilon_l1_norms) else None,
                    std_epsilon_l1_norm=stdev(epsilon_l1_norms) if len(epsilon_l1_norms) else None,

                    mean_z_support_percentage=mean(z_support_percentages) if len(z_support_percentages) else None, 
                    std_z_support_percentage=stdev(z_support_percentages) if len(z_support_percentages) else None,    
                    )

        return ExperimentResult(
            trials=tuple(trials),
            statistics=statistics,
        )

        
def run_sweep(cfg: DictConfig) -> tuple[ParameterSweepResult, ParameterSweepResult]:
    training_results: dict[float, ExperimentResult] = {}
    testing_results: dict[float, ExperimentResult] = {}

    runner = ExperimentRunner()

    for dispersion in cfg.dispersion.grid:
        phi = float(dispersion)
        logging.info(
            "Starting Mallows experiment: phi=%.2f",
            phi,
        )
        experiment = AllModelsComparisonMallows(
            cfg=cfg,
            dispersion=phi,
        )
        result: tuple[ExperimentResult, ExperimentResult] = runner.run(
            experiment=experiment,
            num_trials=cfg.num_trials,
            seed=cfg.seed,
        )
        training_summary, testing_summary = result
        training_results[phi] = training_summary
        testing_results[phi] = testing_summary

    training_sweep_result = ParameterSweepResult(
        parameter_name=ParameterNameMapping.NORM_PHI.label,
        results=training_results,
    )

    testing_sweep_result = ParameterSweepResult(
        parameter_name=ParameterNameMapping.NORM_PHI.label,
        results=testing_results,
    )
    
    return training_sweep_result, testing_sweep_result

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

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    training_sweep_result, testing_sweep_result = run_sweep(cfg=exp_cfg)

    logger = ExperimentLogger()
    logger.setup_logging(output_dir)
    logging.info("Starting %s", exp_cfg.name)
    
    training_output_dir = output_dir / "training"
    training_plots = ViolationPlots(training_sweep_result)
    figure = training_plots.plot_parameter_summary(
        sweep=training_sweep_result,
        parameter=ParameterNameMapping.NORM_PHI,
        confidence_level=0.95,
    )
    logger.save_figure(
        figure,
        filename=f"mallows_dispersion_summary_{exp_cfg.data.training_num_voters}_{exp_cfg.data.training_num_alternatives}_{exp_cfg.data.dimension}.pdf",
        output_dir=training_output_dir,
        dpi=600
    )
    logger.save_figure(
        figure,
        filename=f"mallows_dispersion_summary_{exp_cfg.data.training_num_voters}_{exp_cfg.data.training_num_alternatives}_{exp_cfg.data.dimension}.png",
        output_dir=training_output_dir, dpi=600)
    logger.save_plots(training_plots.plots, training_output_dir)
    training_plots.clear()
    logger.save(result=training_sweep_result, config=cfg, output_dir=training_output_dir)


    testing_output_dir = output_dir / "testing"
    testing_plots = ViolationPlots(testing_sweep_result)
    figure = testing_plots.plot_parameter_summary(
        sweep=testing_sweep_result,
        parameter=ParameterNameMapping.NORM_PHI,
        confidence_level=0.95,
    )
    logger.save_figure(
        figure,
        filename=f"mallows_dispersion_summary_{exp_cfg.data.testing_num_voters}_{exp_cfg.data.testing_num_alternatives}_{exp_cfg.data.dimension}.pdf", output_dir=testing_output_dir, dpi=600)
    logger.save_figure(
        figure,
        filename=f"mallows_dispersion_summary_{exp_cfg.data.testing_num_voters}_{exp_cfg.data.testing_num_alternatives}_{exp_cfg.data.dimension}.png", output_dir=testing_output_dir, dpi=600)
    logger.save_plots(testing_plots.plots, testing_output_dir)
    testing_plots.clear()
    logger.save(result=testing_sweep_result, config=cfg, output_dir=testing_output_dir)


    logging.info("Experiment completed.")

if __name__ == "__main__":
    main()