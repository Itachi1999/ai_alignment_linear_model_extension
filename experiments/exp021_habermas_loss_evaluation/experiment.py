from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean, stdev, fmean
from ai_alignment_linear_model_extension.visualization.plot_data import ParameterSweepResult
import hydra
from omegaconf import DictConfig
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from tqdm.rich import trange

# Experiment imports
from experiments.core.base_experiment import (
    BaseExperiment, ExperimentRunner, ExperimentLogger
)
from experiments.core.experiment_statistics import (
    ExperimentResult, TrialResult, ExperimentStatistics
)
from experiments.core.timer import Timer, TimerResult

# Main Module Imports

from ai_alignment_linear_model_extension.generators.election_generator import  RealElectionGenerator
from ai_alignment_linear_model_extension.preference_graph.preference_graph import PreferenceGraphBuilder, PreferenceEdgeType, PreferenceGraph
from ai_alignment_linear_model_extension.preference_graph.FAS import FeedbackArcSetSolver
from ai_alignment_linear_model_extension.optimization.extension_linear_model import LinearModelSolver
from ai_alignment_linear_model_extension.optimization.linear_KT_min_model import KTMinLinearModelSolver
from ai_alignment_linear_model_extension.optimization.btl_linear import BTLPartialLinearModel
from ai_alignment_linear_model_extension.optimization.hinge_linear import HingePartialLinearModel
from ai_alignment_linear_model_extension.optimization.loss_function_implementation import LP3TrainableLossModel
from ai_alignment_linear_model_extension.evaluation.loss_model_evaluator import LP3LossModelEvaluator
from ai_alignment_linear_model_extension.evaluation.linear_model_evaluator import LinearModelEvaluator
from ai_alignment_linear_model_extension.evaluation.BTL_model_evaluator import BTLModelEvaluator
from ai_alignment_linear_model_extension.visualization.violation_plots import (
    ViolationPlots, ModelType, ParameterNameMapping
)


class Habermas100QuestionsLP(BaseExperiment):
    """ 
    This class implements a synthetic experiment to test the expressibility of LP generated theta against the LP generated theta and epsilon.
    """
    def __init__(self, cfg):
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def setup(self) -> None:
        # self._theta_0 = np.random.normal(loc=self._cfg.data.mean, scale=self._cfg.data.std, size=self._cfg.data.dimension)
        self._json_file_path_list = list(Path(self._cfg.data_path).glob("*.json"))
        
        self._data_generator = RealElectionGenerator(
            d = self._cfg.data.dimension
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
        self._loss_evaluator = LP3LossModelEvaluator()
        # self._evaluator = LinearModelEvaluator()
        self._btl_evaluator = BTLModelEvaluator(eta = self._cfg.optimization.eta)

    def run_trial(
        self,
        trial: int,
        seed: int,
    ) -> TrialResult | tuple[TrialResult, TrialResult]:
        rng = np.random.default_rng(seed)

        timers: list[TimerResult] = []

        selected_json_paths_indices = rng.choice(range(len(self._json_file_path_list)), self._cfg.num_elections, replace=False).tolist()
        
        selected_data_paths = selected_json_paths_indices.copy()
        rng.shuffle(selected_data_paths)
        
        training_data_paths_indices = selected_data_paths[:self._cfg.num_training_elections]
        testing_data_paths_indices = selected_data_paths[self._cfg.num_training_elections:]
        
        assert len(testing_data_paths_indices) == self._cfg.num_test_elections, f"Number of testing elections {len(testing_data_paths_indices)} does not match the expected number {self._cfg.num_test_elections}."
        

        self._training_dag_list = []
        self._training_election_list = []
        self._training_marginal_matrix_list = []
        self.training_alternatives = ()
        
        self._testing_dag_list = []
        self._testing_election_list = []
        self._testing_marginal_matrix_list = []
        self.testing_alternatives = ()

        training_fas_result = None
        # Training elections and graphs
        with Timer("Creating Training ELections and Graphs") as timer:
            # with Timer("Election Generation") as timer:
            for i in training_data_paths_indices:
                json_path = self._json_file_path_list[i]
                election = self._data_generator.generate(json_path=json_path, is_pca=False)
                # timers.append(timer.result)
                logging.debug(f"election preference profile: {election.rankings}")

                # with Timer("Graph Construction") as timer:
                marginal_matrix, graph = self._graph_builder.build(election)
                # timers.append(timer.result)
                logging.debug(f"Marginal Matrix: {marginal_matrix}")
                logging.debug(f"preference graph: {graph}")
                # with Timer("Feedback Arc Set") as timer:
                training_fas_result = self._fas_solver.solve(graph)
                # timers.append(timer.result)
                self.training_alternatives += election.alternatives
                self._training_dag_list.append(training_fas_result.dag)
                self._training_marginal_matrix_list.append(marginal_matrix)
                self._training_election_list.append(election)
                logging.debug(f"FAS Ordering {training_fas_result.ordering}")
                logging.debug(f"Removed Edges: {training_fas_result.removed_edges}")

            self.training_alternatives = tuple(self.training_alternatives)
            self._training_dag_list = tuple(self._training_dag_list)
            self._training_marginal_matrix_list = tuple(self._training_marginal_matrix_list)
            self._training_election_list = tuple(self._training_election_list)
        timers.append(timer.result)

        training_combined_dag = self.combine_dags(self._training_dag_list)
        
        # with Timer("LP2 Solve") as timer:
        #     old_lp_result = self._old_lp_solver.solve(
        #         alternatives=self.training_alternatives,
        #         dimension=self._cfg.data.dimension,
        #         graph=training_combined_dag,
        #     )
        # timers.append(timer.result)
        
        # with Timer("LP3 Solve") as timer:
        #     new_lp_result = self._new_lp_solver.solve(
        #         alternatives=self.training_alternatives,
        #         dimension=self._cfg.data.dimension,
        #         graph=training_combined_dag,
        #     )
            
        # timers.append(timer.result)
        # logging.info(f"LPs are solved")
        with Timer("LP3 Loss Solve") as timer:
            lp3_loss_model = LP3TrainableLossModel(
                graph=training_combined_dag,
                alternatives=self.training_alternatives,
                dimension=self._cfg.data.dimension,
                eta=self._cfg.optimization.eta,
                beta=self._cfg.optimization.beta,
                lambda_=self._cfg.optimization.lambda_,
                kappa=self._cfg.optimization.kappa,
                tau=self._cfg.optimization.tau,
            )
            new_lp_result = lp3_loss_model.fit(max_iter= self._cfg.optimization.max_iterations)
        timers.append(timer.result)
        logging.info(f"LP3 Loss Fitting Time: {timer.result.elapsed_time:.6f} seconds and Convergence: {new_lp_result.converged}, Iterations: {new_lp_result.iterations}, Final Loss: {new_lp_result.objective:.8f}")

        with Timer("BTL Model Fitting") as timer:
            blt_model = BTLPartialLinearModel(
                elections=self._training_election_list,
                marginal_matrices=self._training_marginal_matrix_list
            )
            btl_linear_result = blt_model.fit(max_iter=self._cfg.optimization.max_iterations)
        timers.append(timer.result)
        logging.info(f"BTL Model FItting TIme: {timer.result.elapsed_time:.6f} seconds and Convergence: {btl_linear_result.converged}, Iterations: {btl_linear_result.iterations}, Final Log-Likelihood: {btl_linear_result.log_likelihood}")
        
        # with Timer("Hinge Model Fitting") as timer:
        #     hinge_model = HingePartialLinearModel(
        #         elections=self._training_election_list,
        #         marginal_matrices=self._training_marginal_matrix_list,
        #         margin = self._cfg.optimization.eta,
        #     )
        #     hinge_linear_result = hinge_model.fit(max_iter=self._cfg.optimization.max_iterations)
        # timers.append(timer.result)
        
        # logging.info(f"BTL Convergence: {btl_linear_result.converged}, Iterations: {btl_linear_result.iterations}, Final Log-Likelihood: {btl_linear_result.log_likelihood}")
        # logging.info(f"Hinge Elapsed Time: {timer.result.elapsed_time:.6f} seconds and Convergence: {hinge_linear_result.converged}, Iterations: {hinge_linear_result.iterations}, Final Log-Likelihood: {hinge_linear_result.log_likelihood}")
        
        # Evaluation of the models on the training data
        # with Timer("Old Train LP Evaluation") as timer:
        #     old_training_evaluation = self._evaluator.evaluate(
        #         alternatives=self.training_alternatives,
        #         graph=training_combined_dag,
        #         result=old_lp_result,
        #         testing=False,
        #     )
        # timers.append(timer.result)
        
        with Timer("New Train LP Loss Evaluation") as timer:
            new_training_evaluation = self._loss_evaluator.evaluate(
                alternatives=self.training_alternatives,
                graph=training_combined_dag,
                result=new_lp_result,
                testing=False,
            )
        timers.append(timer.result)

        with Timer("BTL Train Evaluation") as timer:
            btl_training_evaluation = self._btl_evaluator.evaluate(
                training_combined_dag,
                btl_linear_result,
                testing=False,
            )
        timers.append(timer.result)
        
        # with Timer("Hinge Train Evaluation") as timer:
        #     hinge_training_evaluation = self._btl_evaluator.evaluate(
        #         training_combined_dag,
        #         hinge_linear_result,
        #         testing=False,
        #     )
        # timers.append(timer.result)

        training_trial_result = TrialResult(
            trial=trial,
            seed=seed,
            # lp_result=lp_result,
            evaluation_result={
                # ModelType.EPSILON_LP: old_training_evaluation,
                ModelType.LP3_LOSS: new_training_evaluation,
                ModelType.BTL_LINEAR: btl_training_evaluation,
                # ModelType.LINEAR_HINGE: hinge_training_evaluation,
            },
            fas_size=len(training_fas_result.removed_edges),
            timers=tuple(timers),
        )
        
        testing_fas_result = None
        # Training elections and graphs
        with Timer("Creating Training ELections and Graphs") as timer:
            # with Timer("Election Generation") as timer:
            for i in testing_data_paths_indices:
                json_path = self._json_file_path_list[i]
                election = self._data_generator.generate(json_path=json_path, is_pca=False)
                # timers.append(timer.result)
                logging.debug(f"election preference profile: {election.rankings}")
        
                # with Timer("Graph Construction") as timer:
                marginal_matrix, graph = self._graph_builder.build(election)
                # timers.append(timer.result)
                logging.debug(f"Marginal Matrix: {marginal_matrix}")
                logging.debug(f"preference graph: {graph}")
                # with Timer("Feedback Arc Set") as timer:
                testing_fas_result = self._fas_solver.solve(graph)
                # timers.append(timer.result)
                self.testing_alternatives += election.alternatives
                self._testing_dag_list.append(testing_fas_result.dag)
                self._testing_marginal_matrix_list.append(marginal_matrix)
                self._testing_election_list.append(election)
                logging.debug(f"FAS Ordering {testing_fas_result.ordering}")
                logging.debug(f"Removed Edges: {testing_fas_result.removed_edges}")
        
            self.testing_alternatives = tuple(self.testing_alternatives)
            self._testing_dag_list = tuple(self._testing_dag_list)
            self._testing_marginal_matrix_list = tuple(self._testing_marginal_matrix_list)
            self._testing_election_list = tuple(self._testing_election_list)
        timers.append(timer.result)
        
        testing_combined_dag = self.combine_dags(self._testing_dag_list)
        # Evaluation of the models on the testing data
        # with Timer("Old Test LP Evaluation") as timer:
        #     old_testing_evaluation = self._evaluator.evaluate(
        #         alternatives=self.testing_alternatives,
        #         graph=testing_combined_dag,
        #         result=old_lp_result,
        #         testing=True,
        #     )
        # timers.append(timer.result)
        
        with Timer("New Test LP Loss Evaluation") as timer:
            new_testing_evaluation = self._loss_evaluator.evaluate(
                alternatives=self.testing_alternatives,
                graph=testing_combined_dag,
                result=new_lp_result,
                testing=True,
            )
        timers.append(timer.result)
        
        with Timer("BTL Test Evaluation") as timer:
            btl_testing_evaluation = self._btl_evaluator.evaluate(
                testing_combined_dag,
                btl_linear_result,
                testing=True,
                alternatives=self.testing_alternatives
            )
        timers.append(timer.result)
        
        # with Timer("BTL Test Evaluation") as timer:
        #     hinge_testing_evaluation = self._btl_evaluator.evaluate(
        #         testing_combined_dag,
        #         hinge_linear_result,
        #         testing=True,
        #         alternatives=self.testing_alternatives
        #     )
        # timers.append(timer.result)
        
        testing_trial_result = TrialResult(
            trial=trial,
            seed=seed,
            # lp_result=lp_result,
            evaluation_result={
                # ModelType.EPSILON_LP: old_testing_evaluation,
                ModelType.LP3_LOSS: new_testing_evaluation,
                ModelType.BTL_LINEAR: btl_testing_evaluation,
                # ModelType.LINEAR_HINGE: hinge_testing_evaluation,
            },
            fas_size=len(testing_fas_result.removed_edges),
            timers=tuple(timers),
        )
        
        return training_trial_result, testing_trial_result

    def combine_dags(self, dag_list: tuple[PreferenceGraph, ...]) -> PreferenceGraph:
        combined_graph = PreferenceGraph()

        for dag in dag_list:
            combined_graph.add_vertices(dag.vertices)

        for dag in dag_list:
            for edge in dag.edges:
                combined_graph.add_edge(edge=edge)

        return combined_graph

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
            for model in [ModelType.LP3_LOSS, ModelType.BTL_LINEAR]:
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
            for model in [ModelType.LP3_LOSS, ModelType.BTL_LINEAR]:
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

    experiment = Habermas100QuestionsLP(exp_cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    logger = ExperimentLogger()
    logger.setup_logging(output_dir)
    logging.info("Starting %s", exp_cfg.name)

    runner = ExperimentRunner()
    train_result, test_result = runner.run(experiment=experiment, num_trials=exp_cfg.num_trials, seed=exp_cfg.seed)
    
    plots = ViolationPlots(train_result)
    figure = plots.plot_trial_summary(confidence_level=0.95)

    training_output_dir = output_dir / "train"
    training_output_dir.mkdir(parents=True, exist_ok=True)
    logger.save_figure(figure, filename="habermas_100_questions_summary.pdf", output_dir=training_output_dir)
    logger.save_figure(figure, filename="habermas_100_questions_summary.png", output_dir=training_output_dir, dpi=600)
    logger.save_plots(plots.plots, training_output_dir)
    plots.clear()
    logger.save(result=train_result, config=cfg, output_dir=training_output_dir)
    
    
    plots = ViolationPlots(test_result)
    figure = plots.plot_trial_summary(confidence_level=0.95)
    
    testing_output_dir = output_dir / "test"
    testing_output_dir.mkdir(parents=True, exist_ok=True)
    logger.save_figure(figure, filename="habermas_100_questions_summary.pdf", output_dir=testing_output_dir)
    logger.save_figure(figure, filename="habermas_100_questions_summary.png", output_dir=testing_output_dir, dpi=600)
    logger.save_plots(plots.plots, testing_output_dir)
    plots.clear()
    logger.save(result=test_result, config=cfg, output_dir=testing_output_dir)

    logging.info("Experiment completed.")


    # logger.save_summary(result, output_dir)
    # logger.save_trials(result, output_dir)
    # logger.save_config(exp_cfg, output_dir)


if __name__ == "__main__":
    main()