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
        self._evaluator = LinearModelEvaluator()
        self._btl_evaluator = BTLModelEvaluator(eta = self._cfg.optimization.eta)

    def run_trial(
        self,
        trial: int,
        seed: int,
    ) -> TrialResult:
        rng = np.random.default_rng(seed)

        timers: list[TimerResult] = []

        selected_json_paths_indices = rng.choice(range(len(self._json_file_path_list)), self._cfg.num_elections, replace=False).tolist()

        self._dag_list: list[PreferenceGraph] = []
        self._election_list = []
        self._marginal_matrix_list = []
        self.alternatives = ()

        
        with Timer("Creating ELections and Graphs") as timer:
            # with Timer("Election Generation") as timer:
            for i in selected_json_paths_indices:
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
                fas_result = self._fas_solver.solve(graph)
                # timers.append(timer.result)
                self.alternatives += election.alternatives
                self._dag_list.append(fas_result.dag)
                self._marginal_matrix_list.append(marginal_matrix)
                self._election_list.append(election)
                logging.debug(f"FAS Ordering {fas_result.ordering}")
                logging.debug(f"Removed Edges: {fas_result.removed_edges}")

            self.alternatives = tuple(self.alternatives)
            self._dag_list = tuple(self._dag_list)
            self._marginal_matrix_list = tuple(self._marginal_matrix_list)
            self._election_list = tuple(self._election_list)
        timers.append(timer.result)

        combined_dag = self.combine_dags(self._dag_list)
        
        with Timer("LP2 Solve") as timer:
            old_lp_result = self._old_lp_solver.solve(
                alternatives=self.alternatives,
                dimension=self._cfg.data.dimension,
                graph=combined_dag,
            )
        timers.append(timer.result)
        logging.debug("OLD LP:")
        logging.debug(f"Epsilon: {old_lp_result.epsilon}")
        logging.debug(f"Epsilon Norm: {old_lp_result.objective_value}")
        
        with Timer("LP3 Solve") as timer:
            new_lp_result = self._new_lp_solver.solve(
                alternatives=self.alternatives,
                dimension=self._cfg.data.dimension,
                graph=combined_dag,
            )
            
        timers.append(timer.result)
        logging.debug("NEW LP")
        logging.debug(f"Epsilon: {new_lp_result.epsilon}")
        logging.debug(f"Epsilon Norm: {np.sum(np.abs(list(new_lp_result.epsilon.values())), dtype=float)}")
        logging.debug(f"z value: {new_lp_result.z}")
        logging.debug(f"Objetive Value: {new_lp_result.objective_value}")
        
        with Timer("BTL Model Fitting") as timer:
            blt_model = BTLPartialLinearModel(
                elections=self._election_list,
                marginal_matrices=self._marginal_matrix_list
            )
            btl_linear_result = blt_model.fit(max_iter=self._cfg.optimization.max_iterations)
        timers.append(timer.result)
        
        # with Timer("BTL Hinge Model Fitting") as timer:
        #     blt_hinge_model = BTLHingeModel(
        #         marginal_matrix=marginal_matrix,
        #         scores = {alt.id: 0.0 for alt in election.alternatives},
        #         num_voters=self._cfg.data.num_voters,
        #     )
        #     btl_hinge_result = blt_hinge_model.fit(max_iter=self._cfg.optimization.max_iterations)
        # timers.append(timer.result)

        with Timer("Old LP Evaluation") as timer:
            old_evaluation = self._evaluator.evaluate(
                alternatives=self.alternatives,
                graph=combined_dag,
                result=old_lp_result,
            )
        timers.append(timer.result)
        
        with Timer("New LP Evaluation") as timer:
            new_evaluation = self._evaluator.evaluate(
                alternatives=self.alternatives,
                graph=combined_dag,
                result=new_lp_result,
            )
        timers.append(timer.result)

        with Timer("BTL Evaluation") as timer:
            btl_evaluation = self._btl_evaluator.evaluate(
                combined_dag,
                btl_linear_result,
            )
        timers.append(timer.result)

        return TrialResult(
            trial=trial,
            seed=seed,
            # lp_result=lp_result,
            evaluation_result={
                ModelType.EPSILON_LP: old_evaluation,
                ModelType.KT_MIN_LP: new_evaluation,
                ModelType.BTL_LINEAR: btl_evaluation,

            },
            fas_size=len(fas_result.removed_edges),
            timers=tuple(timers),
        )

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
        trials: list[TrialResult],
    ) -> ExperimentResult:
        statistics:dict[ModelType, ExperimentStatistics] = {}
        if len(trials) <= 1:
            for model in [ModelType.EPSILON_LP, ModelType.KT_MIN_LP, ModelType.BTL_LINEAR]:
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
            for model in [ModelType.EPSILON_LP, ModelType.KT_MIN_LP, ModelType.BTL_LINEAR]:
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

# def run_sweep(cfg: DictConfig) -> ParameterSweepResult:
#     results: dict[float, ExperimentResult] = {}

#     runner = ExperimentRunner()

#     data_path = Path(cfg.data_path)
#     json_file_paths = list(data_path.glob("*.json"))

#     logging.info(
#         f"Starting Habermas data with {len(json_file_paths)} questions" 
#     )
#     for i in trange(len(json_file_paths), desc="Question LP Running"):
#         json_file_path = json_file_paths[i]
        
#         experiment = Habermas100QuestionsLP(
#             cfg=cfg,
#             path=json_file_path,
#         )
#         result = runner.run(
#             experiment=experiment,
#             num_trials=cfg.num_trials,
#             seed=cfg.seed,
#         )
#         results[i] = result

#     return ParameterSweepResult(
#         parameter_name=ParameterNameMapping.QUESTION_NUMBER.label,
#         results=results,
#     )


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
    result = runner.run(
        experiment=experiment,
        num_trials=exp_cfg.num_trials,
        seed=exp_cfg.seed,
    )
    
    plots = ViolationPlots(result)
    
    figure = plots.plot_trial_summary(
        confidence_level=0.95,
    )

    logger.save_figure(
        figure,
        filename="habermas_100_questions_summary.pdf",
        output_dir=output_dir,
    )

    logger.save_figure(
        figure,
        filename="habermas_100_questions_summary.png",
        output_dir=output_dir,
        dpi=600,
    )
    
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