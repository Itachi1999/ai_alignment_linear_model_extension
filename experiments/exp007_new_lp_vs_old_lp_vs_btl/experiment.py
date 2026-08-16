from __future__ import annotations

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
from ai_alignment_linear_model_extension.preference_graph.preference_graph import PreferenceGraphBuilder, PreferenceEdgeType
from ai_alignment_linear_model_extension.preference_graph.FAS import FeedbackArcSetSolver
from ai_alignment_linear_model_extension.optimization.extension_linear_model import LinearModelSolver
from ai_alignment_linear_model_extension.optimization.linear_KT_min_model import KTMinLinearModelSolver
from ai_alignment_linear_model_extension.models.btl import BTLModel, BTLResult
from ai_alignment_linear_model_extension.evaluation.linear_model_evaluator import LinearModelEvaluator
from ai_alignment_linear_model_extension.evaluation.BTL_model_evaluator import BTLModelEvaluator
from ai_alignment_linear_model_extension.visualization.violation_plots import (
    ViolationPlots, ModelType
)

class AllModelsComparison(BaseExperiment):
    """ 
    This class implements a synthetic experiment to test the expressibility of LP generated theta against the LP generated theta and epsilon.
    """
    def __init__(self, cfg):
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def setup(self) -> None:
        self.theta_0 = np.random.normal(loc=self._cfg.data.mean, scale=self._cfg.data.std, size=self._cfg.data.dimension)
        self._data_generator = ElectionGenerator(
            feature_generator=GaussianFeatureGenerator(
                mean=self._cfg.data.mean,
                std_dev=self._cfg.data.std,
                seed=self._cfg.seed,
            ),
            preference_generator=KLengthPOPreferenceGenerator(
                seed=self._cfg.seed,
                theta= self.theta_0,
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
        
        self._btl_evaluator = BTLModelEvaluator(eta = self._cfg.optimization.eta)

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
        
        with Timer("BTL Model Fitting") as timer:
            blt_model = BTLModel(
                marginal_matrix=marginal_matrix,
                scores = {alt.id: 0.0 for alt in election.alternatives},
                num_voters=self._cfg.data.num_voters,
            )
            btl_result = blt_model.fit(max_iter=self._cfg.optimization.max_iterations)
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
        
        with Timer("BTL Evaluation") as timer:
            btl_evaluation = self._btl_evaluator.evaluate(
                fas_result.dag,
                btl_result,
            )
        timers.append(timer.result)
        
        # logging.debug(f"The number of PO edges in the original preference graph: {graph.num_po_edges}")
        
        # logging.debug(f"OLD LP Sanity check")
        # # Old LP Sanity Check
        # self.sanity_check(alternatives=election.alternatives, learned_theta=old_lp_result.theta, k = self._cfg.data.k, removed_edges=fas_result.removed_edges)
        
        # logging.debug(f"New LP Sanity check")
        # # New LP Sanity Check
        # self.sanity_check(alternatives=election.alternatives, learned_theta=new_lp_result.theta, k = self._cfg.data.k, removed_edges=fas_result.removed_edges)

        return TrialResult(
            trial=trial,
            seed=seed,
            # lp_result=lp_result,
            evaluation_result={
                ModelType.EPSILON_LP: old_evaluation,
                ModelType.KT_MIN_LP: new_evaluation,
                ModelType.BTL: btl_evaluation
            },
            fas_size=len(fas_result.removed_edges),
            timers=tuple(timers),
        )
    
    def sanity_check(self, alternatives, learned_theta, k, removed_edges = None):
        voter_utilities = {
            alternative.id: float(
                self.theta_0 @ alternative.features
            )
            for alternative in alternatives
        }
        
        # Step 2: Select and order the common k alternatives
        
        po_sequence = tuple(
            sorted(
                voter_utilities,
                key=voter_utilities.get,
                reverse=True,
            )[: k]
        )
        
        alternative_utilities = {
            alternative.id: float(
                learned_theta @ alternative.features
            )
            for alternative in alternatives
        }
        violations_in_main_sequence = [
            (a, b)
            for a, b in zip(po_sequence, po_sequence[1:])
            if alternative_utilities[a] <= alternative_utilities[b]
        ]
        logging.debug(f"The PO sequence: {po_sequence}")
        logging.debug(
            f"Number of PO violations in main sequence: {len(violations_in_main_sequence)}, the violations are: {violations_in_main_sequence}"
        )
        
        
        
        removed_po = [
            edge
            for edge in removed_edges
            if edge.edge_type == PreferenceEdgeType.UNANIMOUS
        ]
        logging.debug(
            f"PO Edges removed in FAS: {len(removed_po)}"
        )
        

    def summarize(
        self,
        trials: list[TrialResult],
    ) -> ExperimentResult:
        statistics:dict[ModelType, ExperimentStatistics] = {}
        for model in [ModelType.EPSILON_LP, ModelType.KT_MIN_LP, ModelType.BTL]:
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
    experiment = AllModelsComparison(exp_cfg)

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