"""
Parameter Optimization Module

This module performs systematic parameter optimization using grid search.
Addresses Day 6 goal: Find optimal balance between precision and recall.
"""

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.retrieval_tuner import RetrievalTuner, RetrievalMetrics, TuningResult


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    param_name: str
    param_values: List
    objective: str  # 'precision', 'recall', 'f1', 'balanced'
    constraints: Optional[Dict] = None


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""
    best_params: Dict
    best_score: float
    all_results: List[Tuple[Dict, float]]
    optimization_history: List[Dict]


class ParameterOptimizer:
    """
    Systematic parameter optimization using grid search.
    
    Optimizes:
    - Hybrid search weights
    - Multi-stage weights
    - Similarity thresholds
    - Query expansion parameters
    """
    
    def __init__(
        self,
        retrieval_tuner: RetrievalTuner,
        validation_queries: List[Dict]
    ):
        """
        Initialize parameter optimizer.
        
        Args:
            retrieval_tuner: RetrievalTuner instance
            validation_queries: Validation queries with ground truth
        """
        self.retrieval_tuner = retrieval_tuner
        self.validation_queries = validation_queries
    
    def optimize(
        self,
        configs: List[OptimizationConfig],
        objective: str = 'balanced'
    ) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Args:
            configs: List of optimization configurations
            objective: Optimization objective ('precision', 'recall', 'f1', 'balanced')
        
        Returns:
            OptimizationResult with best parameters
        """
        # Generate all parameter combinations
        param_names = [cfg.param_name for cfg in configs]
        param_value_lists = [cfg.param_values for cfg in configs]
        
        all_combinations = list(itertools.product(*param_value_lists))
        
        all_results = []
        optimization_history = []
        
        for combination in all_combinations:
            # Build parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Check constraints
            if not self._check_constraints(params, configs):
                continue
            
            # Evaluate this configuration
            score, metrics = self._evaluate_configuration(params, objective)
            
            all_results.append((params, score))
            optimization_history.append({
                'params': params,
                'score': score,
                'metrics': metrics
            })
        
        # Find best configuration
        if not all_results:
            raise ValueError("No valid parameter combinations found")
        
        best_params, best_score = max(all_results, key=lambda x: x[1])
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_history=optimization_history
        )
    
    def optimize_hybrid_weights(
        self,
        objective: str = 'balanced'
    ) -> OptimizationResult:
        """
        Optimize hybrid search weights.
        
        Args:
            objective: Optimization objective
        
        Returns:
            OptimizationResult
        """
        # Test weight combinations
        semantic_weights = [0.3, 0.4, 0.5, 0.6, 0.7]
        keyword_weights = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        all_results = []
        
        for sem_weight in semantic_weights:
            for key_weight in keyword_weights:
                # Skip if weights don't make sense
                if abs(sem_weight + key_weight - 1.0) > 0.2:
                    continue
                
                params = {
                    'semantic_weight': sem_weight,
                    'keyword_weight': key_weight
                }
                
                score, metrics = self._evaluate_hybrid_config(params, objective)
                all_results.append((params, score))
        
        best_params, best_score = max(all_results, key=lambda x: x[1])
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_history=[]
        )
    
    def optimize_multi_stage_weights(
        self,
        objective: str = 'balanced'
    ) -> OptimizationResult:
        """
        Optimize multi-stage retrieval weights.
        
        Args:
            objective: Optimization objective
        
        Returns:
            OptimizationResult
        """
        # Test weight combinations (must sum to ~1.0)
        weight_options = [0.2, 0.25, 0.3, 0.33, 0.35, 0.4, 0.45]
        
        all_results = []
        
        for w1 in weight_options:
            for w2 in weight_options:
                for w3 in weight_options:
                    # Check if weights are reasonable
                    total = w1 + w2 + w3
                    if abs(total - 1.0) > 0.15:
                        continue
                    
                    params = {
                        'stage1_weight': w1,
                        'stage2_weight': w2,
                        'stage3_weight': w3
                    }
                    
                    score, metrics = self._evaluate_multi_stage_config(params, objective)
                    all_results.append((params, score))
        
        best_params, best_score = max(all_results, key=lambda x: x[1])
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_history=[]
        )
    
    def _evaluate_configuration(
        self,
        params: Dict,
        objective: str
    ) -> Tuple[float, RetrievalMetrics]:
        """
        Evaluate a parameter configuration.
        
        Args:
            params: Parameter dictionary
            objective: Optimization objective
        
        Returns:
            Tuple of (score, metrics)
        """
        # This is a placeholder - in practice, would create retriever with params
        # and evaluate on validation set
        metrics = RetrievalMetrics(
            precision_at_k={5: 0.0},
            recall_at_k={5: 0.0},
            map_score=0.0,
            retrieval_errors=0,
            avg_score=0.0
        )
        
        score = self._calculate_score(metrics, objective)
        return score, metrics
    
    def _evaluate_hybrid_config(
        self,
        params: Dict,
        objective: str
    ) -> Tuple[float, RetrievalMetrics]:
        """Evaluate hybrid retriever configuration."""
        tuning_result = self.retrieval_tuner.tune_hybrid_weights(
            self.validation_queries,
            weight_combinations=[(params['semantic_weight'], params['keyword_weight'])]
        )
        
        metrics = tuning_result.metrics
        score = self._calculate_score(metrics, objective)
        
        return score, metrics
    
    def _evaluate_multi_stage_config(
        self,
        params: Dict,
        objective: str
    ) -> Tuple[float, RetrievalMetrics]:
        """Evaluate multi-stage retriever configuration."""
        tuning_result = self.retrieval_tuner.tune_multi_stage_weights(
            self.validation_queries,
            stage_weight_combinations=[(
                params['stage1_weight'],
                params['stage2_weight'],
                params['stage3_weight']
            )]
        )
        
        metrics = tuning_result.metrics
        score = self._calculate_score(metrics, objective)
        
        return score, metrics
    
    def _calculate_score(
        self,
        metrics: RetrievalMetrics,
        objective: str
    ) -> float:
        """
        Calculate optimization score based on objective.
        
        Args:
            metrics: RetrievalMetrics
            objective: Optimization objective
        
        Returns:
            Score (higher is better)
        """
        precision = metrics.precision_at_k.get(5, 0.0)
        recall = metrics.recall_at_k.get(5, 0.0)
        map_score = metrics.map_score
        
        if objective == 'precision':
            return precision
        elif objective == 'recall':
            return recall
        elif objective == 'f1':
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return f1
        else:  # 'balanced'
            # Balance precision and recall with MAP
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            balanced_score = 0.4 * precision + 0.4 * recall + 0.2 * map_score
            # Penalty for retrieval errors
            error_penalty = -metrics.retrieval_errors * 0.05
            return balanced_score + error_penalty
    
    def _check_constraints(
        self,
        params: Dict,
        configs: List[OptimizationConfig]
    ) -> bool:
        """Check if parameters satisfy constraints."""
        for cfg in configs:
            if cfg.constraints:
                param_value = params.get(cfg.param_name)
                if param_value is None:
                    continue
                
                # Check min/max constraints
                if 'min' in cfg.constraints and param_value < cfg.constraints['min']:
                    return False
                if 'max' in cfg.constraints and param_value > cfg.constraints['max']:
                    return False
        
        return True

