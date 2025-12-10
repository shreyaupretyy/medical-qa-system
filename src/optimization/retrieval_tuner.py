"""
Retrieval Parameter Tuning Module

This module optimizes retrieval parameters to balance precision and recall.
Addresses Day 6 issue: Over-correction in Day 5 increased retrieval errors from 7 to 32.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.faiss_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.multi_stage_retriever import MultiStageRetriever
from models.embeddings import EmbeddingModel


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval performance."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    map_score: float
    retrieval_errors: int
    avg_score: float


@dataclass
class TuningResult:
    """Result from parameter tuning."""
    best_params: Dict
    metrics: RetrievalMetrics
    all_configs: List[Tuple[Dict, RetrievalMetrics]]


class RetrievalTuner:
    """
    Optimize retrieval parameters based on validation set.
    
    Tests different configurations:
    - Hybrid search weights (FAISS:BM25 ratios)
    - Similarity thresholds
    - Query expansion aggressiveness
    - Multi-stage weights
    """
    
    def __init__(
        self,
        faiss_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        embedding_model: EmbeddingModel
    ):
        """Initialize retrieval tuner."""
        self.faiss_store = faiss_store
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
    
    def tune_hybrid_weights(
        self,
        validation_queries: List[Dict],
        weight_combinations: Optional[List[Tuple[float, float]]] = None
    ) -> TuningResult:
        """
        Tune hybrid search weights (FAISS vs BM25).
        
        Args:
            validation_queries: List of queries with ground truth
            weight_combinations: List of (semantic_weight, keyword_weight) tuples
                               If None, tests default combinations
        
        Returns:
            TuningResult with best configuration
        """
        if weight_combinations is None:
            # Test different weight combinations
            weight_combinations = [
                (0.7, 0.3),  # Favor semantic
                (0.6, 0.4),  # Slightly favor semantic
                (0.5, 0.5),  # Balanced
                (0.4, 0.6),  # Slightly favor keyword
                (0.3, 0.7),  # Favor keyword
            ]
        
        all_results = []
        
        for semantic_weight, keyword_weight in weight_combinations:
            # Create hybrid retriever with these weights
            hybrid = HybridRetriever(
                faiss_store=self.faiss_store,
                bm25_retriever=self.bm25_retriever,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            
            # Evaluate on validation set
            metrics = self._evaluate_retrieval(
                hybrid,
                validation_queries,
                retrieval_type='hybrid'
            )
            
            config = {
                'semantic_weight': semantic_weight,
                'keyword_weight': keyword_weight,
                'type': 'hybrid'
            }
            
            all_results.append((config, metrics))
        
        # Find best configuration (balance precision and recall)
        best_config, best_metrics = max(
            all_results,
            key=lambda x: self._score_config(x[1])
        )
        
        return TuningResult(
            best_params=best_config,
            metrics=best_metrics,
            all_configs=all_results
        )
    
    def tune_multi_stage_weights(
        self,
        validation_queries: List[Dict],
        stage_weight_combinations: Optional[List[Tuple[float, float, float]]] = None
    ) -> TuningResult:
        """
        Tune multi-stage retrieval weights.
        
        Args:
            validation_queries: List of queries with ground truth
            stage_weight_combinations: List of (stage1, stage2, stage3) weight tuples
        
        Returns:
            TuningResult with best configuration
        """
        if stage_weight_combinations is None:
            # Test different weight combinations
            stage_weight_combinations = [
                (0.4, 0.3, 0.3),  # More weight on semantic
                (0.3, 0.4, 0.3),  # More weight on keyword
                (0.3, 0.3, 0.4),  # More weight on reranking (current)
                (0.33, 0.33, 0.34),  # Balanced
                (0.25, 0.35, 0.4),  # Less semantic, more keyword and rerank
            ]
        
        all_results = []
        
        for stage1_weight, stage2_weight, stage3_weight in stage_weight_combinations:
            # Create multi-stage retriever with these weights
            multi_stage = MultiStageRetriever(
                faiss_store=self.faiss_store,
                bm25_retriever=self.bm25_retriever,
                embedding_model=self.embedding_model,
                stage1_weight=stage1_weight,
                stage2_weight=stage2_weight,
                stage3_weight=stage3_weight
            )
            
            # Evaluate on validation set
            metrics = self._evaluate_retrieval(
                multi_stage,
                validation_queries,
                retrieval_type='multi_stage'
            )
            
            config = {
                'stage1_weight': stage1_weight,
                'stage2_weight': stage2_weight,
                'stage3_weight': stage3_weight,
                'type': 'multi_stage'
            }
            
            all_results.append((config, metrics))
        
        # Find best configuration
        best_config, best_metrics = max(
            all_results,
            key=lambda x: self._score_config(x[1])
        )
        
        return TuningResult(
            best_params=best_config,
            metrics=best_metrics,
            all_configs=all_results
        )
    
    def tune_similarity_thresholds(
        self,
        validation_queries: List[Dict],
        thresholds: Optional[List[float]] = None
    ) -> TuningResult:
        """
        Tune similarity score thresholds.
        
        Args:
            validation_queries: List of queries with ground truth
            thresholds: List of threshold values to test
        
        Returns:
            TuningResult with best threshold
        """
        if thresholds is None:
            thresholds = [0.0, 0.1, 0.2, 0.3, 0.4]
        
        all_results = []
        
        for threshold in thresholds:
            # Use multi-stage retriever with threshold
            multi_stage = MultiStageRetriever(
                faiss_store=self.faiss_store,
                bm25_retriever=self.bm25_retriever,
                embedding_model=self.embedding_model
            )
            
            # Evaluate with threshold
            metrics = self._evaluate_retrieval(
                multi_stage,
                validation_queries,
                retrieval_type='multi_stage',
                min_score_threshold=threshold
            )
            
            config = {
                'min_score_threshold': threshold,
                'type': 'threshold'
            }
            
            all_results.append((config, metrics))
        
        # Find best threshold
        best_config, best_metrics = max(
            all_results,
            key=lambda x: self._score_config(x[1])
        )
        
        return TuningResult(
            best_params=best_config,
            metrics=best_metrics,
            all_configs=all_results
        )
    
    def _evaluate_retrieval(
        self,
        retriever,
        validation_queries: List[Dict],
        retrieval_type: str = 'hybrid',
        min_score_threshold: float = 0.0
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance on validation set.
        
        Args:
            retriever: Retrieval object (HybridRetriever or MultiStageRetriever)
            validation_queries: List of queries with ground truth
            retrieval_type: Type of retriever ('hybrid' or 'multi_stage')
            min_score_threshold: Minimum score threshold
        
        Returns:
            RetrievalMetrics
        """
        precision_scores = {1: [], 3: [], 5: [], 10: []}
        recall_scores = {1: [], 3: [], 5: [], 10: []}
        map_scores = []
        retrieval_errors = 0
        all_scores = []
        
        for query_data in validation_queries:
            query = query_data.get('query', '')
            case_desc = query_data.get('case_description', '')
            full_query = f"{case_desc} {query}"
            
            # Get ground truth guideline IDs
            expected_guidelines = set()
            if 'guideline_id' in query_data:
                expected_guidelines.add(query_data['guideline_id'])
            if 'source_guideline' in query_data:
                # Try to match source guideline name to ID
                source = query_data['source_guideline']
                # This is simplified - in practice, need to map names to IDs
                expected_guidelines.add(source)
            
            # Retrieve documents
            try:
                if retrieval_type == 'hybrid':
                    results = retriever.search(
                        full_query,
                        top_k=10,
                        faiss_top_k=20,
                        bm25_top_k=20
                    )
                    retrieved_docs = [r[0] for r in results]
                    scores = [r[1] for r in results]
                else:  # multi_stage
                    results = retriever.retrieve(
                        full_query,
                        top_k=10,
                        min_score_threshold=min_score_threshold
                    )
                    retrieved_docs = [r.document for r in results]
                    scores = [r.final_score for r in results]
                
                all_scores.extend(scores)
                
                # Calculate precision and recall at k
                retrieved_guideline_ids = set()
                for doc in retrieved_docs:
                    guideline_id = doc.metadata.get('guideline_id', '')
                    if guideline_id:
                        # Handle chunk IDs (GL_001_0 -> GL_001)
                        base_id = guideline_id.split('_')[0] + '_' + guideline_id.split('_')[1] if '_' in guideline_id else guideline_id
                        retrieved_guideline_ids.add(base_id)
                
                # Calculate metrics at different k
                for k in [1, 3, 5, 10]:
                    top_k_retrieved = list(retrieved_guideline_ids)[:k]
                    relevant_retrieved = len([g for g in top_k_retrieved if g in expected_guidelines])
                    
                    precision = relevant_retrieved / k if k > 0 else 0.0
                    recall = relevant_retrieved / len(expected_guidelines) if expected_guidelines else 0.0
                    
                    precision_scores[k].append(precision)
                    recall_scores[k].append(recall)
                
                # Calculate MAP (simplified)
                if expected_guidelines:
                    relevant_positions = []
                    for i, doc in enumerate(retrieved_docs[:10], 1):
                        guideline_id = doc.metadata.get('guideline_id', '')
                        base_id = guideline_id.split('_')[0] + '_' + guideline_id.split('_')[1] if '_' in guideline_id else guideline_id
                        if base_id in expected_guidelines:
                            relevant_positions.append(i)
                    
                    if relevant_positions:
                        # Simplified MAP: average precision
                        ap = sum(1.0 / pos for pos in relevant_positions) / len(expected_guidelines)
                        map_scores.append(ap)
                    else:
                        map_scores.append(0.0)
                else:
                    map_scores.append(0.0)
                
                # Check for retrieval errors (no relevant docs retrieved)
                if expected_guidelines and not retrieved_guideline_ids.intersection(expected_guidelines):
                    retrieval_errors += 1
                    
            except Exception as e:
                retrieval_errors += 1
                # Add zeros for this query
                for k in [1, 3, 5, 10]:
                    precision_scores[k].append(0.0)
                    recall_scores[k].append(0.0)
                map_scores.append(0.0)
        
        # Calculate averages
        avg_precision = {k: np.mean(scores) for k, scores in precision_scores.items()}
        avg_recall = {k: np.mean(scores) for k, scores in recall_scores.items()}
        avg_map = np.mean(map_scores) if map_scores else 0.0
        avg_score = np.mean(all_scores) if all_scores else 0.0
        
        return RetrievalMetrics(
            precision_at_k=avg_precision,
            recall_at_k=avg_recall,
            map_score=avg_map,
            retrieval_errors=retrieval_errors,
            avg_score=avg_score
        )
    
    def _score_config(self, metrics: RetrievalMetrics) -> float:
        """
        Score a configuration based on metrics.
        
        Balances precision and recall with penalty for retrieval errors.
        
        Args:
            metrics: RetrievalMetrics object
        
        Returns:
            Combined score (higher is better)
        """
        # Weight precision@5 and recall@5 equally
        precision_score = metrics.precision_at_k.get(5, 0.0)
        recall_score = metrics.recall_at_k.get(5, 0.0)
        
        # F1 score as base
        f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0.0
        
        # Add MAP contribution
        map_contribution = metrics.map_score * 0.3
        
        # Penalty for retrieval errors (normalized by number of queries)
        # This is simplified - in practice, need to know total queries
        error_penalty = -metrics.retrieval_errors * 0.1
        
        return f1 + map_contribution + error_penalty

