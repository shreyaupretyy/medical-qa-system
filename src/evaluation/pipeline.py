"""
Comprehensive Evaluation Pipeline

This module integrates all evaluation components:
- Ground truth processing
- Metrics calculation
- Error analysis
- Visualization and reporting
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ground_truth_processor import GroundTruthProcessor
from evaluation.metrics_calculator import MetricsCalculator
from evaluation.analyzer import ErrorAnalyzer
from evaluation.visualizer import EvaluationVisualizer
from reasoning.rag_pipeline import MultiStageRAGPipeline, load_pipeline, PipelineResult
from reasoning.medical_reasoning import EvidenceMatch


class MedicalQAEvaluator:
    """
    Comprehensive evaluation framework for medical QA system.
    
    Integrates:
    - Ground truth processing
    - Retrieval evaluation (Tier 1)
    - Reasoning evaluation (Tier 2)
    - Systematic analysis (Tier 3)
    - Visualization and reporting
    """
    
    def __init__(
        self,
        pipeline: Optional[MultiStageRAGPipeline] = None,
        output_dir: str = "reports"
    ):
        """
        Initialize evaluator.
        
        Args:
            pipeline: Multi-stage RAG pipeline (if None, will load)
            output_dir: Directory for reports and outputs
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.gt_processor = GroundTruthProcessor()
        self.metrics_calculator = MetricsCalculator()
        self.error_analyzer = ErrorAnalyzer()
        self.visualizer = EvaluationVisualizer(output_dir=str(self.output_dir))
        
        # Results storage
        self.evaluation_results = []
        self.all_pipeline_results = []
    
    def evaluate_system(
        self,
        clinical_cases_path: str,
        guidelines_path: Optional[str] = None,
        split: str = 'all',
        max_cases: Optional[int] = None,
        save_detailed_results: bool = True
    ) -> Dict:
        """
        Run comprehensive evaluation of the system.
        
        Args:
            clinical_cases_path: Path to clinical_cases.json
            guidelines_path: Optional path to guidelines JSON
            split: Dataset split ('all', 'dev', 'test')
            max_cases: Maximum number of cases to evaluate
            save_detailed_results: Whether to save detailed results JSON
            
        Returns:
            Complete evaluation results dictionary
        """
        print("="*70)
        print("COMPREHENSIVE MEDICAL QA EVALUATION")
        print("="*70)
        
        # Step 1: Load and process ground truth
        print("\n[STEP 1] Processing ground truth data...")
        processed_cases = self.gt_processor.process_evaluation_dataset(
            clinical_cases_path=clinical_cases_path,
            guidelines_path=guidelines_path,
            split=split
        )
        
        if max_cases:
            processed_cases = processed_cases[:max_cases]
        
        print(f"  Loaded {len(processed_cases)} evaluation cases")
        
        # Step 2: Load pipeline if not provided
        if self.pipeline is None:
            print("\n[STEP 2] Loading multi-stage RAG pipeline...")
            self.pipeline = load_pipeline()
        
        # Step 3: Run pipeline on all cases
        print(f"\n[STEP 3] Running pipeline on {len(processed_cases)} cases...")
        self.all_pipeline_results = []
        
        start_time = time.time()
        total_cases = len(processed_cases)
        for i, case in enumerate(processed_cases, 1):
            print(f"  Processing case {i}/{total_cases}...", end="\r", flush=True)
            
            # Run pipeline
            result = self.pipeline.answer_question(
                question_id=case['question_id'],
                case_description=case['case_description'],
                question=case['question'],
                options=case['options'],
                correct_answer=case['correct_answer']
            )
            if i % 5 == 0 or i == total_cases:
                print(f"  Processed {i}/{total_cases} cases...          ", flush=True)
            
            # Store result with ground truth
            evaluation_result = {
                'pipeline_result': result,
                'ground_truth': case['ground_truth'],
                'case_metadata': {
                    'question_id': case['question_id'],
                    'category': case['category'],
                    'difficulty': case['difficulty'],
                    'relevance_level': case['relevance_level']
                }
            }
            
            self.all_pipeline_results.append(evaluation_result)
        
        total_time = time.time() - start_time
        print(f"  Completed in {total_time:.1f} seconds ({total_time/len(processed_cases):.2f}s per case)")
        
        # Step 4: Calculate retrieval metrics
        print("\n[STEP 4] Calculating retrieval metrics...")
        retrieval_metrics = self._evaluate_retrieval()
        
        # Step 5: Calculate reasoning metrics
        print("[STEP 5] Calculating reasoning metrics...")
        reasoning_metrics, classification_metrics = self._evaluate_reasoning()
        
        # Step 6: Calculate safety metrics
        print("[STEP 6] Calculating medical safety metrics...")
        safety_metrics = self._evaluate_safety()
        
        # Step 7: Error analysis
        print("[STEP 7] Performing error analysis...")
        error_analysis = self._analyze_errors()
        
        # Step 8: Performance segmentation
        print("[STEP 8] Segmenting performance...")
        performance_segmentation = self._segment_performance()
        
        # Step 9: Build confusion matrix
        print("[STEP 9] Building confusion matrix...")
        confusion_matrix = self._build_confusion_matrix()
        
        # Step 10: Identify pitfalls
        print("[STEP 10] Identifying pitfalls...")
        pitfalls = self._identify_pitfalls()
        
        # Step 11: Generate visualizations
        print("[STEP 11] Generating visualizations...")
        chart_paths = self._generate_visualizations(
            retrieval_metrics, reasoning_metrics, confusion_matrix, error_analysis
        )
        
        # Step 12: Compile comprehensive results
        print("[STEP 12] Compiling evaluation report...")
        comprehensive_results = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'total_cases': len(processed_cases),
                'split': split,
                'evaluation_time_seconds': total_time
            },
            'retrieval': {
                'precision_at_k': {k: float(v) for k, v in retrieval_metrics.precision_at_k.items()},
                'recall_at_k': {k: float(v) for k, v in retrieval_metrics.recall_at_k.items()},
                'map_score': float(retrieval_metrics.map_score),
                'mrr': float(retrieval_metrics.mrr),
                'context_relevance_scores': [float(s) for s in retrieval_metrics.context_relevance_scores],
                'medical_concept_coverage': float(retrieval_metrics.medical_concept_coverage),
                'guideline_coverage': float(retrieval_metrics.guideline_coverage)
            },
            'reasoning': {
                'exact_match_accuracy': float(reasoning_metrics.exact_match_accuracy),
                'semantic_accuracy': float(reasoning_metrics.semantic_accuracy),
                'partial_credit_accuracy': float(reasoning_metrics.partial_credit_accuracy),
                'brier_score': float(reasoning_metrics.brier_score),
                'expected_calibration_error': float(reasoning_metrics.expected_calibration_error),
                'reasoning_chain_completeness': float(reasoning_metrics.reasoning_chain_completeness),
                'evidence_utilization_rate': float(reasoning_metrics.evidence_utilization_rate),
                'confidence_distribution': reasoning_metrics.confidence_distribution,
                'hallucination_rate': float(reasoning_metrics.hallucination_rate),
                'cannot_answer_misuse_rate': float(reasoning_metrics.cannot_answer_misuse_rate),
                'method_accuracy': reasoning_metrics.method_accuracy,
                'cot_tot_delta': float(reasoning_metrics.cot_tot_delta),
                'verifier_pass_rate': float(reasoning_metrics.verifier_pass_rate)
            },
            'classification': {
                'per_option': classification_metrics.per_option,
                'macro_precision': float(classification_metrics.macro_precision),
                'macro_recall': float(classification_metrics.macro_recall),
                'macro_f1': float(classification_metrics.macro_f1),
                'weighted_f1': float(classification_metrics.weighted_f1),
                'balanced_accuracy': float(classification_metrics.balanced_accuracy),
                'confusion_matrix': classification_metrics.confusion_matrix,
                'option_distribution': classification_metrics.option_distribution,
                'fp_fn_breakdown': classification_metrics.fp_fn_breakdown
            },
            'safety': {
                'dangerous_error_count': int(safety_metrics.dangerous_error_count),
                'contraindication_check_accuracy': float(safety_metrics.contraindication_check_accuracy),
                'urgency_recognition_accuracy': float(safety_metrics.urgency_recognition_accuracy),
                'safety_score': float(safety_metrics.safety_score)
            },
            'error_analysis': error_analysis,
            'performance_segmentation': performance_segmentation,
            'confusion_matrix': confusion_matrix,
            'pitfalls': pitfalls,
            'chart_paths': chart_paths,
            'recommendations': self._generate_recommendations(error_analysis, pitfalls)
        }
        
        # Step 13: Save results
        if save_detailed_results:
            results_path = self.output_dir / "evaluation_results.json"
            print(f"\n[STEP 13] Saving detailed results to {results_path}...")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Step 14: Generate HTML report (DISABLED)
        # print("[STEP 14] Generating HTML report...")
        # html_path = self.visualizer.generate_html_report(
        #     comprehensive_results, chart_paths
        # )
        # print(f"  HTML report saved to {html_path}")
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        self._print_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _evaluate_retrieval(self):
        """Evaluate retrieval performance."""
        retrieval_queries = []
        
        for eval_result in self.all_pipeline_results:
            pipeline_result = eval_result['pipeline_result']
            ground_truth = eval_result['ground_truth']
            
            # Extract retrieved documents
            retrieved_docs = []
            for ret_result in pipeline_result.retrieval_results:
                doc = ret_result.document
                retrieved_docs.append({
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'score': ret_result.final_score
                })
            
            # Build query data
            query_data = {
                'retrieved_docs': retrieved_docs,
                'relevant_docs': ground_truth['relevant_doc_ids'],
                'question': pipeline_result.question,
                'expected_guideline_id': ground_truth['expected_guideline_id'],
                'required_concepts': ground_truth['required_concepts']
            }
            
            retrieval_queries.append(query_data)
        
        return self.metrics_calculator.evaluate_retrieval(retrieval_queries)
    
    def _evaluate_reasoning(self):
        """Evaluate reasoning performance."""
        reasoning_results = []
        
        for eval_result in self.all_pipeline_results:
            pipeline_result = eval_result['pipeline_result']
            
            result_dict = {
                'selected_answer': pipeline_result.selected_answer,
                'correct_answer': pipeline_result.correct_answer,
                'confidence_score': pipeline_result.confidence_score,
                'is_correct': pipeline_result.is_correct,
                'options': pipeline_result.options,
                'reasoning_method': pipeline_result.pipeline_metadata.get('reasoning', {}).get('method', 'unknown'),
                'reasoning_steps': [
                    {
                        'description': step.description,
                        'reasoning': step.reasoning,
                        'evidence_used': step.evidence_used
                    }
                    for step in pipeline_result.reasoning.reasoning_steps
                ],
                'supporting_guidelines': pipeline_result.reasoning.supporting_guidelines,
                # New: evidence details for utilization scoring
                # Collect supporting evidence across all options to reward citation
                'supporting_evidence': [
                    {
                        'guideline_id': doc.metadata.get('guideline_id', ''),
                        'excerpt': excerpt,
                        'score': score
                    }
                    for match in getattr(pipeline_result.reasoning, "evidence_matches", {}).values()
                    for (doc, excerpt, score) in match.supporting_evidence
                ]
            }
            
            reasoning_results.append(result_dict)
        
        return self.metrics_calculator.evaluate_reasoning(reasoning_results)
    
    def _evaluate_safety(self):
        """Evaluate medical safety metrics."""
        safety_results = []
        
        for eval_result in self.all_pipeline_results:
            pipeline_result = eval_result['pipeline_result']
            
            result_dict = {
                'is_correct': pipeline_result.is_correct,
                'confidence_score': pipeline_result.confidence_score,
                'reasoning_steps': [
                    {
                        'description': step.description,
                        'reasoning': step.reasoning
                    }
                    for step in pipeline_result.reasoning.reasoning_steps
                ]
            }
            
            safety_results.append(result_dict)
        
        return self.metrics_calculator.evaluate_medical_safety(safety_results)
    
    def _analyze_errors(self):
        """Perform error analysis."""
        results = []
        ground_truths = []
        
        for eval_result in self.all_pipeline_results:
            pipeline_result = eval_result['pipeline_result']
            gt = eval_result['ground_truth']
            
            # Convert to format expected by analyzer
            result_dict = {
                'question_id': pipeline_result.question_id,
                'question': pipeline_result.question,
                'selected_answer': pipeline_result.selected_answer,
                'correct_answer': pipeline_result.correct_answer,
                'is_correct': pipeline_result.is_correct,
                'confidence_score': pipeline_result.confidence_score,
                'retrieval_results': [
                    {
                        'document': {
                            'content': r.document.content,
                            'metadata': r.document.metadata
                        }
                    }
                    for r in pipeline_result.retrieval_results
                ],
                'reasoning': {
                    'reasoning_steps': [
                        {
                            'description': step.description,
                            'reasoning': step.reasoning
                        }
                        for step in pipeline_result.reasoning.reasoning_steps
                    ]
                }
            }
            
            results.append(result_dict)
            ground_truths.append(gt)
        
        # Analyze errors
        error_categories = self.error_analyzer.analyze_errors(results, ground_truths)
        
        # Convert to serializable format
        error_analysis = {
            'error_categories': {
                error_type: {
                    'error_type': cat.error_type,
                    'description': cat.description,
                    'count': cat.count,
                    'examples': cat.examples,
                    'root_causes': cat.root_causes,
                    'proposed_solutions': cat.proposed_solutions
                }
                for error_type, cat in error_categories.items()
            }
        }
        
        return error_analysis
    
    def _segment_performance(self):
        """Segment performance by various dimensions."""
        results = []
        ground_truths = []
        
        for eval_result in self.all_pipeline_results:
            pipeline_result = eval_result['pipeline_result']
            gt = eval_result['ground_truth']
            
            result_dict = {
                'question_id': pipeline_result.question_id,
                'question': pipeline_result.question,
                'selected_answer': pipeline_result.selected_answer,
                'correct_answer': pipeline_result.correct_answer,
                'is_correct': pipeline_result.is_correct,
                'confidence_score': pipeline_result.confidence_score,
                'difficulty': eval_result['case_metadata']['difficulty']
            }
            
            results.append(result_dict)
            
            gt_dict = {
                'category': eval_result['case_metadata']['category'],
                'relevance_level': eval_result['case_metadata']['relevance_level']
            }
            ground_truths.append(gt_dict)
        
        return self.error_analyzer.segment_performance(results, ground_truths)
    
    def _build_confusion_matrix(self):
        """Build confusion matrix."""
        results = []
        ground_truths = []
        
        for eval_result in self.all_pipeline_results:
            pipeline_result = eval_result['pipeline_result']
            gt = eval_result['ground_truth']
            
            result_dict = {
                'question_id': pipeline_result.question_id,
                'pipeline_metadata': {
                    'query_understanding': pipeline_result.pipeline_metadata.get('query_understanding', {})
                },
                'retrieval_results': [
                    {
                        'document': {
                            'metadata': r.document.metadata
                        }
                    }
                    for r in pipeline_result.retrieval_results
                ]
            }
            
            results.append(result_dict)
            ground_truths.append(gt)
        
        return self.error_analyzer.build_confusion_matrix(results, ground_truths)
    
    def _identify_pitfalls(self):
        """Identify common pitfalls."""
        results = []
        ground_truths = []
        
        for eval_result in self.all_pipeline_results:
            pipeline_result = eval_result['pipeline_result']
            gt = eval_result['ground_truth']
            
            result_dict = {
                'question_id': pipeline_result.question_id,
                'case_description': pipeline_result.case_description,
                'question': pipeline_result.question,
                'selected_answer': pipeline_result.selected_answer,
                'correct_answer': pipeline_result.correct_answer,
                'is_correct': pipeline_result.is_correct,
                'confidence_score': pipeline_result.confidence_score,
                'retrieval_results': [
                    {
                        'document': {
                            'content': r.document.content,
                            'metadata': r.document.metadata
                        }
                    }
                    for r in pipeline_result.retrieval_results
                ],
                'reasoning': {
                    'reasoning_steps': [
                        {
                            'description': step.description,
                            'reasoning': step.reasoning
                        }
                        for step in pipeline_result.reasoning.reasoning_steps
                    ]
                }
            }
            
            results.append(result_dict)
            ground_truths.append(gt)
        
        return self.error_analyzer.identify_pitfalls(results, ground_truths)
    
    def _generate_visualizations(
        self,
        retrieval_metrics,
        reasoning_metrics,
        confusion_matrix,
        error_analysis
    ):
        """Generate all visualizations."""
        chart_paths = {}
        
        # Prepare metrics dictionary
        metrics_dict = {
            'retrieval': {
                'precision_at_k': retrieval_metrics.precision_at_k,
                'recall_at_k': retrieval_metrics.recall_at_k,
                'map_score': retrieval_metrics.map_score,
                'context_relevance_scores': retrieval_metrics.context_relevance_scores,
                'medical_concept_coverage': retrieval_metrics.medical_concept_coverage,
                'guideline_coverage': retrieval_metrics.guideline_coverage
            },
            'reasoning': {
                'exact_match_accuracy': reasoning_metrics.exact_match_accuracy,
                'brier_score': reasoning_metrics.brier_score,
                'expected_calibration_error': reasoning_metrics.expected_calibration_error,
                'confidence_distribution': reasoning_metrics.confidence_distribution
            }
        }
        
        # Generate calibration data
        calibration_data = {}
        for eval_result in self.all_pipeline_results:
            pipeline_result = eval_result['pipeline_result']
            if pipeline_result.is_correct is not None:
                conf = pipeline_result.confidence_score
                bin_idx = int(conf * 10)
                bin_name = f"{bin_idx*10}-{(bin_idx+1)*10}%"
                
                if bin_name not in calibration_data:
                    calibration_data[bin_name] = {
                        'predicted_confidence': (bin_idx + 0.5) / 10,
                        'actual_accuracy': 0.0,
                        'count': 0,
                        'correct': 0
                    }
                
                calibration_data[bin_name]['count'] += 1
                if pipeline_result.is_correct:
                    calibration_data[bin_name]['correct'] += 1
        
        # Calculate actual accuracies
        for bin_name in calibration_data:
            if calibration_data[bin_name]['count'] > 0:
                calibration_data[bin_name]['actual_accuracy'] = (
                    calibration_data[bin_name]['correct'] / 
                    calibration_data[bin_name]['count']
                )
        
        # Generate charts
        chart_paths['performance_summary'] = self.visualizer.plot_performance_summary(metrics_dict)
        chart_paths['confusion_matrix'] = self.visualizer.plot_confusion_matrix(confusion_matrix)
        
        # Error analysis
        error_categories = error_analysis.get('error_categories', {})
        pitfalls = self._identify_pitfalls()
        chart_paths['error_analysis'] = self.visualizer.plot_error_analysis(
            error_categories, pitfalls
        )
        
        # Retrieval quality (DISABLED)
        # chart_paths['retrieval_quality'] = self.visualizer.plot_retrieval_quality(
        #     metrics_dict['retrieval']
        # )
        
        return chart_paths
    
    def _generate_recommendations(self, error_analysis, pitfalls):
        """Generate improvement recommendations."""
        recommendations = []
        
        # Based on error analysis
        error_categories = error_analysis.get('error_categories', {})
        
        if 'retrieval' in error_categories:
            recommendations.append(
                "Improve retrieval precision by enhancing medical entity recognition"
            )
            recommendations.append(
                "Expand medical synonym dictionary for better keyword matching"
            )
        
        if 'reasoning' in error_categories:
            recommendations.append(
                "Enhance reasoning chain completeness with more structured steps"
            )
            recommendations.append(
                "Implement evidence aggregation with confidence weighting"
            )
        
        if 'knowledge' in error_categories:
            recommendations.append(
                "Expand medical knowledge base with additional guidelines"
            )
            recommendations.append(
                "Add medical safety checks for contraindications"
            )
        
        # Based on pitfalls
        high_severity_pitfalls = [p for p in pitfalls if p.get('severity') == 'high']
        if high_severity_pitfalls:
            recommendations.append(
                "Implement confidence calibration to reduce overconfident wrong answers"
            )
        
        # General recommendations
        recommendations.extend([
            "Add query expansion with medical terminology",
            "Improve cross-encoder reranking with medical domain fine-tuning",
            "Implement active learning to identify difficult cases"
        ])
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\n[OVERALL PERFORMANCE]")
        reasoning = results.get('reasoning', {})
        print(f"  Answer Accuracy: {reasoning.get('exact_match_accuracy', 0.0):.1%}")
        print(f"  Brier Score: {reasoning.get('brier_score', 0.0):.3f} (lower is better)")
        print(f"  Calibration Error: {reasoning.get('expected_calibration_error', 0.0):.3f}")
        
        print(f"\n[RETRIEVAL PERFORMANCE]")
        retrieval = results.get('retrieval', {})
        print(f"  MAP Score: {retrieval.get('map_score', 0.0):.3f}")
        print(f"  Precision@5: {retrieval.get('precision_at_k', {}).get(5, 0.0):.3f}")
        print(f"  Recall@5: {retrieval.get('recall_at_k', {}).get(5, 0.0):.3f}")
        print(f"  Concept Coverage: {retrieval.get('medical_concept_coverage', 0.0):.1%}")
        
        print(f"\n[ERROR ANALYSIS]")
        error_analysis = results.get('error_analysis', {})
        error_categories = error_analysis.get('error_categories', {})
        for error_type, error_data in error_categories.items():
            print(f"  {error_type.title()}: {error_data.get('count', 0)} errors")
        
        print(f"\n[PITFALLS IDENTIFIED]")
        pitfalls = results.get('pitfalls', [])
        print(f"  Total Pitfalls: {len(pitfalls)}")
        for pitfall in pitfalls[:3]:  # Top 3
            print(f"  - {pitfall.get('pitfall', 'Unknown')}: {pitfall.get('count', 0)} occurrences")
        
        print(f"\n[REPORTS GENERATED]")
        print(f"  Results JSON: {self.output_dir / 'evaluation_results.json'}")
        # print(f"  HTML Report: {self.output_dir / 'evaluation_report.html'}")  # Disabled
        print(f"  Charts: {self.output_dir / 'charts'}")
        
        print("\n" + "="*70)


def main():
    """Run comprehensive evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Medical QA Evaluation')
    parser.add_argument('--cases', type=str, default='data/processed/clinical_cases.json',
                       help='Path to clinical cases JSON')
    parser.add_argument('--guidelines', type=str, default='data/raw/medical_guidelines.json',
                       help='Path to guidelines JSON')
    parser.add_argument('--split', type=str, default='all', choices=['all', 'dev', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--max-cases', type=int, default=None,
                       help='Maximum number of cases to evaluate')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MedicalQAEvaluator(output_dir=args.output_dir)
    
    # Run evaluation
    results = evaluator.evaluate_system(
        clinical_cases_path=args.cases,
        guidelines_path=args.guidelines,
        split=args.split,
        max_cases=args.max_cases
    )
    
    print("\n[OK] Evaluation complete! Check reports directory for detailed results.")


if __name__ == "__main__":
    main()

