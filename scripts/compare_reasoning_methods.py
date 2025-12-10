"""
Comprehensive Reasoning Method Comparison

Evaluates and compares different reasoning approaches on medical MCQs:
1. Chain-of-Thought (CoT) Prompting - Standard LLM reasoning
2. Tree-of-Thought (ToT) Reasoning - Structured multi-branch reasoning
3. Structured Medical Reasoning - 5-step clinical reasoning

Results are saved in a professional report format for analysis.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.embeddings import EmbeddingModel
from models.ollama_model import OllamaModel
from retrieval.faiss_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.multi_stage_retriever import MultiStageRetriever
from reasoning.medical_reasoning import MedicalReasoningEngine
from reasoning.tree_of_thought import TreeOfThoughtReasoner
from improvements.structured_reasoner import StructuredMedicalReasoner
from evaluation.metrics_calculator import MetricsCalculator


@dataclass
class ReasoningResult:
    """Result from a reasoning method evaluation."""
    method_name: str
    description: str
    accuracy: float
    avg_reasoning_time_ms: float
    total_time_seconds: float
    num_cases: int
    correct_predictions: int
    # Confidence calibration metrics
    brier_score: float
    ece: float  # Expected Calibration Error
    # Answer distribution
    answer_distribution: Dict[str, int]
    # Reasoning quality metrics
    avg_reasoning_length: float  # Average length of reasoning text
    avg_evidence_usage: float  # Average number of evidence sources cited
    reasoning_coherence: float  # Coherence score (0-1)


class ReasoningMethodComparator:
    """Compares different reasoning methods."""
    
    def __init__(self, config_path: str):
        """Initialize comparator with config."""
        print("\n[SETUP] Loading models and retrievers...")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = EmbeddingModel(
            model_name=self.config['models']['embedding']['model_name'],
            device=self.config['models']['embedding'].get('device', 'cuda')
        )
        
        # Initialize LLM
        print("Loading LLM model...")
        self.llm_model = OllamaModel(
            model_name=self.config['models']['llm']['model_name'],
            temperature=self.config['models']['llm'].get('temperature', 0.1),
            max_tokens=self.config['models']['llm'].get('max_tokens', 512)
        )
        
        # Initialize FAISS store
        print("Loading FAISS index...")
        self.faiss_store = FAISSVectorStore(
            embedding_model=self.embedding_model
        )
        index_path = Path(self.config['paths']['index_dir'])
        self.faiss_store.load_index(str(index_path))
        
        # Initialize BM25 retriever
        print("Loading BM25 retriever...")
        self.bm25 = BM25Retriever(documents=self.faiss_store.documents)
        
        # Initialize multi-stage retriever for all methods
        print("Initializing multi-stage retriever...")
        self.retriever = MultiStageRetriever(
            faiss_store=self.faiss_store,
            bm25_retriever=self.bm25,
            embedding_model=self.embedding_model,
            stage1_k=150,
            stage2_k=100,
            stage3_k=25
        )
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        print("[OK] All models loaded successfully\n")
    
    def _analyze_reasoning_quality(self, reasoning_text: str, evidence_count: int) -> Dict[str, float]:
        """Analyze quality metrics of reasoning text."""
        # Length analysis
        reasoning_length = len(reasoning_text.split())
        
        # Coherence (simple heuristic: presence of transition words)
        words = set(reasoning_text.lower().split())
        transition_words = {'because', 'therefore', 'thus', 'hence', 'however', 'although', 
                           'furthermore', 'additionally', 'consequently', 'given', 'based'}
        coherence_markers = len(words.intersection(transition_words))
        coherence_score = min(coherence_markers / 3.0, 1.0)  # Normalize to 0-1
        
        return {
            'length': reasoning_length,
            'evidence_usage': evidence_count,
            'coherence': coherence_score
        }
    
    def evaluate_chain_of_thought(self, queries: List[Dict]) -> ReasoningResult:
        """Evaluate standard Chain-of-Thought prompting."""
        print("\n[1/3] Evaluating Chain-of-Thought Prompting...")
        start_time = time.time()
        
        reasoning_times = []
        correct = 0
        confidences = []
        answers = []
        predictions = []
        ground_truths = []
        reasoning_qualities = []
        
        cot_engine = MedicalReasoningEngine(
            embedding_model=self.embedding_model,
            llm_model=self.llm_model
        )
        
        for idx, query_data in enumerate(queries, 1):
            if idx % 10 == 0:
                print(f"  Processing case {idx}/{len(queries)}...")
            
            question = query_data['question']
            case_desc = query_data.get('case_description', '')
            options = query_data['options']
            correct_answer = query_data['correct_answer']
            
            # Retrieve context
            reasoning_start = time.time()
            full_query = f"{case_desc} {question}"
            retrieved_docs = self.retriever.retrieve(full_query, top_k=25)
            
            # Apply Chain-of-Thought reasoning
            result = cot_engine.reason_and_select_answer(
                question=question,
                case_description=case_desc,
                options=options,
                retrieved_contexts=[r.document for r in retrieved_docs]
            )
            
            reasoning_times.append((time.time() - reasoning_start) * 1000)
            
            # Evaluate
            if result.selected_answer == correct_answer:
                correct += 1
            
            confidences.append(result.confidence_score)
            answers.append(result.selected_answer)
            predictions.append(1 if result.selected_answer == correct_answer else 0)
            ground_truths.append(1)
            
            # Analyze reasoning quality
            reasoning_text = result.rationale if hasattr(result, 'rationale') else ""
            evidence_count = len(result.supporting_guidelines) if hasattr(result, 'supporting_guidelines') else 0
            quality = self._analyze_reasoning_quality(reasoning_text, evidence_count)
            reasoning_qualities.append(quality)
        
        total_time = time.time() - start_time
        accuracy = correct / len(queries)
        
        # Calculate calibration metrics
        brier_score = self._calculate_brier_score(confidences, predictions)
        ece = self._calculate_ece(confidences, predictions)
        
        # Calculate quality metrics
        avg_length = np.mean([q['length'] for q in reasoning_qualities])
        avg_evidence = np.mean([q['evidence_usage'] for q in reasoning_qualities])
        avg_coherence = np.mean([q['coherence'] for q in reasoning_qualities])
        
        return ReasoningResult(
            method_name="Chain-of-Thought",
            description="Standard LLM reasoning with step-by-step prompting",
            accuracy=accuracy,
            avg_reasoning_time_ms=np.mean(reasoning_times),
            total_time_seconds=total_time,
            num_cases=len(queries),
            correct_predictions=correct,
            brier_score=brier_score,
            ece=ece,
            answer_distribution=self._count_answers(answers),
            avg_reasoning_length=avg_length,
            avg_evidence_usage=avg_evidence,
            reasoning_coherence=avg_coherence
        )
    
    def evaluate_tree_of_thought(self, queries: List[Dict]) -> ReasoningResult:
        """Evaluate Tree-of-Thought reasoning."""
        print("\n[2/3] Evaluating Tree-of-Thought Reasoning...")
        start_time = time.time()
        
        reasoning_times = []
        correct = 0
        confidences = []
        answers = []
        predictions = []
        ground_truths = []
        reasoning_qualities = []
        
        tot_reasoner = TreeOfThoughtReasoner(llm_model=self.llm_model)
        
        for idx, query_data in enumerate(queries, 1):
            if idx % 10 == 0:
                print(f"  Processing case {idx}/{len(queries)}...")
            
            question = query_data['question']
            case_desc = query_data.get('case_description', '')
            options = query_data['options']
            correct_answer = query_data['correct_answer']
            
            # Retrieve context
            reasoning_start = time.time()
            full_query = f"{case_desc} {question}"
            retrieved_docs = self.retriever.retrieve(full_query, top_k=25)
            
            # Apply Tree-of-Thought reasoning
            result = tot_reasoner.reason(
                question=question,
                case_description=case_desc,
                options=options,
                retrieved_contexts=[r.document for r in retrieved_docs],
                num_snippets=5
            )
            
            reasoning_times.append((time.time() - reasoning_start) * 1000)
            
            # Evaluate
            if result.selected_answer == correct_answer:
                correct += 1
            
            confidences.append(result.confidence_score)
            answers.append(result.selected_answer)
            predictions.append(1 if result.selected_answer == correct_answer else 0)
            ground_truths.append(1)
            
            # Analyze reasoning quality
            reasoning_text = result.final_reasoning if hasattr(result, 'final_reasoning') else ""
            evidence_count = len(getattr(result, 'branches', []))
            quality = self._analyze_reasoning_quality(reasoning_text, evidence_count)
            reasoning_qualities.append(quality)
        
        total_time = time.time() - start_time
        accuracy = correct / len(queries)
        
        # Calculate calibration metrics
        brier_score = self._calculate_brier_score(confidences, predictions)
        ece = self._calculate_ece(confidences, predictions)
        
        # Calculate quality metrics
        avg_length = np.mean([q['length'] for q in reasoning_qualities])
        avg_evidence = np.mean([q['evidence_usage'] for q in reasoning_qualities])
        avg_coherence = np.mean([q['coherence'] for q in reasoning_qualities])
        
        return ReasoningResult(
            method_name="Tree-of-Thought",
            description="Structured multi-branch reasoning with explicit thought paths",
            accuracy=accuracy,
            avg_reasoning_time_ms=np.mean(reasoning_times),
            total_time_seconds=total_time,
            num_cases=len(queries),
            correct_predictions=correct,
            brier_score=brier_score,
            ece=ece,
            answer_distribution=self._count_answers(answers),
            avg_reasoning_length=avg_length,
            avg_evidence_usage=avg_evidence,
            reasoning_coherence=avg_coherence
        )
    
    def evaluate_structured_medical(self, queries: List[Dict]) -> ReasoningResult:
        """Evaluate Structured Medical Reasoning."""
        print("\n[3/3] Evaluating Structured Medical Reasoning...")
        start_time = time.time()
        
        reasoning_times = []
        correct = 0
        confidences = []
        answers = []
        predictions = []
        ground_truths = []
        reasoning_qualities = []
        
        structured_reasoner = StructuredMedicalReasoner(llm_model=self.llm_model)
        
        for idx, query_data in enumerate(queries, 1):
            if idx % 10 == 0:
                print(f"  Processing case {idx}/{len(queries)}...")
            
            question = query_data['question']
            case_desc = query_data.get('case_description', '')
            options = query_data['options']
            correct_answer = query_data['correct_answer']
            
            # Retrieve context
            reasoning_start = time.time()
            full_query = f"{case_desc} {question}"
            retrieved_docs = self.retriever.retrieve(full_query, top_k=25)
            
            # Apply Structured Medical Reasoning
            result = structured_reasoner.reason(
                question=question,
                case_description=case_desc,
                options=options,
                retrieved_contexts=[r.document for r in retrieved_docs]
            )
            
            reasoning_times.append((time.time() - reasoning_start) * 1000)
            
            # Evaluate
            if result.selected_answer == correct_answer:
                correct += 1
            
            confidences.append(result.confidence_score)
            answers.append(result.selected_answer)
            predictions.append(1 if result.selected_answer == correct_answer else 0)
            ground_truths.append(1)
            
            # Analyze reasoning quality
            reasoning_text = result.rationale if hasattr(result, 'rationale') else ""
            evidence_count = len(result.supporting_guidelines) if hasattr(result, 'supporting_guidelines') else 0
            quality = self._analyze_reasoning_quality(reasoning_text, evidence_count)
            reasoning_qualities.append(quality)
        
        total_time = time.time() - start_time
        accuracy = correct / len(queries)
        
        # Calculate calibration metrics
        brier_score = self._calculate_brier_score(confidences, predictions)
        ece = self._calculate_ece(confidences, predictions)
        
        # Calculate quality metrics
        avg_length = np.mean([q['length'] for q in reasoning_qualities])
        avg_evidence = np.mean([q['evidence_usage'] for q in reasoning_qualities])
        avg_coherence = np.mean([q['coherence'] for q in reasoning_qualities])
        
        return ReasoningResult(
            method_name="Structured Medical",
            description="5-step clinical reasoning: feature extraction → differential → evidence → treatment → selection",
            accuracy=accuracy,
            avg_reasoning_time_ms=np.mean(reasoning_times),
            total_time_seconds=total_time,
            num_cases=len(queries),
            correct_predictions=correct,
            brier_score=brier_score,
            ece=ece,
            answer_distribution=self._count_answers(answers),
            avg_reasoning_length=avg_length,
            avg_evidence_usage=avg_evidence,
            reasoning_coherence=avg_coherence
        )
    
    def _calculate_brier_score(self, confidences: List[float], predictions: List[int]) -> float:
        """Calculate Brier score for calibration."""
        brier = np.mean([(conf - pred) ** 2 for conf, pred in zip(confidences, predictions)])
        return float(brier)
    
    def _calculate_ece(self, confidences: List[float], predictions: List[int], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [(conf >= bin_lower) and (conf < bin_upper) 
                     for conf in confidences]
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean([predictions[i] for i, in_b in enumerate(in_bin) if in_b])
                avg_confidence_in_bin = np.mean([confidences[i] for i, in_b in enumerate(in_bin) if in_b])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def _count_answers(self, answers: List[str]) -> Dict[str, int]:
        """Count answer distribution."""
        from collections import Counter
        return dict(Counter(answers))


def load_evaluation_queries(dataset_path: str, num_cases: int = 100, seed: int = 42) -> List[Dict]:
    """Load and prepare evaluation queries."""
    print(f"\n[INFO] Loading {num_cases} evaluation cases from {dataset_path}...")
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data if isinstance(data, list) else data.get('questions', [])
    
    # Sample questions
    np.random.seed(seed)
    if len(questions) > num_cases:
        indices = np.random.choice(len(questions), num_cases, replace=False)
        questions = [questions[i] for i in indices]
    
    print(f"[OK] Loaded {len(questions)} evaluation queries")
    return questions


def generate_comparison_report(results: List[ReasoningResult], output_path: str):
    """Generate professional comparison report."""
    print("\n" + "="*80)
    print("REASONING METHOD COMPARISON REPORT")
    print("="*80)
    
    # Summary statistics
    best_accuracy = max(results, key=lambda x: x.accuracy)
    best_calibration = min(results, key=lambda x: x.brier_score)
    fastest = min(results, key=lambda x: x.avg_reasoning_time_ms)
    
    # Print comparison table
    print("\n" + "-"*80)
    print("PERFORMANCE COMPARISON")
    print("-"*80)
    print(f"{'Method':<25} {'Accuracy':<12} {'Brier':<10} {'ECE':<10} {'Time(ms)':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result.method_name:<25} "
              f"{result.accuracy:<12.4f} "
              f"{result.brier_score:<10.4f} "
              f"{result.ece:<10.4f} "
              f"{result.avg_reasoning_time_ms:<12.2f}")
    
    print("-"*80)
    
    # Key findings
    print("\n" + "-"*80)
    print("KEY FINDINGS")
    print("-"*80)
    print(f"Best Accuracy: {best_accuracy.method_name} ({best_accuracy.accuracy:.4f})")
    print(f"Best Calibration (Brier): {best_calibration.method_name} ({best_calibration.brier_score:.4f})")
    print(f"Fastest: {fastest.method_name} ({fastest.avg_reasoning_time_ms:.2f}ms per case)")
    
    # Reasoning quality analysis
    print("\n" + "-"*80)
    print("REASONING QUALITY METRICS")
    print("-"*80)
    for result in results:
        print(f"{result.method_name}:")
        print(f"  Avg Reasoning Length: {result.avg_reasoning_length:.1f} words")
        print(f"  Avg Evidence Usage: {result.avg_evidence_usage:.1f} sources")
        print(f"  Reasoning Coherence: {result.reasoning_coherence:.3f}")
        print(f"  Correct Predictions: {result.correct_predictions}/{result.num_cases}")
    
    # Save detailed JSON report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_cases': results[0].num_cases,
        'results': [asdict(r) for r in results],
        'summary': {
            'best_accuracy': {
                'method': best_accuracy.method_name,
                'score': best_accuracy.accuracy
            },
            'best_calibration': {
                'method': best_calibration.method_name,
                'brier_score': best_calibration.brier_score
            },
            'fastest': {
                'method': fastest.method_name,
                'time_ms': fastest.avg_reasoning_time_ms
            }
        },
        'recommendations': [
            f"{best_accuracy.method_name} achieves the highest accuracy for medical MCQs.",
            f"{best_calibration.method_name} provides the best-calibrated confidence scores.",
            f"{fastest.method_name} is recommended for time-sensitive applications."
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Detailed report saved to: {output_path}")
    print("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Compare reasoning methods on medical QA')
    parser.add_argument('--num-cases', type=int, default=15, help='Number of cases to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for case selection')
    args = parser.parse_args()
    
    print("="*80)
    print("REASONING METHOD COMPARISON")
    print("="*80)
    print("\nComparing:")
    print("1. Chain-of-Thought Prompting")
    print("2. Tree-of-Thought Reasoning")
    print("3. Structured Medical Reasoning")
    print(f"\nEvaluating on {args.num_cases} cases...")
    
    # Paths
    config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    dataset_path = Path(__file__).parent.parent / "data" / "processed" / "questions" / "questions_1.json"
    output_path = Path(__file__).parent.parent / "reports" / "reasoning_method_comparison.json"
    
    # Load config as JSON (convert YAML if needed)
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Save as temporary JSON
    temp_config = Path(__file__).parent.parent / "config" / "temp_config.json"
    with open(temp_config, 'w') as f:
        json.dump(config, f)
    
    # Initialize comparator
    comparator = ReasoningMethodComparator(str(temp_config))
    
    # Load evaluation queries
    queries = load_evaluation_queries(str(dataset_path), num_cases=args.num_cases, seed=args.seed)
    
    # Run evaluations
    results = []
    results.append(comparator.evaluate_chain_of_thought(queries))
    results.append(comparator.evaluate_tree_of_thought(queries))
    results.append(comparator.evaluate_structured_medical(queries))
    
    # Generate report
    generate_comparison_report(results, str(output_path))
    
    # Cleanup
    temp_config.unlink()


if __name__ == "__main__":
    main()
