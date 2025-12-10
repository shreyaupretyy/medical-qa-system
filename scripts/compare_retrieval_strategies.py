"""
Comprehensive Retrieval Strategy Comparison

This script evaluates and compares different retrieval strategies:
1. Single-stage vs Multi-stage retrieval
2. Concept-based vs Semantic search
3. Hybrid approaches

Results are saved in a professional report format for analysis.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.embeddings import EmbeddingModel
from retrieval.faiss_store import FAISSVectorStore
from retrieval.multi_stage_retriever import MultiStageRetriever
from retrieval.bm25_retriever import BM25Retriever
from evaluation.ground_truth_processor import GroundTruthProcessor
from evaluation.metrics_calculator import MetricsCalculator


@dataclass
class RetrievalResult:
    """Results for a single retrieval strategy."""
    strategy_name: str
    description: str
    map_score: float
    mrr: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    avg_query_time_ms: float
    total_time_seconds: float
    num_cases: int


@dataclass
class ComparisonReport:
    """Complete comparison report."""
    timestamp: str
    num_cases: int
    dataset_path: str
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    recommendations: List[str]


class RetrievalStrategyComparator:
    """Compare different retrieval strategies."""
    
    def __init__(self, embedding_model: EmbeddingModel, faiss_store: FAISSVectorStore):
        self.embedding_model = embedding_model
        self.faiss_store = faiss_store
        self.bm25 = BM25Retriever(documents=faiss_store.documents)
        self.metrics_calculator = MetricsCalculator()
        
    def evaluate_single_stage_faiss(self, queries: List[Dict], k: int = 25) -> RetrievalResult:
        """Evaluate single-stage FAISS semantic search."""
        print("\n[1/6] Evaluating Single-Stage FAISS (Pure Semantic Search)...")
        start_time = time.time()
        query_times = []
        all_results = []
        
        for query_data in queries:
            query = query_data['question']
            query_start = time.time()
            
            # Pure semantic search
            results = self.faiss_store.search(query, top_k=k)
            
            query_times.append((time.time() - query_start) * 1000)
            
            all_results.append({
                'query': query,
                'retrieved_docs': [{'metadata': doc.metadata} for doc, score in results],
                'relevant_docs': query_data['relevant_doc_ids'],
                'expected_guideline_id': query_data['expected_guideline_id'],
                'required_concepts': query_data['required_concepts']
            })
        
        total_time = time.time() - start_time
        metrics = self.metrics_calculator.evaluate_retrieval(all_results)
        
        return RetrievalResult(
            strategy_name="Single-Stage FAISS",
            description="Pure semantic search using dense embeddings (MiniLM-L6-v2)",
            map_score=metrics.map_score,
            mrr=metrics.mrr,
            precision_at_1=metrics.precision_at_k.get(1, 0.0),
            precision_at_3=metrics.precision_at_k.get(3, 0.0),
            precision_at_5=metrics.precision_at_k.get(5, 0.0),
            recall_at_1=metrics.recall_at_k.get(1, 0.0),
            recall_at_3=metrics.recall_at_k.get(3, 0.0),
            recall_at_5=metrics.recall_at_k.get(5, 0.0),
            avg_query_time_ms=np.mean(query_times),
            total_time_seconds=total_time,
            num_cases=len(queries)
        )
    
    def evaluate_single_stage_bm25(self, queries: List[Dict], k: int = 25) -> RetrievalResult:
        """Evaluate single-stage BM25 keyword search."""
        print("\n[2/6] Evaluating Single-Stage BM25 (Pure Keyword Search)...")
        start_time = time.time()
        query_times = []
        all_results = []
        
        for query_data in queries:
            query = query_data['question']
            query_start = time.time()
            
            # Pure keyword search
            results = self.bm25.search(query, top_k=k)
            
            query_times.append((time.time() - query_start) * 1000)
            
            all_results.append({
                'query': query,
                'retrieved_docs': [{'metadata': doc.metadata} for doc, score in results],
                'relevant_docs': query_data['relevant_doc_ids'],
                'expected_guideline_id': query_data['expected_guideline_id'],
                'required_concepts': query_data['required_concepts']
            })
        
        total_time = time.time() - start_time
        metrics = self.metrics_calculator.evaluate_retrieval(all_results)
        
        return RetrievalResult(
            strategy_name="Single-Stage BM25",
            description="Pure keyword-based search using BM25 algorithm",
            map_score=metrics.map_score,
            mrr=metrics.mrr,
            precision_at_1=metrics.precision_at_k.get(1, 0.0),
            precision_at_3=metrics.precision_at_k.get(3, 0.0),
            precision_at_5=metrics.precision_at_k.get(5, 0.0),
            recall_at_1=metrics.recall_at_k.get(1, 0.0),
            recall_at_3=metrics.recall_at_k.get(3, 0.0),
            recall_at_5=metrics.recall_at_k.get(5, 0.0),
            avg_query_time_ms=np.mean(query_times),
            total_time_seconds=total_time,
            num_cases=len(queries)
        )
    
    def evaluate_hybrid_linear(self, queries: List[Dict], k: int = 25, 
                              faiss_weight: float = 0.65, bm25_weight: float = 0.35) -> RetrievalResult:
        """Evaluate hybrid approach with linear combination."""
        print("\n[3/6] Evaluating Hybrid Linear Combination (FAISS + BM25)...")
        start_time = time.time()
        query_times = []
        all_results = []
        
        for query_data in queries:
            query = query_data['question']
            query_start = time.time()
            
            # Get results from both
            faiss_results = self.faiss_store.search(query, top_k=k*2)
            bm25_results = self.bm25.search(query, top_k=k*2)
            
            # Combine scores
            combined_scores = {}
            for doc, score in faiss_results:
                doc_id = doc.metadata.get('guideline_id', '') + '_' + str(doc.metadata.get('chunk_index', ''))
                combined_scores[doc_id] = {
                    'score': score * faiss_weight,
                    'metadata': doc.metadata
                }
            
            for doc, score in bm25_results:
                doc_id = doc.metadata.get('guideline_id', '') + '_' + str(doc.metadata.get('chunk_index', ''))
                if doc_id in combined_scores:
                    combined_scores[doc_id]['score'] += score * bm25_weight
                else:
                    combined_scores[doc_id] = {
                        'score': score * bm25_weight,
                        'metadata': doc.metadata
                    }
            
            # Sort by combined score
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:k]
            
            query_times.append((time.time() - query_start) * 1000)
            
            all_results.append({
                'query': query,
                'retrieved_docs': [{'metadata': r[1]['metadata']} for r in sorted_results],
                'relevant_docs': query_data['relevant_doc_ids'],
                'expected_guideline_id': query_data['expected_guideline_id'],
                'required_concepts': query_data['required_concepts']
            })
        
        total_time = time.time() - start_time
        metrics = self.metrics_calculator.evaluate_retrieval(all_results)
        
        return RetrievalResult(
            strategy_name="Hybrid Linear",
            description=f"Linear combination of FAISS ({faiss_weight}) and BM25 ({bm25_weight})",
            map_score=metrics.map_score,
            mrr=metrics.mrr,
            precision_at_1=metrics.precision_at_k.get(1, 0.0),
            precision_at_3=metrics.precision_at_k.get(3, 0.0),
            precision_at_5=metrics.precision_at_k.get(5, 0.0),
            recall_at_1=metrics.recall_at_k.get(1, 0.0),
            recall_at_3=metrics.recall_at_k.get(3, 0.0),
            recall_at_5=metrics.recall_at_k.get(5, 0.0),
            avg_query_time_ms=np.mean(query_times),
            total_time_seconds=total_time,
            num_cases=len(queries)
        )
    
    def evaluate_multi_stage(self, queries: List[Dict], k: int = 25) -> RetrievalResult:
        """Evaluate multi-stage retrieval (FAISS -> BM25 -> Cross-encoder)."""
        print("\n[4/6] Evaluating Multi-Stage Retrieval (3-stage pipeline)...")
        
        # Initialize multi-stage retriever
        retriever = MultiStageRetriever(
            faiss_store=self.faiss_store,
            bm25_retriever=self.bm25,
            embedding_model=self.embedding_model,
            stage1_k=150,
            stage2_k=100,
            stage3_k=k,
            stage1_weight=0.65,
            stage2_weight=0.35,
            stage3_weight=0.35
        )
        
        start_time = time.time()
        query_times = []
        all_results = []
        
        for query_data in queries:
            query = query_data['question']
            
            query_start = time.time()
            results = retriever.retrieve(query, top_k=k)
            query_times.append((time.time() - query_start) * 1000)
            
            all_results.append({
                'query': query,
                'retrieved_docs': [{'metadata': r.document.metadata} for r in results],
                'relevant_docs': query_data['relevant_doc_ids'],
                'expected_guideline_id': query_data['expected_guideline_id'],
                'required_concepts': query_data['required_concepts']
            })
        
        total_time = time.time() - start_time
        metrics = self.metrics_calculator.evaluate_retrieval(all_results)
        
        return RetrievalResult(
            strategy_name="Multi-Stage (3-stage)",
            description="Stage 1: FAISS (k=150) -> Stage 2: BM25 filter (k=100) -> Stage 3: Cross-encoder rerank (k=25)",
            map_score=metrics.map_score,
            mrr=metrics.mrr,
            precision_at_1=metrics.precision_at_k.get(1, 0.0),
            precision_at_3=metrics.precision_at_k.get(3, 0.0),
            precision_at_5=metrics.precision_at_k.get(5, 0.0),
            recall_at_1=metrics.recall_at_k.get(1, 0.0),
            recall_at_3=metrics.recall_at_k.get(3, 0.0),
            recall_at_5=metrics.recall_at_k.get(5, 0.0),
            avg_query_time_ms=np.mean(query_times),
            total_time_seconds=total_time,
            num_cases=len(queries)
        )
    
    def evaluate_concept_first(self, queries: List[Dict], k: int = 25) -> RetrievalResult:
        """Evaluate concept-first retrieval (BM25 -> FAISS refinement)."""
        print("\n[5/6] Evaluating Concept-First Retrieval (BM25 -> FAISS)...")
        start_time = time.time()
        query_times = []
        all_results = []
        
        for query_data in queries:
            query = query_data['question']
            query_start = time.time()
            
            # Stage 1: BM25 to get concept-relevant documents
            bm25_results = self.bm25.search(query, top_k=k*3)
            
            # Stage 2: Re-rank with semantic search
            bm25_doc_ids = set()
            for doc, score in bm25_results:
                doc_id = doc.metadata.get('guideline_id', '') + '_' + str(doc.metadata.get('chunk_index', ''))
                bm25_doc_ids.add(doc_id)
            
            # Get FAISS results
            faiss_results = self.faiss_store.search(query, top_k=k*3)
            
            # Prioritize documents that appear in both
            final_results = []
            seen = set()
            
            # First, add documents in both (highest relevance)
            for doc, score in faiss_results:
                doc_id = doc.metadata.get('guideline_id', '') + '_' + str(doc.metadata.get('chunk_index', ''))
                if doc_id in bm25_doc_ids and doc_id not in seen:
                    final_results.append({'metadata': doc.metadata})
                    seen.add(doc_id)
                    if len(final_results) >= k:
                        break
            
            # Then add remaining FAISS results
            if len(final_results) < k:
                for doc, score in faiss_results:
                    doc_id = doc.metadata.get('guideline_id', '') + '_' + str(doc.metadata.get('chunk_index', ''))
                    if doc_id not in seen:
                        final_results.append({'metadata': doc.metadata})
                        seen.add(doc_id)
                        if len(final_results) >= k:
                            break
            
            query_times.append((time.time() - query_start) * 1000)
            
            all_results.append({
                'query': query,
                'retrieved_docs': final_results,
                'relevant_docs': query_data['relevant_doc_ids'],
                'expected_guideline_id': query_data['expected_guideline_id'],
                'required_concepts': query_data['required_concepts']
            })
        
        total_time = time.time() - start_time
        metrics = self.metrics_calculator.evaluate_retrieval(all_results)
        
        return RetrievalResult(
            strategy_name="Concept-First",
            description="BM25 keyword filter followed by FAISS semantic refinement",
            map_score=metrics.map_score,
            mrr=metrics.mrr,
            precision_at_1=metrics.precision_at_k.get(1, 0.0),
            precision_at_3=metrics.precision_at_k.get(3, 0.0),
            precision_at_5=metrics.precision_at_k.get(5, 0.0),
            recall_at_1=metrics.recall_at_k.get(1, 0.0),
            recall_at_3=metrics.recall_at_k.get(3, 0.0),
            recall_at_5=metrics.recall_at_k.get(5, 0.0),
            avg_query_time_ms=np.mean(query_times),
            total_time_seconds=total_time,
            num_cases=len(queries)
        )
    
    def evaluate_semantic_first(self, queries: List[Dict], k: int = 25) -> RetrievalResult:
        """Evaluate semantic-first retrieval (FAISS -> BM25 refinement)."""
        print("\n[6/6] Evaluating Semantic-First Retrieval (FAISS -> BM25)...")
        start_time = time.time()
        query_times = []
        all_results = []
        
        for query_data in queries:
            query = query_data['question']
            query_start = time.time()
            
            # Stage 1: FAISS semantic search
            faiss_results = self.faiss_store.search(query, top_k=k*3)
            
            # Stage 2: Filter with BM25
            faiss_doc_ids = set()
            for doc, score in faiss_results:
                doc_id = doc.metadata.get('guideline_id', '') + '_' + str(doc.metadata.get('chunk_index', ''))
                faiss_doc_ids.add(doc_id)
            
            # Get BM25 results
            bm25_results = self.bm25.search(query, top_k=k*3)
            
            # Prioritize documents that appear in both
            final_results = []
            seen = set()
            
            # First, add documents in both (highest relevance)
            for doc, score in bm25_results:
                doc_id = doc.metadata.get('guideline_id', '') + '_' + str(doc.metadata.get('chunk_index', ''))
                if doc_id in faiss_doc_ids and doc_id not in seen:
                    final_results.append({'metadata': doc.metadata})
                    seen.add(doc_id)
                    if len(final_results) >= k:
                        break
            
            # Then add remaining BM25 results
            if len(final_results) < k:
                for doc, score in bm25_results:
                    doc_id = doc.metadata.get('guideline_id', '') + '_' + str(doc.metadata.get('chunk_index', ''))
                    if doc_id not in seen:
                        final_results.append({'metadata': doc.metadata})
                        seen.add(doc_id)
                        if len(final_results) >= k:
                            break
            
            query_times.append((time.time() - query_start) * 1000)
            
            all_results.append({
                'query': query,
                'retrieved_docs': final_results,
                'relevant_docs': query_data['relevant_doc_ids'],
                'expected_guideline_id': query_data['expected_guideline_id'],
                'required_concepts': query_data['required_concepts']
            })
        
        total_time = time.time() - start_time
        metrics = self.metrics_calculator.evaluate_retrieval(all_results)
        
        return RetrievalResult(
            strategy_name="Semantic-First",
            description="FAISS semantic search followed by BM25 keyword refinement",
            map_score=metrics.map_score,
            mrr=metrics.mrr,
            precision_at_1=metrics.precision_at_k.get(1, 0.0),
            precision_at_3=metrics.precision_at_k.get(3, 0.0),
            precision_at_5=metrics.precision_at_k.get(5, 0.0),
            recall_at_1=metrics.recall_at_k.get(1, 0.0),
            recall_at_3=metrics.recall_at_k.get(3, 0.0),
            recall_at_5=metrics.recall_at_k.get(5, 0.0),
            avg_query_time_ms=np.mean(query_times),
            total_time_seconds=total_time,
            num_cases=len(queries)
        )


def load_evaluation_queries(dataset_path: str, num_cases: int = 50, seed: int = 42) -> List[Dict]:
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
    
    # Process with ground truth processor
    processor = GroundTruthProcessor()
    
    # Load guidelines for relevance judgments
    guidelines_dir = Path(__file__).parent.parent / "data" / "guidelines"
    all_guidelines = []
    for filepath in sorted(guidelines_dir.glob('guideline_*.txt')):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.split('\n')
        name = lines[0].replace('# ', '') if lines else ''
        all_guidelines.append({'title': name, 'content': content})
    
    # Process questions
    queries = []
    for question in questions:
        relevance = processor.create_relevance_judgments(question, all_guidelines)
        
        queries.append({
            'question': question.get('question', ''),
            'case_description': question.get('case_description', ''),
            'relevant_doc_ids': relevance['relevant_doc_ids'],
            'expected_guideline_id': relevance['expected_guideline_id'],
            'required_concepts': processor.extract_medical_concepts(question.get('question', ''))
        })
    
    print(f"[OK] Loaded {len(queries)} evaluation queries")
    return queries


def generate_comparison_report(results: List[RetrievalResult], output_path: str):
    """Generate professional comparison report."""
    print("\n" + "="*80)
    print("RETRIEVAL STRATEGY COMPARISON REPORT")
    print("="*80)
    
    # Summary statistics
    best_map = max(results, key=lambda x: x.map_score)
    best_recall = max(results, key=lambda x: x.recall_at_5)
    fastest = min(results, key=lambda x: x.avg_query_time_ms)
    
    # Print comparison table
    print("\n" + "-"*80)
    print("PERFORMANCE COMPARISON")
    print("-"*80)
    print(f"{'Strategy':<25} {'MAP':<8} {'MRR':<8} {'P@5':<8} {'R@5':<8} {'Speed(ms)':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result.strategy_name:<25} "
              f"{result.map_score:<8.4f} "
              f"{result.mrr:<8.4f} "
              f"{result.precision_at_5:<8.4f} "
              f"{result.recall_at_5:<8.4f} "
              f"{result.avg_query_time_ms:<12.2f}")
    
    print("-"*80)
    
    # Key findings
    print("\n" + "-"*80)
    print("KEY FINDINGS")
    print("-"*80)
    print(f"Best MAP Score: {best_map.strategy_name} ({best_map.map_score:.4f})")
    print(f"Best Recall@5: {best_recall.strategy_name} ({best_recall.recall_at_5:.4f})")
    print(f"Fastest: {fastest.strategy_name} ({fastest.avg_query_time_ms:.2f}ms per query)")
    
    # Recommendations
    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)
    
    recommendations = []
    
    if best_map.strategy_name == "Multi-Stage (3-stage)":
        recommendations.append("Multi-stage retrieval provides the best overall performance with cross-encoder reranking.")
    elif best_map.strategy_name.startswith("Hybrid"):
        recommendations.append("Hybrid approaches balance semantic and keyword search effectively.")
    
    if fastest.avg_query_time_ms < 50:
        recommendations.append(f"{fastest.strategy_name} offers excellent speed for real-time applications.")
    
    if best_recall.recall_at_5 > 0.7:
        recommendations.append(f"{best_recall.strategy_name} excels at finding relevant documents (high recall).")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Save detailed report
    report = ComparisonReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_cases=results[0].num_cases if results else 0,
        dataset_path="data/processed/questions/questions_1.json",
        results=[asdict(r) for r in results],
        summary={
            'best_map': {'strategy': best_map.strategy_name, 'score': best_map.map_score},
            'best_recall': {'strategy': best_recall.strategy_name, 'score': best_recall.recall_at_5},
            'fastest': {'strategy': fastest.strategy_name, 'time_ms': fastest.avg_query_time_ms}
        },
        recommendations=recommendations
    )
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(asdict(report), f, indent=2)
    
    print(f"\n[OK] Detailed report saved to: {output_file}")
    print("="*80)


def main():
    """Main comparison workflow."""
    print("="*80)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("="*80)
    print("\nComparing:")
    print("1. Single-Stage FAISS (Semantic)")
    print("2. Single-Stage BM25 (Keyword)")
    print("3. Hybrid Linear Combination")
    print("4. Multi-Stage Pipeline")
    print("5. Concept-First (BM25 -> FAISS)")
    print("6. Semantic-First (FAISS -> BM25)")
    print("\nEvaluating on 100 cases...")
    
    # Initialize components
    print("\n[SETUP] Loading embedding model and FAISS index...")
    embedding_model = EmbeddingModel()
    faiss_store = FAISSVectorStore(embedding_model)
    
    index_dir = Path(__file__).parent.parent / "data" / "indexes"
    faiss_store.load_index(str(index_dir))
    print(f"[OK] Loaded FAISS index with {len(faiss_store.documents)} documents")
    
    # Load evaluation queries
    dataset_path = Path(__file__).parent.parent / "data" / "processed" / "questions" / "questions_1.json"
    queries = load_evaluation_queries(str(dataset_path), num_cases=100, seed=42)
    
    # Initialize comparator
    comparator = RetrievalStrategyComparator(embedding_model, faiss_store)
    
    # Run all evaluations
    results = []
    
    results.append(comparator.evaluate_single_stage_faiss(queries))
    results.append(comparator.evaluate_single_stage_bm25(queries))
    results.append(comparator.evaluate_hybrid_linear(queries))
    results.append(comparator.evaluate_multi_stage(queries))
    results.append(comparator.evaluate_concept_first(queries))
    results.append(comparator.evaluate_semantic_first(queries))
    
    # Generate report
    output_path = Path(__file__).parent.parent / "reports" / "retrieval_strategy_comparison.json"
    generate_comparison_report(results, str(output_path))


if __name__ == "__main__":
    main()
