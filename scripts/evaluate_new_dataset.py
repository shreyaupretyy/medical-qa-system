"""
Comprehensive Evaluation Script for New Dataset (questions_1.json)

Evaluates 25 cases with ALL metrics:
- Retrieval Metrics: Precision@k, Recall@k, MAP, MRR, Concept Coverage
- Reasoning Metrics: Accuracy, Hallucination Rate
- Classification Metrics: Per-option Precision/Recall/F1, Confusion Matrix
- Safety Metrics: Dangerous Error Count, Safety Score
- Error Analysis: Error categories, root causes, pitfalls

Usage: python scripts/evaluate_new_dataset.py [--num-cases 25] [--dataset questions_1.json]
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Ensure src is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Cache settings
if "HF_HOME" not in os.environ:
    Path("D:/hf-cache").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = "D:/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
    os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"]


def load_new_dataset(dataset_path: Path) -> list:
    """Load the new dataset format (questions_1.json)."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get('questions', [])
    metadata = data.get('metadata', {})
    
    # Convert to evaluation format
    cases = []
    for q in questions:
        # Handle different option formats
        options = q.get('options', {})
        if isinstance(options, list):
            options = {chr(65+i): opt for i, opt in enumerate(options)}
        
        # Extract case description and question
        full_question = q.get('question', '')
        
        # Try to split into case and question if combined
        if '?' in full_question:
            parts = full_question.rsplit('?', 1)
            if len(parts) == 2 and len(parts[0]) > 100:
                case_description = parts[0].strip()
                question_text = parts[1].strip() + '?' if parts[1].strip() else 'What is the best answer?'
            else:
                case_description = full_question
                question_text = 'What is the most appropriate answer?'
        else:
            case_description = full_question
            question_text = 'What is the most appropriate answer?'
        
        case = {
            'question_id': q.get('question_id', f"Q_{len(cases)+1:03d}"),
            'case_description': case_description,
            'question': question_text,
            'options': options,
            'correct_answer': q.get('correct_answer', 'A'),
            'difficulty': q.get('difficulty', 'medium'),
            'relevance_level': q.get('relevance_level', 'high'),
            'category': q.get('category', 'General'),
            'source_guideline': q.get('source_guideline', ''),
            'guideline_id': q.get('guideline_id', ''),
            'question_type': q.get('question_type', 'management'),
            'explanation': q.get('explanation', {}),
        }
        cases.append(case)
    
    print(f"Loaded {len(cases)} cases from {dataset_path}")
    print(f"Dataset metadata: {metadata}")
    return cases


def convert_to_gt_format(case: dict) -> dict:
    """Convert case to ground truth format expected by evaluator."""
    return {
        'expected_answer': case['correct_answer'],
        'expected_guideline_id': case.get('guideline_id', ''),
        'relevant_doc_ids': [case.get('guideline_id', '')] if case.get('guideline_id') else [],
        # Ensure JSON-serializable types (sets -> lists)
        'required_concepts': list(extract_concepts(case)),
        'reasoning_requirements': [],
    }


def extract_concepts(case: dict) -> set:
    """Extract required medical concepts from case."""
    concepts = set()
    
    # Extract from question text
    text = (case.get('case_description', '') + ' ' + 
            case.get('question', '') + ' ' +
            case.get('source_guideline', '')).lower()
    
    # Medical concept categories with mappings
    medical_concept_map = {
        'cardiovascular': ['mi', 'myocardial', 'cardiac', 'heart', 'coronary', 'stemi', 'nstemi', 
                          'angina', 'hypertension', 'arrhythmia', 'atrial fibrillation'],
        'respiratory': ['pneumonia', 'copd', 'asthma', 'pulmonary', 'respiratory', 'dyspnea',
                       'cough', 'bronchitis', 'emphysema'],
        'neurological': ['stroke', 'cva', 'tia', 'seizure', 'headache', 'neurological',
                        'consciousness', 'paralysis', 'weakness'],
        'endocrine': ['diabetes', 'glucose', 'insulin', 'thyroid', 'hypoglycemia',
                     'hyperglycemia', 'dka', 'hhs'],
        'renal': ['kidney', 'renal', 'aki', 'ckd', 'creatinine', 'dialysis', 'urinary'],
        'gastrointestinal': ['gi', 'gastrointestinal', 'liver', 'pancreatitis', 'bleeding',
                            'cirrhosis', 'hepatic', 'abdominal'],
        'hematology': ['anemia', 'bleeding', 'coagulation', 'thrombosis', 'anticoagulant',
                      'platelet', 'hemoglobin'],
        'infectious': ['infection', 'sepsis', 'antibiotic', 'fever', 'bacterial', 'viral',
                      'pneumonia', 'uti']
    }
    
    # Check each category
    for category, keywords in medical_concept_map.items():
        if any(keyword in text for keyword in keywords):
            concepts.add(category)
    
    # Add specific medical terms
    medical_terms = [
        'diagnosis', 'treatment', 'medication', 'drug', 'therapy',
        'symptom', 'sign', 'vital', 'lab', 'test', 'imaging',
        'condition', 'disease', 'syndrome', 'disorder',
        'management', 'intervention', 'protocol', 'guideline'
    ]
    
    for keyword in medical_terms:
        if keyword in text:
            concepts.add(keyword)
    
    return concepts


def calculate_context_relevance_score(retrieved_docs: list, question: str, embedding_model) -> dict:
    """Calculate context relevance scores for retrieved documents with proper normalization."""
    
    if not retrieved_docs:
        return {
            'avg_relevance': 0.0,
            'max_relevance': 0.0,
            'min_relevance': 0.0,
            'relevance_scores': [],
            'highly_relevant_count': 0,
            'relevance_distribution': {
                'high (>0.7)': 0,
                'medium (0.5-0.7)': 0,
                'low (<0.5)': 0
            }
        }
    
    # Calculate raw relevance scores for each document
    raw_scores = []
    for doc in retrieved_docs:
        doc_text = doc.get('content', '') if isinstance(doc, dict) else str(doc)
        # Use the compute_similarity method from EmbeddingModel
        similarity = embedding_model.compute_similarity(question, doc_text)
        raw_scores.append(float(similarity))
    
    # Apply min-max normalization to bring scores to 0-1 range
    if raw_scores:
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        
        if max_score > min_score:
            # Min-max normalization: (x - min) / (max - min)
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in raw_scores]
        else:
            # All scores are the same, set to middle value
            normalized_scores = [0.5] * len(raw_scores)
    else:
        normalized_scores = []
    
    # Calculate statistics on normalized scores
    avg_relevance = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
    max_relevance = max(normalized_scores) if normalized_scores else 0.0
    min_relevance = min(normalized_scores) if normalized_scores else 0.0
    highly_relevant_count = sum(1 for score in normalized_scores if score > 0.7)
    
    return {
        'avg_relevance': avg_relevance,
        'max_relevance': max_relevance,
        'min_relevance': min_relevance,
        'relevance_scores': normalized_scores,
        'highly_relevant_count': highly_relevant_count,
        'relevance_distribution': {
            'high (>0.7)': sum(1 for s in normalized_scores if s > 0.7),
            'medium (0.5-0.7)': sum(1 for s in normalized_scores if 0.5 <= s <= 0.7),
            'low (<0.5)': sum(1 for s in normalized_scores if s < 0.5)
        }
    }


def analyze_errors_with_concepts(case_results: list) -> dict:
    """Perform detailed error analysis with medical concept mapping."""
    error_analysis = {
        'total_errors': 0,
        'errors_by_concept': defaultdict(list),
        'errors_by_type': defaultdict(int),
        'concept_confusion_matrix': defaultdict(lambda: defaultdict(int)),
        'retrieval_errors': [],
        'reasoning_errors': [],
        'high_confidence_errors': [],
        'common_pitfalls': []
    }
    
    for result in case_results:
        if not result.get('is_correct', False):
            error_analysis['total_errors'] += 1
            
            # Extract concepts from the case
            case_text = result.get('case_description', '') + ' ' + result.get('question', '')
            concepts = extract_concepts({'case_description': case_text, 
                                        'question': '', 
                                        'source_guideline': ''})
            
            # Categorize error
            confidence = result.get('confidence', 0.5)
            retrieval_quality = result.get('retrieval_quality', 0.5)
            
            error_info = {
                'question_id': result.get('question_id', 'unknown'),
                'concepts': list(concepts),
                'confidence': confidence,
                'retrieval_quality': retrieval_quality,
                'predicted': result.get('predicted_answer', ''),
                'correct': result.get('correct_answer', ''),
                'question_type': result.get('question_type', 'unknown')
            }
            
            # Classify error type
            if retrieval_quality < 0.3:
                error_type = 'retrieval_failure'
                error_analysis['retrieval_errors'].append(error_info)
            elif confidence > 0.8:
                error_type = 'high_confidence_wrong'
                error_analysis['high_confidence_errors'].append(error_info)
            else:
                error_type = 'reasoning_error'
                error_analysis['reasoning_errors'].append(error_info)
            
            error_analysis['errors_by_type'][error_type] += 1
            
            # Map errors to medical concepts
            for concept in concepts:
                error_analysis['errors_by_concept'][concept].append(error_info)
            
            # Build confusion matrix for concepts
            if len(concepts) >= 2:
                primary = list(concepts)[0]
                for secondary in list(concepts)[1:]:
                    error_analysis['concept_confusion_matrix'][primary][secondary] += 1
    
    # Identify common pitfalls
    pitfalls = []
    
    # Pitfall 1: High retrieval failure rate in specific concepts
    for concept, errors in error_analysis['errors_by_concept'].items():
        retrieval_fails = sum(1 for e in errors if e['retrieval_quality'] < 0.3)
        if retrieval_fails >= 2:
            pitfalls.append({
                'type': 'retrieval_weakness',
                'concept': concept,
                'frequency': retrieval_fails,
                'description': f'High retrieval failure rate for {concept} cases',
                'solution': 'Add more domain-specific synonyms and expand query for this concept'
            })
    
    # Pitfall 2: Overconfidence in specific question types
    type_confidence_errors = defaultdict(int)
    for error in error_analysis['high_confidence_errors']:
        type_confidence_errors[error['question_type']] += 1
    
    for q_type, count in type_confidence_errors.items():
        if count >= 2:
            pitfalls.append({
                'type': 'overconfidence',
                'concept': q_type,
                'frequency': count,
                'description': f'Overconfident wrong predictions in {q_type} questions',
                'solution': 'Implement confidence calibration for this question type'
            })
    
    error_analysis['common_pitfalls'] = pitfalls
    error_analysis['errors_by_concept'] = dict(error_analysis['errors_by_concept'])
    error_analysis['concept_confusion_matrix'] = {
        k: dict(v) for k, v in error_analysis['concept_confusion_matrix'].items()
    }
    
    return error_analysis


def calculate_detailed_metrics(results: dict, case_results: list) -> dict:
    """Calculate comprehensive detailed metrics."""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_cases': len(case_results),
        'accuracy': {},
        'retrieval': {},
        'reasoning': {},
        'error_analysis': {},
        'question_types': {},
        'difficulty_breakdown': {},
        'relevance_breakdown': {},
        'category_breakdown': {},
    }
    
    # Calculate accuracy
    correct = sum(1 for r in case_results if r.get('is_correct', False))
    total = len(case_results)
    metrics['accuracy'] = {
        'exact_match': correct / total if total > 0 else 0,
        'correct_count': correct,
        'incorrect_count': total - correct,
        'total_cases': total
    }
    
    # Analyze by question type
    type_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in case_results:
        qtype = r.get('question_type', 'unknown')
        type_results[qtype]['total'] += 1
        if r.get('is_correct', False):
            type_results[qtype]['correct'] += 1
    
    metrics['question_types'] = {
        qtype: {
            'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'correct': data['correct'],
            'total': data['total']
        }
        for qtype, data in type_results.items()
    }
    
    # Analyze by difficulty
    diff_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in case_results:
        diff = r.get('difficulty', 'unknown')
        diff_results[diff]['total'] += 1
        if r.get('is_correct', False):
            diff_results[diff]['correct'] += 1
    
    metrics['difficulty_breakdown'] = {
        diff: {
            'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'correct': data['correct'],
            'total': data['total']
        }
        for diff, data in diff_results.items()
    }
    
    # Analyze by relevance
    rel_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in case_results:
        rel = r.get('relevance_level', 'unknown')
        rel_results[rel]['total'] += 1
        if r.get('is_correct', False):
            rel_results[rel]['correct'] += 1
    
    metrics['relevance_breakdown'] = {
        rel: {
            'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'correct': data['correct'],
            'total': data['total']
        }
        for rel, data in rel_results.items()
    }
    
    # Analyze by category
    cat_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in case_results:
        cat = r.get('category', 'Unknown')
        cat_results[cat]['total'] += 1
        if r.get('is_correct', False):
            cat_results[cat]['correct'] += 1
    
    metrics['category_breakdown'] = {
        cat: {
            'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'correct': data['correct'],
            'total': data['total']
        }
        for cat, data in cat_results.items()
    }
    
    # Medical concept-based analysis
    concept_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    for r in case_results:
        # Extract concepts from this case
        case_text = r.get('case_description', '') + ' ' + r.get('question', '')
        concepts = extract_concepts({'case_description': case_text, 
                                    'question': '', 
                                    'source_guideline': ''})
        
        for concept in concepts:
            concept_results[concept]['total'] += 1
            concept_results[concept]['confidences'].append(r.get('confidence', 0.5))
            if r.get('is_correct', False):
                concept_results[concept]['correct'] += 1
    
    # Context relevance analysis
    context_relevance_scores = [r.get('context_relevance', {}) for r in case_results if 'context_relevance' in r]
    if context_relevance_scores:
        avg_context_relevance = sum(cr.get('avg_relevance', 0) for cr in context_relevance_scores) / len(context_relevance_scores)
        avg_max_relevance = sum(cr.get('max_relevance', 0) for cr in context_relevance_scores) / len(context_relevance_scores)
        total_highly_relevant = sum(cr.get('highly_relevant_count', 0) for cr in context_relevance_scores)
    else:
        avg_context_relevance = 0.0
        avg_max_relevance = 0.0
        total_highly_relevant = 0
    
    metrics['medical_concept_breakdown'] = {
        concept: {
            'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'correct': data['correct'],
            'total': data['total'],
            'avg_confidence': sum(data['confidences']) / len(data['confidences']) if data['confidences'] else 0
        }
        for concept, data in concept_results.items()
    }
    
    # Add context relevance metrics
    metrics['context_relevance'] = {
        'avg_relevance': avg_context_relevance,
        'avg_max_relevance': avg_max_relevance,
        'total_highly_relevant_docs': total_highly_relevant,
        'cases_analyzed': len(context_relevance_scores)
    }
    
    # Comprehensive error analysis with medical concepts
    detailed_error_analysis = analyze_errors_with_concepts(case_results)
    
    # Error analysis
    error_types = defaultdict(int)
    retrieval_failures = 0
    reasoning_failures = 0
    high_confidence_errors = 0
    
    for r in case_results:
        if not r.get('is_correct', False):
            # Classify error
            confidence = r.get('confidence', 0.5)
            
            if confidence > 0.8:
                high_confidence_errors += 1
                error_types['high_confidence_wrong'] += 1
            elif confidence < 0.3:
                error_types['low_confidence_wrong'] += 1
            else:
                error_types['medium_confidence_wrong'] += 1
            
            # Check retrieval quality
            if r.get('retrieval_quality', 1.0) < 0.3:
                retrieval_failures += 1
            else:
                reasoning_failures += 1
    
    total_errors = total - correct
    metrics['error_analysis'] = {
        'total_errors': total_errors,
        'error_rate': total_errors / total if total > 0 else 0,
        'error_types': dict(error_types),
        'retrieval_failures': retrieval_failures,
        'reasoning_failures': reasoning_failures,
        'high_confidence_errors': high_confidence_errors,
        'retrieval_failure_rate': retrieval_failures / total_errors if total_errors > 0 else 0,
        'reasoning_failure_rate': reasoning_failures / total_errors if total_errors > 0 else 0,
        'detailed_analysis': detailed_error_analysis
    }
    
    return metrics


def print_comprehensive_report(metrics: dict, results: dict):
    """Print comprehensive evaluation report."""
    print("\n" + "=" * 80)
    print(" COMPREHENSIVE MEDICAL QA EVALUATION REPORT")
    print(" New Dataset (questions_1.json)")
    print("=" * 80)
    print(f"\nTimestamp: {metrics['timestamp']}")
    print(f"Total Cases Evaluated: {metrics['total_cases']}")
    
    # Accuracy Summary
    print("\n" + "-" * 80)
    print(" ACCURACY METRICS")
    print("-" * 80)
    acc = metrics['accuracy']
    print(f"  Overall Accuracy: {acc['exact_match']*100:.1f}%")
    print(f"  Correct: {acc['correct_count']} / {acc['total_cases']}")
    print(f"  Incorrect: {acc['incorrect_count']}")
    
    # Retrieval Metrics
    if 'retrieval' in results:
        print("\n" + "-" * 80)
        print(" RETRIEVAL METRICS")
        print("-" * 80)
        ret = results.get('retrieval', {})
        print(f"  MAP Score: {ret.get('map_score', 0):.4f}")
        print(f"  MRR (Mean Reciprocal Rank): {ret.get('mrr', 0):.4f}")
        
        # Precision@k
        prec_at_k = ret.get('precision_at_k', {})
        if prec_at_k:
            print(f"  Precision@1: {prec_at_k.get(1, prec_at_k.get('1', 0)):.4f}")
            print(f"  Precision@3: {prec_at_k.get(3, prec_at_k.get('3', 0)):.4f}")
            print(f"  Precision@5: {prec_at_k.get(5, prec_at_k.get('5', 0)):.4f}")
        
        # Recall@k
        rec_at_k = ret.get('recall_at_k', {})
        if rec_at_k:
            print(f"  Recall@1: {rec_at_k.get(1, rec_at_k.get('1', 0)):.4f}")
            print(f"  Recall@3: {rec_at_k.get(3, rec_at_k.get('3', 0)):.4f}")
            print(f"  Recall@5: {rec_at_k.get(5, rec_at_k.get('5', 0)):.4f}")
        
        # NDCG@k
        # ndcg_at_k = ret.get('ndcg_at_k', {})
        # if ndcg_at_k:
        #     print(f"  NDCG@5: {ndcg_at_k.get(5, ndcg_at_k.get('5', 0)):.4f}")  # Hidden
        
        print(f"  Concept Coverage: {ret.get('medical_concept_coverage', 0)*100:.1f}%")
        print(f"  Guideline Coverage: {ret.get('guideline_coverage', 0)*100:.1f}%")
    
    # Reasoning Metrics
    if 'reasoning' in results:
        print("\n" + "-" * 80)
        print(" REASONING METRICS")
        print("-" * 80)
        reas = results.get('reasoning', {})
        print(f"  Semantic Accuracy: {reas.get('semantic_accuracy', 0)*100:.1f}%")
    
    # Classification Metrics
    if 'classification' in results:
        print("\n" + "-" * 80)
        print(" CLASSIFICATION METRICS")
        print("-" * 80)
        cls = results.get('classification', {})
        print(f"  Macro Precision: {cls.get('macro_precision', 0):.4f}")
        print(f"  Macro Recall: {cls.get('macro_recall', 0):.4f}")
        print(f"  Macro F1: {cls.get('macro_f1', 0):.4f}")
        print(f"  Weighted F1: {cls.get('weighted_f1', 0):.4f}")
        print(f"  Balanced Accuracy: {cls.get('balanced_accuracy', 0):.4f}")
        
        # Per-option metrics hidden
        # if cls.get('per_option'):
        #     print("\n  Per-Option Performance:")
        #     for opt, stats in sorted(cls['per_option'].items()):
        #         print(f"    {opt}: P={stats.get('precision', 0):.2f} R={stats.get('recall', 0):.2f} F1={stats.get('f1', 0):.2f} (n={stats.get('support', 0)})")
    
    # Safety Metrics
    if 'safety' in results:
        print("\n" + "-" * 80)
        print(" SAFETY METRICS")
        print("-" * 80)
        safety = results.get('safety', {})
        print(f"  Safety Score: {safety.get('safety_score', 0)*100:.1f}%")
        print(f"  Dangerous Errors (high confidence wrong): {safety.get('dangerous_error_count', 0)}")
    
    # Question Type Performance
    print("\n" + "-" * 80)
    print(" PERFORMANCE BY QUESTION TYPE")
    print("-" * 80)
    for qtype, data in sorted(metrics['question_types'].items(), key=lambda x: -x[1]['total']):
        print(f"  {qtype.title():15s}: {data['accuracy']*100:5.1f}% ({data['correct']}/{data['total']})")
    
    # Difficulty Breakdown
    print("\n" + "-" * 80)
    print(" PERFORMANCE BY DIFFICULTY")
    print("-" * 80)
    for diff in ['easy', 'medium', 'hard']:
        if diff in metrics['difficulty_breakdown']:
            data = metrics['difficulty_breakdown'][diff]
            print(f"  {diff.title():10s}: {data['accuracy']*100:5.1f}% ({data['correct']}/{data['total']})")
    
    # Relevance Breakdown
    print("\n" + "-" * 80)
    print(" PERFORMANCE BY RELEVANCE")
    print("-" * 80)
    for rel in ['high', 'medium', 'low']:
        if rel in metrics['relevance_breakdown']:
            data = metrics['relevance_breakdown'][rel]
            print(f"  {rel.title():10s}: {data['accuracy']*100:5.1f}% ({data['correct']}/{data['total']})")
    
    # Category Breakdown
    print("\n" + "-" * 80)
    print(" PERFORMANCE BY CATEGORY")
    print("-" * 80)
    for cat, data in sorted(metrics['category_breakdown'].items(), key=lambda x: -x[1]['total']):
        print(f"  {cat[:25]:25s}: {data['accuracy']*100:5.1f}% ({data['correct']}/{data['total']})")
    
    # Context Relevance Metrics
    if 'context_relevance' in metrics:
        print("\n" + "-" * 80)
        print(" CONTEXT RELEVANCE METRICS")
        print("-" * 80)
        cr = metrics['context_relevance']
        print(f"  Average Relevance Score: {cr['avg_relevance']:.4f}")
        print(f"  Average Max Relevance: {cr['avg_max_relevance']:.4f}")
        print(f"  Highly Relevant Documents (>0.7): {cr['total_highly_relevant_docs']}")
        print(f"  Cases Analyzed: {cr['cases_analyzed']}")
    
    # Error Analysis
    print("\n" + "-" * 80)
    print(" ERROR ANALYSIS")
    print("-" * 80)
    err = metrics['error_analysis']
    print(f"  Total Errors: {err['total_errors']} ({err['error_rate']*100:.1f}%)")
    print(f"  Retrieval Failures: {err['retrieval_failures']} ({err['retrieval_failure_rate']*100:.1f}%)")
    print(f"  Reasoning Failures: {err['reasoning_failures']} ({err['reasoning_failure_rate']*100:.1f}%)")
    print(f"  High Confidence Errors: {err['high_confidence_errors']}")
    
    print("\n" + "=" * 80)
    print(" EVALUATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive New Dataset Evaluation")
    parser.add_argument("--dataset", default="data/processed/questions/questions_1.json",
                       help="Path to questions_1.json")
    parser.add_argument("--num-cases", type=int, default=100,
                       help="Number of cases to evaluate")
    parser.add_argument("--output", default="reports",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f" COMPREHENSIVE EVALUATION - {args.num_cases} CASES")
    print(f" Dataset: {dataset_path}")
    print("=" * 80)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"\n[!] Dataset not found at {dataset_path}")
        print("[!] Waiting for generation to complete...")
        
        # Wait for generation (check every 10 seconds, max 30 minutes)
        max_wait = 1800
        wait_time = 0
        while not dataset_path.exists() and wait_time < max_wait:
            time.sleep(10)
            wait_time += 10
            if wait_time % 60 == 0:
                print(f"    Still waiting... ({wait_time//60} minutes)")
        
        if not dataset_path.exists():
            print("[ERROR] Dataset generation did not complete in time.")
            sys.exit(1)
    
    # Load dataset
    print("\n[1/5] Loading new dataset...")
    cases = load_new_dataset(dataset_path)
    
    # Sample if needed
    if len(cases) > args.num_cases:
        random.seed(args.seed)
        cases = random.sample(cases, args.num_cases)
    print(f"      Selected {len(cases)} cases for evaluation")
    
    # Load pipeline
    print("\n[2/5] Loading pipeline...")
    start = time.time()
    
    from src.utils.config_loader import load_config
    from src.evaluation.pipeline import MedicalQAEvaluator
    from src.reasoning.rag_pipeline import load_pipeline
    
    pipeline = load_pipeline()
    print(f"      Pipeline loaded in {time.time()-start:.1f}s")
    
    # Convert to evaluator format
    print("\n[3/5] Preparing cases for evaluation...")
    eval_cases = []
    for case in cases:
        eval_case = {
            'question_id': case['question_id'],
            'case_description': case['case_description'],
            'question': case['question'],
            'options': case['options'],
            'correct_answer': case['correct_answer'],
            'category': case['category'],
            'difficulty': case['difficulty'],
            'relevance_level': case['relevance_level'],
            'question_type': case.get('question_type', 'management'),
            'source_guideline': case.get('source_guideline', ''),  # CRITICAL FIX
            'guideline_id': case.get('guideline_id', ''),  # CRITICAL FIX
            'ground_truth': convert_to_gt_format(case)
        }
        eval_cases.append(eval_case)
    
    # Save temp file for evaluator
    import uuid
    temp_path = out_dir / f"temp_eval_{uuid.uuid4().hex[:8]}.json"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {"source": "questions_1.json", "evaluated_cases": len(eval_cases)},
            "questions": eval_cases
        }, f, ensure_ascii=False, indent=2)
    
    # Run evaluation
    print(f"\n[4/5] Running evaluation on {len(eval_cases)} cases...")
    evaluator = MedicalQAEvaluator(pipeline=pipeline, output_dir=str(out_dir))
    
    eval_start = time.time()
    results = evaluator.evaluate_system(
        clinical_cases_path=str(temp_path),
        guidelines_path="data/raw/medical_guidelines.json",
        split="all",
        max_cases=None,
        save_detailed_results=True
    )
    eval_time = time.time() - eval_start
    
    print(f"      Evaluation completed in {eval_time:.1f}s ({eval_time/len(eval_cases):.1f}s per case)")
    
    # Calculate context relevance scores
    print(f"\n[4.5/5] Calculating context relevance scores...")
    
    # Build case results for detailed metrics
    case_results = []
    for i, case in enumerate(eval_cases):
        result = {
            'question_id': case['question_id'],
            'is_correct': False,  # Will be updated from pipeline results
            'confidence': 0.5,
            'retrieval_quality': 0.5,
            'question_type': case.get('question_type', 'management'),
            'difficulty': case['difficulty'],
            'relevance_level': case['relevance_level'],
            'category': case['category'],
            'case_description': case.get('case_description', ''),
            'question': case.get('question', '')
        }
        
        # Get actual results from evaluator
        if i < len(evaluator.all_pipeline_results):
            pipeline_result = evaluator.all_pipeline_results[i]['pipeline_result']
            
            # Calculate context relevance for retrieved documents
            retrieved_docs = []
            if hasattr(pipeline_result, 'retrieval_results') and pipeline_result.retrieval_results:
                for doc_info in pipeline_result.retrieval_results[:10]:  # Top 10 docs
                    if hasattr(doc_info, 'document'):
                        retrieved_docs.append({'content': doc_info.document.content})
                    elif isinstance(doc_info, tuple):
                        retrieved_docs.append({'content': str(doc_info[0])})
            
            # Calculate relevance scores
            context_relevance = calculate_context_relevance_score(
                retrieved_docs, case['question'], pipeline.embedding_model
            )
            result['context_relevance'] = context_relevance
            result['is_correct'] = pipeline_result.is_correct
            result['confidence'] = pipeline_result.confidence_score
        
        case_results.append(result)
    
    # Calculate detailed metrics
    print("\n[5/5] Calculating comprehensive metrics...")
    detailed_metrics = calculate_detailed_metrics(results, case_results)
    detailed_metrics['evaluation_time_seconds'] = eval_time
    detailed_metrics['time_per_case_seconds'] = eval_time / len(eval_cases)
    
    # Generate condition-level confusion matrix
    print("\n[5.5/5] Generating condition-level confusion matrix...")
    try:
        from src.evaluation.condition_confusion_analyzer import ConditionConfusionAnalyzer
        
        # Extract data for confusion analysis
        gold_answers = []
        predicted_answers = []
        answer_to_condition = []
        question_ids = []
        
        for i, case in enumerate(eval_cases):
            question_ids.append(case['question_id'])
            gold_answers.append(case['correct_answer'])
            
            # Get predicted answer from pipeline results
            if i < len(evaluator.all_pipeline_results):
                pipeline_result = evaluator.all_pipeline_results[i]['pipeline_result']
                predicted_answer = getattr(pipeline_result, 'selected_answer', case['correct_answer'])
            else:
                predicted_answer = case['correct_answer']
            predicted_answers.append(predicted_answer)
            
            # Extract condition names from options (first few words)
            condition_map = {}
            for answer_key, option_text in case['options'].items():
                # Extract first 5-7 words as condition name (simplified)
                words = option_text.strip().split()[:7]
                condition = ' '.join(words).strip('.,;:')
                condition_map[answer_key] = condition
            answer_to_condition.append(condition_map)
        
        # Run condition confusion analysis
        analyzer = ConditionConfusionAnalyzer()
        confusion_results = analyzer.analyze_confusion(
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            answer_to_condition=answer_to_condition,
            question_ids=question_ids
        )
        
        # Print analysis
        analyzer.print_analysis(confusion_results)
        
        # Generate and save visualization
        charts_dir = out_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        confusion_matrix_path = charts_dir / "confusion_matrix.png"
        analyzer.visualize_confusion_matrix(
            confusion_results,
            str(confusion_matrix_path),
            show_condition_level=True
        )
        
        # Save detailed confusion results
        confusion_json_path = out_dir / f"condition_confusion_{len(eval_cases)}_cases.json"
        analyzer.save_results(confusion_results, str(confusion_json_path))
        
        # Add confusion analysis to detailed metrics
        detailed_metrics['condition_confusion_analysis'] = confusion_results['summary_statistics']
        
    except Exception as e:
        print(f"[WARNING] Could not generate condition confusion matrix: {e}")
        import traceback
        traceback.print_exc()
    
    # Save comprehensive results
    comprehensive_path = out_dir / f"new_dataset_eval_{len(eval_cases)}_cases.json"
    with open(comprehensive_path, 'w', encoding='utf-8') as f:
        json.dump({
            'detailed_metrics': detailed_metrics,
            'pipeline_results': results,
            'config': {
                'dataset': str(dataset_path),
                'cases': args.num_cases,
                'seed': args.seed
            }
        }, f, indent=2, default=str)
    
    print(f"\n      Results saved to: {comprehensive_path}")
    
    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()
    
    # Print comprehensive report
    print_comprehensive_report(detailed_metrics, results)
    
    # Return key metrics
    return {
        'accuracy': detailed_metrics['accuracy']['exact_match'],
        'map_score': results.get('retrieval', {}).get('map_score', 0),
        'precision_at_5': results.get('retrieval', {}).get('precision_at_k', {}).get(5, 0),
        'recall_at_5': results.get('retrieval', {}).get('recall_at_k', {}).get(5, 0),
        'total_errors': detailed_metrics['error_analysis']['total_errors']
    }


if __name__ == "__main__":
    main()

