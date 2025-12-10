"""
Metrics Calculator for Medical QA Evaluation

This module implements all evaluation metrics for:
- Retrieval evaluation (Precision@k, Recall@k, MAP, MRR)
- Reasoning evaluation (Accuracy, Brier Score, ECE)
- Medical-specific metrics (Concept coverage, Safety validation)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    precision_at_k: Dict[int, float]  # k -> precision
    recall_at_k: Dict[int, float]  # k -> recall
    map_score: float  # Mean Average Precision
    mrr: float  # Mean Reciprocal Rank
    context_relevance_scores: List[float]  # Per-query relevance scores
    medical_concept_coverage: float  # Percentage of concepts retrieved
    guideline_coverage: float  # Percentage of relevant guidelines retrieved


@dataclass
class ReasoningMetrics:
    """Reasoning evaluation metrics."""
    exact_match_accuracy: float
    semantic_accuracy: float
    partial_credit_accuracy: float
    brier_score: float
    expected_calibration_error: float
    reasoning_chain_completeness: float
    evidence_utilization_rate: float
    confidence_distribution: Dict[str, int]  # Confidence bins -> count
    hallucination_rate: float
    cannot_answer_misuse_rate: float
    method_accuracy: Dict[str, float]  # e.g., {"CoT": acc, "ToT": acc, "Structured": acc}
    cot_tot_delta: float  # ToT acc - CoT acc
    verifier_pass_rate: float  # placeholder (0 if not available)


@dataclass
class ClassificationMetrics:
    """Per-option classification metrics."""
    per_option: Dict[str, Dict[str, float]]  # option -> {precision, recall, f1, support}
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    confusion_matrix: Dict[str, Dict[str, int]]  # true -> {pred -> count}
    option_distribution: Dict[str, int]
    fp_fn_breakdown: Dict[str, Dict[str, int]]  # option -> {fp, fn, tp, tn}


@dataclass
class MedicalSafetyMetrics:
    """Medical safety-specific metrics."""
    dangerous_error_count: int
    contraindication_check_accuracy: float
    urgency_recognition_accuracy: float
    safety_score: float  # Overall safety (0-1)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for medical QA evaluation.
    
    Implements all Tier 1 (Retrieval) and Tier 2 (Reasoning) metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    # ==================== RETRIEVAL METRICS ====================
    
    def calculate_precision_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str],
        k: int
    ) -> float:
        """
        Calculate Precision@k.
        
        Args:
            retrieved_docs: List of retrieved document IDs (ordered)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank
            
        Returns:
            Precision@k score (0-1)
        """
        if k == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        if not top_k:
            return 0.0
        
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_docs)
        return relevant_retrieved / len(top_k)
    
    def calculate_recall_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str],
        k: int
    ) -> float:
        """
        Calculate Recall@k.
        
        Args:
            retrieved_docs: List of retrieved document IDs (ordered)
            relevant_docs: Set of relevant document IDs
            k: Cutoff rank
            
        Returns:
            Recall@k score (0-1)
        """
        if not relevant_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_docs)
        return relevant_retrieved / len(relevant_docs)
    
    def calculate_average_precision(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str]
    ) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            retrieved_docs: List of retrieved document IDs (ordered)
            relevant_docs: Set of relevant document IDs
            
        Returns:
            Average Precision score (0-1)
        """
        if not relevant_docs:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / len(relevant_docs)

    def calculate_mrr(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str]
    ) -> float:
        """Mean Reciprocal Rank for a single query."""
        for idx, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / idx
        return 0.0
    
    def calculate_context_relevance_score(
        self,
        retrieved_docs: List[Dict],  # Documents with metadata
        question: str,
        expected_guideline_id: Optional[str] = None
    ) -> List[float]:
        """
        Calculate context relevance scores (0-2 scale).
        
        Returns:
            List of relevance scores per retrieved document
            2: Highly relevant, 1: Somewhat relevant, 0: Not relevant
        """
        scores = []
        
        for doc in retrieved_docs:
            score = 0.0
            
            # Check if it's the expected guideline
            if expected_guideline_id:
                doc_guideline_id = doc.get('metadata', {}).get('guideline_id', '')
                if doc_guideline_id == expected_guideline_id:
                    score = 2.0
                    scores.append(score)
                    continue
            
            # Check category match
            doc_category = doc.get('metadata', {}).get('category', '').lower()
            question_lower = question.lower()
            
            # Medical category keywords
            category_keywords = {
                'cardiology': ['heart', 'cardiac', 'chest pain', 'mi', 'stem', 'troponin'],
                'gastroenterology': ['abdomen', 'liver', 'gi', 'gastro', 'hepatic'],
                'neurology': ['brain', 'neurological', 'seizure', 'stroke', 'cns'],
                'pulmonology': ['lung', 'respiratory', 'breathing', 'pneumonia'],
                'nephrology': ['kidney', 'renal', 'nephro'],
                'endocrinology': ['diabetes', 'glucose', 'insulin', 'thyroid'],
            }
            
            # Check for category match
            for category, keywords in category_keywords.items():
                if category in doc_category:
                    if any(kw in question_lower for kw in keywords):
                        score = 2.0
                        break
                    else:
                        score = max(score, 1.0)
            
            # If no category match, check content similarity
            if score == 0.0:
                doc_content = doc.get('content', '').lower()
                # Simple keyword overlap
                question_words = set(question_lower.split())
                doc_words = set(doc_content.split())
                overlap = len(question_words & doc_words)
                if overlap > 3:
                    score = 1.0
            
            scores.append(score)
        
        return scores
    
    def calculate_medical_concept_coverage(
        self,
        retrieved_docs: List[Dict],
        required_concepts: Set[str]
    ) -> float:
        """
        Calculate medical concept coverage.
        
        Args:
            retrieved_docs: List of retrieved documents
            required_concepts: Set of required medical concepts
            
        Returns:
            Coverage percentage (0-1)
        """
        if not required_concepts:
            return 1.0
        
        # Extract all text from retrieved documents
        all_text = ' '.join([
            doc.get('content', '') + ' ' + doc.get('metadata', {}).get('title', '')
            for doc in retrieved_docs
        ]).lower()
        
        # Check which concepts are present
        found_concepts = sum(
            1 for concept in required_concepts
            if concept.lower() in all_text
        )
        
        return found_concepts / len(required_concepts) if required_concepts else 0.0
    
    def evaluate_retrieval(
        self,
        all_queries: List[Dict]
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance across all queries.
        
        Args:
            all_queries: List of query results with:
                - retrieved_docs: List of (doc_id, score) or doc dicts
                - relevant_docs: Set of relevant document IDs
                - question: Question text
                - expected_guideline_id: Expected guideline ID
                - required_concepts: Set of required medical concepts
                
        Returns:
            RetrievalMetrics object
        """
        k_values = [1, 3, 5, 10]
        precision_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        ap_scores = []
        mrr_scores = []
        relevance_scores = []
        concept_coverage_scores = []
        guideline_coverage_scores = []
        
        # DEBUG: Print number of queries and first query details
        print(f"\n[DEBUG] Evaluating retrieval for {len(all_queries)} queries")
        if all_queries:
            first_q = all_queries[0]
            print(f"[DEBUG] First query:")
            print(f"  - relevant_docs type: {type(first_q.get('relevant_docs'))}")
            print(f"  - relevant_docs value: {first_q.get('relevant_docs')}")
            print(f"  - expected_guideline_id: {first_q.get('expected_guideline_id')}")
            print(f"  - num retrieved_docs: {len(first_q.get('retrieved_docs', []))}")
        
        for query_result in all_queries:
            retrieved_docs = query_result.get('retrieved_docs', [])
            relevant_docs = query_result.get('relevant_docs', set())
            question = query_result.get('question', '')
            expected_guideline_id = query_result.get('expected_guideline_id')
            required_concepts = query_result.get('required_concepts', set())
            
            # Convert retrieved_docs to list of IDs if needed
            # For recall calculation, we match on guideline ID basis (any chunk from relevant guideline counts)
            if retrieved_docs and isinstance(retrieved_docs[0], dict):
                doc_ids = []
                guideline_ids = []  # Track guideline IDs separately for matching
                for doc in retrieved_docs:
                    guideline_id = doc.get('metadata', {}).get('guideline_id', '')
                    chunk_index = doc.get('metadata', {}).get('chunk_index', '')
                    
                    # Add chunk-specific ID for precision calculation
                    chunk_id = guideline_id + '_' + str(chunk_index)
                    if chunk_id not in doc_ids:
                        doc_ids.append(chunk_id)
                    
                    # Track guideline IDs for recall matching
                    if guideline_id and guideline_id not in guideline_ids:
                        guideline_ids.append(guideline_id)
                        # Also add to doc_ids for matching
                        doc_ids.append(guideline_id)
            else:
                doc_ids = [str(doc) for doc in retrieved_docs]
            
            # DEBUG: Print doc_ids for first query
            if len(ap_scores) == 0 and relevant_docs:
                print(f"  Doc IDs extracted: {doc_ids[:5]}")  # First 5
            
            # Normalize relevant_docs to handle both formats
            normalized_relevant = set()
            for rel_id in relevant_docs:
                normalized_relevant.add(str(rel_id))
                # Also add chunk variants if it's a guideline ID
                # NOTE: Don't add all chunk indices - only match actual chunks that exist
                # The retrieval will return actual chunks, so we should match on guideline ID basis
                if isinstance(rel_id, str) and rel_id.startswith('GL_'):
                    # Add base guideline ID (any chunk from this guideline is relevant)
                    normalized_relevant.add(rel_id)
                    # Don't add all chunk indices - this inflates the denominator
                    # Instead, we'll match on guideline ID basis in the calculation
            
            # Calculate precision and recall at different k
            for k in k_values:
                prec = self.calculate_precision_at_k(doc_ids, normalized_relevant, k)
                rec = self.calculate_recall_at_k(doc_ids, normalized_relevant, k)
                precision_scores[k].append(prec)
                recall_scores[k].append(rec)
            
            # Calculate Average Precision
            ap = self.calculate_average_precision(doc_ids, normalized_relevant)
            ap_scores.append(ap)

            # Calculate MRR
            mrr_scores.append(self.calculate_mrr(doc_ids, normalized_relevant))
            
            # Context relevance
            if retrieved_docs and isinstance(retrieved_docs[0], dict):
                rel_scores = self.calculate_context_relevance_score(
                    retrieved_docs, question, expected_guideline_id
                )
                relevance_scores.extend(rel_scores)
            
            # Concept coverage
            if required_concepts:
                coverage = self.calculate_medical_concept_coverage(
                    retrieved_docs if isinstance(retrieved_docs[0], dict) else [],
                    required_concepts
                )
                concept_coverage_scores.append(coverage)
            
            # Guideline coverage
            if expected_guideline_id:
                found_guideline = any(
                    doc.get('metadata', {}).get('guideline_id') == expected_guideline_id
                    for doc in (retrieved_docs if retrieved_docs and isinstance(retrieved_docs[0], dict) else [])
                )
                guideline_coverage_scores.append(1.0 if found_guideline else 0.0)
        
        # Aggregate results
        return RetrievalMetrics(
            precision_at_k={k: np.mean(scores) for k, scores in precision_scores.items()},
            recall_at_k={k: np.mean(scores) for k, scores in recall_scores.items()},
            map_score=np.mean(ap_scores) if ap_scores else 0.0,
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            context_relevance_scores=relevance_scores,
            medical_concept_coverage=np.mean(concept_coverage_scores) if concept_coverage_scores else 0.0,
            guideline_coverage=np.mean(guideline_coverage_scores) if guideline_coverage_scores else 0.0
        )
    
    # ==================== REASONING METRICS ====================
    
    def calculate_exact_match_accuracy(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """Calculate exact match accuracy."""
        if not predictions or len(predictions) != len(ground_truth):
            return 0.0
        
        correct = sum(
            1 for pred, gt in zip(predictions, ground_truth)
            if pred.upper().strip() == gt.upper().strip()
        )
        return correct / len(predictions)
    
    def calculate_semantic_accuracy(
        self,
        predictions: List[str],
        ground_truth: List[str],
        option_texts: List[Dict[str, str]]  # List of {A: "text", B: "text", ...}
    ) -> float:
        """
        Calculate semantic accuracy using option text similarity.
        
        For now, uses simple text matching. Could be enhanced with embeddings.
        """
        if not predictions or len(predictions) != len(ground_truth):
            return 0.0
        
        correct = 0
        for pred, gt, options in zip(predictions, ground_truth, option_texts):
            pred_text = options.get(pred.upper().strip(), '').lower()
            gt_text = options.get(gt.upper().strip(), '').lower()
            
            # Simple word overlap
            pred_words = set(pred_text.split())
            gt_words = set(gt_text.split())
            overlap = len(pred_words & gt_words)
            total_words = len(pred_words | gt_words)
            
            if total_words > 0:
                similarity = overlap / total_words
                if similarity > 0.7:  # Threshold for semantic match
                    correct += 1
            elif pred.upper().strip() == gt.upper().strip():
                correct += 1
        
        return correct / len(predictions) if predictions else 0.0
    
    def calculate_brier_score(
        self,
        confidences: List[float],
        correct: List[bool]
    ) -> float:
        """
        Calculate Brier Score.
        
        Brier Score = 1/N * Σ(confidence_i - correctness_i)²
        Lower is better (perfect = 0)
        """
        if not confidences or len(confidences) != len(correct):
            return 1.0  # Worst possible score
        
        correctness = [1.0 if c else 0.0 for c in correct]
        squared_errors = [(conf - corr) ** 2 for conf, corr in zip(confidences, correctness)]
        return np.mean(squared_errors)
    
    def calculate_expected_calibration_error(
        self,
        confidences: List[float],
        correct: List[bool],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Bins predictions by confidence and compares average confidence
        vs actual accuracy in each bin.
        """
        if not confidences or len(confidences) != len(correct):
            return 1.0
        
        correctness = [1.0 if c else 0.0 for c in correct]
        
        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = [
                (conf, corr)
                for conf, corr in zip(confidences, correctness)
                if bin_lower <= conf < bin_upper
            ]
            
            if not in_bin:
                continue
            
            bin_confidences, bin_correctness = zip(*in_bin)
            bin_size = len(in_bin)
            bin_accuracy = np.mean(bin_correctness)
            bin_confidence = np.mean(bin_confidences)
            
            ece += (bin_size / len(confidences)) * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def calculate_reasoning_chain_completeness(
        self,
        reasoning_steps: List[List[Dict]]  # List of reasoning steps per query
    ) -> float:
        """
        Calculate reasoning chain completeness.
        
        Checks if reasoning chains have sufficient steps and logical flow.
        """
        if not reasoning_steps:
            return 0.0
        
        completeness_scores = []
        for steps in reasoning_steps:
            # Minimum 3 steps expected for good reasoning
            if len(steps) >= 3:
                completeness_scores.append(1.0)
            elif len(steps) >= 2:
                completeness_scores.append(0.7)
            elif len(steps) >= 1:
                completeness_scores.append(0.4)
            else:
                completeness_scores.append(0.0)
        
        return np.mean(completeness_scores) if completeness_scores else 0.0
    
    def evaluate_reasoning(
        self,
        all_results: List[Dict]
    ) -> (ReasoningMetrics, ClassificationMetrics):
        """
        Evaluate reasoning performance and classification metrics.
        """
        predictions = []
        ground_truth = []
        confidences = []
        correct = []
        option_texts = []
        reasoning_steps_list = []
        has_evidence = []
        evidence_counts = []
        reasoning_methods = []
        
        for result in all_results:
            predictions.append(result.get('selected_answer', ''))
            ground_truth.append(result.get('correct_answer', ''))
            confidences.append(result.get('confidence_score', 0.0))
            correct.append(result.get('is_correct', False))
            option_texts.append(result.get('options', {}))
            reasoning_steps_list.append(result.get('reasoning_steps', []))
            # Count evidence from supporting_evidence or supporting_guidelines
            support_evs = result.get('supporting_evidence', [])
            support_guides = result.get('supporting_guidelines', [])
            
            has_evidence.append((len(support_evs) > 0) or (len(support_guides) > 0))
            # FIXED: Count both supporting_evidence and supporting_guidelines
            evidence_counts.append(len(support_evs) + len(support_guides))
            reasoning_methods.append(result.get('reasoning_method', 'unknown'))
        
        # Classification metrics
        labels = sorted({lbl for opts in option_texts for lbl in opts.keys()})
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)} or {"UNK": 0}
        num_labels = len(label_to_idx)
        cm = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
        option_distribution = defaultdict(int)
        
        for gt, pred in zip(ground_truth, predictions):
            gi = label_to_idx.get(gt, 0)
            pi = label_to_idx.get(pred, gi)
            cm[gi][pi] += 1
            option_distribution[pred] += 1
        
        per_option = {}
        fp_fn_breakdown = {}
        recalls = []
        precisions = []
        supports = []
        for lbl, idx in label_to_idx.items():
            tp = cm[idx][idx]
            fp = sum(cm[r][idx] for r in range(num_labels) if r != idx)
            fn = sum(cm[idx][c] for c in range(num_labels) if c != idx)
            tn = sum(cm[r][c] for r in range(num_labels) for c in range(num_labels)) - (tp+fp+fn)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            per_option[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": tp+fn}
            fp_fn_breakdown[lbl] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            recalls.append(rec)
            precisions.append(prec)
            supports.append(tp+fn)
        
        macro_precision = float(np.mean(precisions)) if precisions else 0.0
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean([per_option[l]["f1"] for l in per_option])) if per_option else 0.0
        total_support = sum(supports) if supports else 0
        weighted_f1 = float(np.sum([per_option[l]["f1"] * per_option[l]["support"] for l in per_option]) / total_support) if total_support else 0.0
        balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
        labels_list = list(label_to_idx.keys())
        confusion_matrix = {
            lbl: {lbl2: cm[label_to_idx[lbl]][label_to_idx[lbl2]] for lbl2 in labels_list}
            for lbl in labels_list
        }
        
        classification_metrics = ClassificationMetrics(
            per_option=per_option,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            balanced_accuracy=balanced_accuracy,
            confusion_matrix=confusion_matrix,
            option_distribution=dict(option_distribution),
            fp_fn_breakdown=fp_fn_breakdown
        )
        
        # Reasoning metrics
        exact_match = self.calculate_exact_match_accuracy(predictions, ground_truth)
        semantic_match = self.calculate_semantic_accuracy(predictions, ground_truth, option_texts)
        
        partial_correct = sum(
            1 for conf, corr in zip(confidences, correct)
            if corr and conf > 0.7
        )
        partial_accuracy = partial_correct / len(correct) if correct else 0.0
        
        brier = self.calculate_brier_score(confidences, correct)
        ece = self.calculate_expected_calibration_error(confidences, correct)
        
        chain_completeness = self.calculate_reasoning_chain_completeness(reasoning_steps_list)
        evidence_utilization = sum(has_evidence) / len(has_evidence) if has_evidence else 0.0
        # Reward richer evidence: scale by average evidence count (capped)
        if evidence_counts:
            avg_evidence = np.mean(evidence_counts)
            evidence_utilization *= min(1.0, (avg_evidence / 2.0))
        
        confidence_bins = defaultdict(int)
        for conf in confidences:
            bin_idx = int(conf * 10)
            confidence_bins[f"{bin_idx*10}-{(bin_idx+1)*10}%"] += 1
        
        answered = sum(1 for p in predictions if "cannot answer" not in p.lower())
        hallucinations = sum(
            1 for p, ev in zip(predictions, has_evidence)
            if ("cannot answer" not in p.lower()) and (not ev)
        )
        hallucination_rate = (hallucinations / answered) if answered else 0.0
        
        cannot_misuse = sum(
            1 for p, gt in zip(predictions, ground_truth)
            if ("cannot answer" in p.lower()) and gt and ("cannot answer" not in str(gt).lower())
        )
        cannot_misuse_rate = cannot_misuse / len(predictions) if predictions else 0.0
        
        method_groups = defaultdict(list)
        for m, corr in zip(reasoning_methods, correct):
            method_groups[m].append(corr)
        method_acc = {m: float(np.mean(vals)) if vals else 0.0 for m, vals in method_groups.items()}
        cot_acc = method_acc.get("CoT", 0.0)
        tot_acc = method_acc.get("ToT", 0.0)
        cot_tot_delta = tot_acc - cot_acc
        
        reasoning_metrics = ReasoningMetrics(
            exact_match_accuracy=exact_match,
            semantic_accuracy=semantic_match,
            partial_credit_accuracy=partial_accuracy,
            brier_score=brier,
            expected_calibration_error=ece,
            reasoning_chain_completeness=chain_completeness,
            evidence_utilization_rate=evidence_utilization,
            confidence_distribution=dict(confidence_bins),
            hallucination_rate=hallucination_rate,
            cannot_answer_misuse_rate=cannot_misuse_rate,
            method_accuracy=method_acc,
            cot_tot_delta=cot_tot_delta,
            verifier_pass_rate=0.0
        )
        
        return reasoning_metrics, classification_metrics
    
    # ==================== MEDICAL SAFETY METRICS ====================
    
    def evaluate_medical_safety(
        self,
        all_results: List[Dict],
        contraindications: Optional[List[Dict]] = None
    ) -> MedicalSafetyMetrics:
        """
        Evaluate medical safety metrics.
        
        Args:
            all_results: List of result dicts
            contraindications: Optional list of contraindication checks
        """
        dangerous_errors = 0
        contraindication_checks = []
        urgency_recognition = []
        
        for result in all_results:
            # Check for dangerous errors (high confidence wrong answers)
            if (result.get('is_correct') == False and 
                result.get('confidence_score', 0.0) > 0.8):
                dangerous_errors += 1
            
            # Check contraindication awareness (if available)
            if contraindications:
                # This would require more detailed analysis
                # For now, assume system checks contraindications if reasoning mentions them
                reasoning_text = ' '.join([
                    step.get('description', '') + ' ' + step.get('reasoning', '')
                    for step in result.get('reasoning_steps', [])
                ]).lower()
                
                has_contraindication_check = any(
                    'contraindication' in reasoning_text or
                    'contraindicated' in reasoning_text or
                    'avoid' in reasoning_text
                    for _ in [True]
                )
                contraindication_checks.append(has_contraindication_check)
        
        contraindication_accuracy = (
            sum(contraindication_checks) / len(contraindication_checks)
            if contraindication_checks else 0.0
        )
        
        # Safety score: lower dangerous errors = higher safety
        total_questions = len(all_results)
        safety_score = 1.0 - (dangerous_errors / total_questions) if total_questions > 0 else 0.0
        
        return MedicalSafetyMetrics(
            dangerous_error_count=dangerous_errors,
            contraindication_check_accuracy=contraindication_accuracy,
            urgency_recognition_accuracy=0.0,  # Would need urgency labels
            safety_score=max(0.0, safety_score)
        )

