"""
Error Analysis and Performance Segmentation

This module provides:
- Error categorization and root cause analysis
- Confusion matrix for medical conditions
- Performance segmentation by various dimensions
- Pitfall identification and solution tracking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import json


@dataclass
class ErrorCategory:
    """Categorized error with analysis."""
    error_type: str  # 'retrieval', 'reasoning', 'knowledge', 'system'
    description: str
    count: int
    examples: List[Dict]
    root_causes: List[str]
    proposed_solutions: List[str]


@dataclass
class ConfusionMatrixEntry:
    """Entry in confusion matrix."""
    predicted_category: str
    actual_category: str
    count: int
    question_ids: List[str]


class ErrorAnalyzer:
    """
    Analyze errors and failure modes in medical QA system.
    """
    
    def __init__(self):
        """Initialize error analyzer."""
        self.error_categories = {}
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        self.confusion_details = defaultdict(lambda: defaultdict(list))
    
    def categorize_error(
        self,
        result: Dict,
        ground_truth: Dict
    ) -> str:
        """
        Categorize an error into one of four types.
        
        Returns:
            Error category: 'retrieval', 'reasoning', 'knowledge', 'system', or 'none'
        """
        if result.get('is_correct', True):
            return 'none'
        
        # Check if retrieval failed
        retrieved_docs = result.get('retrieval_results', [])
        expected_guideline = ground_truth.get('expected_guideline_id', '')
        
        retrieval_failed = True
        if expected_guideline:
            for doc_result in retrieved_docs:
                doc = doc_result.get('document', {})
                doc_metadata = doc.get('metadata', {})
                if doc_metadata.get('guideline_id') == expected_guideline:
                    retrieval_failed = False
                    break
        
        if retrieval_failed and expected_guideline:
            return 'retrieval'
        
        # Check if reasoning failed (retrieval OK but wrong answer)
        if retrieved_docs and not retrieval_failed:
            # Check if reasoning used the retrieved information
            reasoning_text = ' '.join([
                step.get('description', '') + ' ' + step.get('reasoning', '')
                for step in result.get('reasoning', {}).get('reasoning_steps', [])
            ]).lower()
            
            # Check if relevant content was mentioned
            relevant_content_found = False
            for doc_result in retrieved_docs[:3]:  # Check top 3
                doc = doc_result.get('document', {})
                doc_content = doc.get('content', '').lower()
                # Simple check: if key terms from doc appear in reasoning
                doc_keywords = set(doc_content.split()[:20])  # First 20 words
                reasoning_words = set(reasoning_text.split())
                if len(doc_keywords & reasoning_words) > 3:
                    relevant_content_found = True
                    break
            
            if not relevant_content_found:
                return 'reasoning'
            else:
                return 'knowledge'  # Retrieved and used, but wrong conclusion
        
        # System errors (technical failures)
        if not retrieved_docs:
            return 'system'
        
        return 'reasoning'  # Default to reasoning error
    
    def analyze_errors(
        self,
        all_results: List[Dict],
        all_ground_truth: List[Dict]
    ) -> Dict[str, ErrorCategory]:
        """
        Analyze all errors and categorize them.
        
        Args:
            all_results: List of pipeline results
            all_ground_truth: List of ground truth data
            
        Returns:
            Dictionary of error categories with analysis
        """
        error_counts = defaultdict(int)
        error_examples = defaultdict(list)
        
        # Categorize each error
        for result, gt in zip(all_results, all_ground_truth):
            error_type = self.categorize_error(result, gt)
            
            if error_type != 'none':
                error_counts[error_type] += 1
                
                # Store example (limit to 5 per category)
                if len(error_examples[error_type]) < 5:
                    error_examples[error_type].append({
                        'question_id': result.get('question_id', ''),
                        'question': result.get('question', ''),
                        'selected_answer': result.get('selected_answer', ''),
                        'correct_answer': result.get('correct_answer', ''),
                        'confidence': result.get('confidence_score', 0.0),
                        'error_type': error_type
                    })
        
        # Build error categories with analysis
        error_categories = {}
        
        for error_type, count in error_counts.items():
            root_causes = self._identify_root_causes(error_type, all_results, all_ground_truth)
            solutions = self._propose_solutions(error_type, root_causes)
            
            error_categories[error_type] = ErrorCategory(
                error_type=error_type,
                description=self._get_error_description(error_type),
                count=count,
                examples=error_examples[error_type],
                root_causes=root_causes,
                proposed_solutions=solutions
            )
        
        return error_categories
    
    def _get_error_description(self, error_type: str) -> str:
        """Get human-readable error description."""
        descriptions = {
            'retrieval': 'System failed to retrieve relevant medical information',
            'reasoning': 'System retrieved relevant info but made incorrect reasoning',
            'knowledge': 'System has incorrect medical knowledge or interpretation',
            'system': 'Technical system failure (no retrieval, crashes, etc.)'
        }
        return descriptions.get(error_type, 'Unknown error type')
    
    def _identify_root_causes(
        self,
        error_type: str,
        all_results: List[Dict],
        all_ground_truth: List[Dict]
    ) -> List[str]:
        """Identify root causes for an error type."""
        root_causes = []
        
        if error_type == 'retrieval':
            # Analyze retrieval failures
            failed_retrievals = [
                (r, gt) for r, gt in zip(all_results, all_ground_truth)
                if self.categorize_error(r, gt) == 'retrieval'
            ]
            
            # Check common patterns
            vague_queries = sum(
                1 for r, _ in failed_retrievals
                if len(r.get('question', '').split()) < 10
            )
            if vague_queries > len(failed_retrievals) * 0.3:
                root_causes.append('Vague or underspecified queries')
            
            category_mismatches = sum(
                1 for _, gt in failed_retrievals
                if gt.get('category', '') not in ['Cardiology', 'Gastroenterology', 'Neurology']
            )
            if category_mismatches > len(failed_retrievals) * 0.4:
                root_causes.append('Limited coverage in certain medical specialties')
            
            root_causes.append('Semantic search may not capture medical terminology nuances')
            root_causes.append('BM25 keyword matching may miss synonyms')
        
        elif error_type == 'reasoning':
            root_causes.append('Insufficient chain-of-thought reasoning steps')
            root_causes.append('Failure to properly weight evidence from multiple sources')
            root_causes.append('Over-reliance on single retrieved document')
            root_causes.append('Missing critical symptom analysis')
        
        elif error_type == 'knowledge':
            root_causes.append('Incorrect interpretation of medical guidelines')
            root_causes.append('Missing context about patient-specific factors')
            root_causes.append('Failure to consider contraindications')
            root_causes.append('Incorrect application of treatment protocols')
        
        elif error_type == 'system':
            root_causes.append('Technical failures in retrieval pipeline')
            root_causes.append('Missing or corrupted index files')
            root_causes.append('Timeout or resource constraints')
        
        return root_causes
    
    def _propose_solutions(
        self,
        error_type: str,
        root_causes: List[str]
    ) -> List[str]:
        """Propose solutions based on error type and root causes."""
        solutions = []
        
        if error_type == 'retrieval':
            solutions.append('Enhance query understanding with medical entity recognition')
            solutions.append('Expand medical synonym dictionary for BM25')
            solutions.append('Improve FAISS embedding model with medical domain fine-tuning')
            solutions.append('Add query expansion with medical terminology')
            solutions.append('Implement hybrid retrieval with more weight on medical keywords')
        
        elif error_type == 'reasoning':
            solutions.append('Increase minimum reasoning steps requirement')
            solutions.append('Implement evidence aggregation with confidence weighting')
            solutions.append('Add medical logic rules for common scenarios')
            solutions.append('Improve chain-of-thought prompting with medical examples')
            solutions.append('Add validation step to check reasoning completeness')
        
        elif error_type == 'knowledge':
            solutions.append('Expand medical knowledge base with more guidelines')
            solutions.append('Add medical expert validation of reasoning chains')
            solutions.append('Implement medical safety checks (contraindications, interactions)')
            solutions.append('Add context-aware interpretation of guidelines')
            solutions.append('Create medical concept mapping for better understanding')
        
        elif error_type == 'system':
            solutions.append('Add error handling and fallback mechanisms')
            solutions.append('Implement retry logic for failed retrievals')
            solutions.append('Add system health monitoring')
            solutions.append('Optimize index loading and caching')
        
        return solutions
    
    def build_confusion_matrix(
        self,
        all_results: List[Dict],
        all_ground_truth: List[Dict]
    ) -> Dict[str, Dict[str, int]]:
        """
        Build confusion matrix by medical category.
        
        Args:
            all_results: List of pipeline results
            all_ground_truth: List of ground truth data
            
        Returns:
            Confusion matrix: {actual_category: {predicted_category: count}}
        """
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        confusion_details = defaultdict(lambda: defaultdict(list))
        
        for result, gt in zip(all_results, all_ground_truth):
            # Get actual category
            actual_category = gt.get('category', 'unknown')
            
            # Get predicted category (from query understanding or reasoning)
            predicted_category = 'unknown'
            if 'pipeline_metadata' in result:
                query_meta = result['pipeline_metadata'].get('query_understanding', {})
                predicted_category = query_meta.get('specialty', 'unknown')
            
            # If not found, try to infer from retrieved documents
            if predicted_category == 'unknown':
                retrieved_docs = result.get('retrieval_results', [])
                if retrieved_docs:
                    doc = retrieved_docs[0].get('document', {})
                    predicted_category = doc.get('metadata', {}).get('category', 'unknown')
            
            confusion_matrix[actual_category][predicted_category] += 1
            confusion_details[actual_category][predicted_category].append(
                result.get('question_id', '')
            )
        
        # Convert to regular dict
        matrix = {
            actual: dict(predicted)
            for actual, predicted in confusion_matrix.items()
        }
        
        self.confusion_matrix = matrix
        self.confusion_details = confusion_details
        
        return matrix
    
    def identify_confused_conditions(self, confusion_matrix: Dict) -> List[Tuple[str, str, int]]:
        """
        Identify frequently confused condition pairs.
        
        Returns:
            List of (actual_category, predicted_category, count) tuples
        """
        confused_pairs = []
        
        for actual_cat, predictions in confusion_matrix.items():
            for pred_cat, count in predictions.items():
                if actual_cat != pred_cat and count > 0:
                    confused_pairs.append((actual_cat, pred_cat, count))
        
        # Sort by count (most confused first)
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return confused_pairs
    
    def segment_performance(
        self,
        all_results: List[Dict],
        all_ground_truth: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Segment performance by various dimensions.
        
        Returns:
            Dictionary with performance by:
            - category (medical specialty)
            - question_type (diagnosis, treatment, management)
            - complexity (simple, moderate, complex)
            - relevance_level (high, medium, low)
        """
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from evaluation.ground_truth_processor import GroundTruthProcessor
        processor = GroundTruthProcessor()
        
        segments = {
            'by_category': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'by_question_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'by_complexity': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'by_relevance_level': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'by_confidence_range': defaultdict(lambda: {'correct': 0, 'total': 0})
        }
        
        for result, gt in zip(all_results, all_ground_truth):
            is_correct = result.get('is_correct', False)
            confidence = result.get('confidence_score', 0.0)
            
            # By category
            category = gt.get('category', 'unknown')
            segments['by_category'][category]['total'] += 1
            if is_correct:
                segments['by_category'][category]['correct'] += 1
            
            # By question type
            question_text = result.get('question', '')
            q_type = processor.get_question_type(question_text)
            segments['by_question_type'][q_type]['total'] += 1
            if is_correct:
                segments['by_question_type'][q_type]['correct'] += 1
            
            # By complexity
            question_dict = {
                'difficulty': result.get('difficulty', 'medium'),
                'question': question_text
            }
            complexity = processor.get_complexity_level(question_dict)
            segments['by_complexity'][complexity]['total'] += 1
            if is_correct:
                segments['by_complexity'][complexity]['correct'] += 1
            
            # By relevance level
            rel_level = gt.get('relevance_level', 'unknown')
            segments['by_relevance_level'][rel_level]['total'] += 1
            if is_correct:
                segments['by_relevance_level'][rel_level]['correct'] += 1
            
            # By confidence range
            conf_range = f"{int(confidence * 10) * 10}-{int(confidence * 10) * 10 + 10}%"
            segments['by_confidence_range'][conf_range]['total'] += 1
            if is_correct:
                segments['by_confidence_range'][conf_range]['correct'] += 1
        
        # Calculate accuracies
        performance = {}
        for segment_name, segment_data in segments.items():
            performance[segment_name] = {
                key: {
                    'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
                for key, stats in segment_data.items()
            }
        
        return performance
    
    def identify_pitfalls(
        self,
        all_results: List[Dict],
        all_ground_truth: List[Dict]
    ) -> List[Dict]:
        """
        Identify common pitfalls and failure patterns.
        
        Returns:
            List of pitfall dictionaries with description and examples
        """
        pitfalls = []
        
        # Pitfall 1: High confidence wrong answers
        high_conf_wrong = [
            r for r, gt in zip(all_results, all_ground_truth)
            if not r.get('is_correct', True) and r.get('confidence_score', 0.0) > 0.8
        ]
        if high_conf_wrong:
            pitfalls.append({
                'pitfall': 'Overconfident Wrong Answers',
                'description': 'System shows high confidence (>80%) but gives incorrect answers',
                'count': len(high_conf_wrong),
                'severity': 'high',
                'examples': [
                    {
                        'question_id': r.get('question_id', ''),
                        'confidence': r.get('confidence_score', 0.0),
                        'selected': r.get('selected_answer', ''),
                        'correct': r.get('correct_answer', '')
                    }
                    for r in high_conf_wrong[:3]
                ],
                'solution': 'Implement confidence calibration and add uncertainty estimation'
            })
        
        # Pitfall 2: Retrieval of wrong guideline
        wrong_guideline = []
        for r, gt in zip(all_results, all_ground_truth):
            expected_guideline = gt.get('expected_guideline_id', '')
            if expected_guideline:
                retrieved_guidelines = [
                    doc.get('document', {}).get('metadata', {}).get('guideline_id', '')
                    for doc in r.get('retrieval_results', [])
                ]
                if expected_guideline not in retrieved_guidelines and retrieved_guidelines:
                    wrong_guideline.append(r)
        
        if wrong_guideline:
            pitfalls.append({
                'pitfall': 'Wrong Guideline Retrieved',
                'description': 'System retrieves guidelines that do not match the question',
                'count': len(wrong_guideline),
                'severity': 'high',
                'examples': [
                    {
                        'question_id': r.get('question_id', ''),
                        'expected': gt.get('expected_guideline_id', ''),
                        'retrieved': [
                            doc.get('document', {}).get('metadata', {}).get('guideline_id', '')
                            for doc in r.get('retrieval_results', [])[:3]
                        ]
                    }
                    for r, gt in zip(wrong_guideline[:3], 
                                    [gt for _, gt in zip(wrong_guideline[:3], all_ground_truth)])
                ],
                'solution': 'Improve query understanding and add medical entity recognition'
            })
        
        # Pitfall 3: Missing critical symptoms
        missing_symptoms = []
        for r, gt in zip(all_results, all_ground_truth):
            reasoning_text = ' '.join([
                step.get('description', '') + ' ' + step.get('reasoning', '')
                for step in r.get('reasoning', {}).get('reasoning_steps', [])
            ]).lower()
            
            case_text = r.get('case_description', '').lower()
            # Extract symptoms from case
            symptom_keywords = ['pain', 'fever', 'nausea', 'vomiting', 'jaundice', 'chest pain']
            case_symptoms = [s for s in symptom_keywords if s in case_text]
            
            # Check if reasoning mentions symptoms
            mentioned_symptoms = [s for s in case_symptoms if s in reasoning_text]
            
            if len(case_symptoms) > 0 and len(mentioned_symptoms) < len(case_symptoms) * 0.5:
                missing_symptoms.append(r)
        
        if missing_symptoms:
            pitfalls.append({
                'pitfall': 'Missing Critical Symptoms',
                'description': 'Reasoning fails to consider important symptoms from case description',
                'count': len(missing_symptoms),
                'severity': 'medium',
                'examples': [
                    {
                        'question_id': r.get('question_id', ''),
                        'case_symptoms': case_text.split()[:10]  # Simplified
                    }
                    for r in missing_symptoms[:3]
                ],
                'solution': 'Enhance symptom extraction and ensure all symptoms are considered in reasoning'
            })
        
        # Pitfall 4: Medical terminology misunderstanding
        terminology_errors = []
        for r, gt in zip(all_results, all_ground_truth):
            if not r.get('is_correct', True):
                # Check if reasoning shows misunderstanding of medical terms
                reasoning_text = ' '.join([
                    step.get('reasoning', '')
                    for step in r.get('reasoning', {}).get('reasoning_steps', [])
                ]).lower()
                
                # Common misunderstandings
                misunderstandings = [
                    ('mi', 'myocardial infarction'),
                    ('stem', 'stemi'),
                    ('gi', 'gastrointestinal'),
                ]
                
                for short, full in misunderstandings:
                    if short in reasoning_text and full not in reasoning_text:
                        terminology_errors.append(r)
                        break
        
        if terminology_errors:
            pitfalls.append({
                'pitfall': 'Medical Terminology Misunderstanding',
                'description': 'System fails to properly interpret medical abbreviations or terms',
                'count': len(terminology_errors),
                'severity': 'medium',
                'examples': [
                    {
                        'question_id': r.get('question_id', ''),
                        'reasoning_excerpt': ' '.join([
                            step.get('reasoning', '')
                            for step in r.get('reasoning', {}).get('reasoning_steps', [])
                        ])[:200]
                    }
                    for r in terminology_errors[:3]
                ],
                'solution': 'Add medical terminology expansion and abbreviation resolution'
            })
        
        return pitfalls

