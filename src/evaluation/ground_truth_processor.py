"""
Ground Truth Processor for Medical QA Evaluation

This module processes clinical cases to extract:
- Medical concepts for evaluation
- Relevance judgments for retrieval
- Expected guidelines and evidence
"""

import json
import re
from typing import List, Dict, Set, Optional
from pathlib import Path
from collections import defaultdict


class GroundTruthProcessor:
    """
    Process ground truth data for evaluation.
    
    Extracts medical concepts, creates relevance judgments,
    and prepares evaluation datasets.
    """
    
    def __init__(self):
        """Initialize ground truth processor."""
        self.medical_concept_patterns = self._init_medical_patterns()
    
    def _init_medical_patterns(self) -> Dict[str, List[str]]:
        """Initialize medical concept extraction patterns."""
        return {
            'conditions': [
                r'\b(?:myocardial infarction|MI|STEMI|NSTEMI|ACS)\b',
                r'\b(?:diabetes|DM|type [12] diabetes)\b',
                r'\b(?:pneumonia|pneumonitis)\b',
                r'\b(?:seizure|epilepsy|convulsion)\b',
                r'\b(?:abscess|infection|bacteremia)\b',
            ],
            'medications': [
                r'\b(?:metronidazole|aspirin|insulin|antibiotic)\b',
                r'\b(?:mg|dose|dosage|therapy|treatment)\b',
            ],
            'procedures': [
                r'\b(?:ultrasound|CT|MRI|endoscopy|biopsy)\b',
                r'\b(?:surgery|procedure|intervention)\b',
            ],
            'symptoms': [
                r'\b(?:pain|fever|jaundice|nausea|vomiting)\b',
                r'\b(?:chest pain|abdominal pain|headache)\b',
            ]
        }
    
    def extract_medical_concepts(self, text: str) -> Set[str]:
        """
        Extract medical concepts from text.
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            Set of extracted medical concepts
        """
        concepts = set()
        text_lower = text.lower()
        
        # Extract using patterns
        for category, patterns in self.medical_concept_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                concepts.update(matches)
        
        # Extract medical terms (capitalized words, abbreviations)
        words = text.split()
        for word in words:
            # Medical abbreviations (2-4 uppercase letters)
            if re.match(r'^[A-Z]{2,4}$', word):
                concepts.add(word)
            # Medical terms (often capitalized)
            if word[0].isupper() and len(word) > 4:
                concepts.add(word.lower())
        
        return concepts
    
    def load_clinical_cases(self, file_path: str) -> Dict:
        """
        Load clinical cases from JSON file.
        
        Args:
            file_path: Path to clinical_cases.json
            
        Returns:
            Dictionary with metadata and questions
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def create_relevance_judgments(
        self,
        question: Dict,
        all_guidelines: List[Dict]
    ) -> Dict:
        """
        Create relevance judgments for a question.
        
        Args:
            question: Question dictionary with metadata
            all_guidelines: List of all available guidelines
            
        Returns:
            Dictionary with:
                - relevant_doc_ids: Set of relevant document IDs
                - expected_guideline_id: Expected guideline ID
                - required_concepts: Set of required medical concepts
        """
        # Get expected guideline
        expected_guideline_id = question.get('guideline_id', '')
        source_guideline = question.get('source_guideline', '')
        
        # Find relevant guidelines
        relevant_doc_ids = set()
        
        # Primary guideline (if specified) - THIS IS THE KEY FIX
        if expected_guideline_id:
            relevant_doc_ids.add(expected_guideline_id)
        
        # Fallback: Try to infer guideline ID from source_guideline or category
        # This ensures we have SOME relevant docs even if all_guidelines is empty
        if not relevant_doc_ids:
            # If source guideline contains keywords, create a guideline ID
            if source_guideline:
                # Try to match common patterns
                if 'cardiovascular' in source_guideline.lower() or 'acs' in source_guideline.lower():
                    relevant_doc_ids.add('GL_001')
                elif 'stroke' in source_guideline.lower():
                    relevant_doc_ids.add('GL_002')
                elif 'diabetes' in source_guideline.lower():
                    relevant_doc_ids.add('GL_003')
                elif 'hypertension' in source_guideline.lower():
                    relevant_doc_ids.add('GL_004')
                elif 'asthma' in source_guideline.lower():
                    relevant_doc_ids.add('GL_005')
                elif 'copd' in source_guideline.lower():
                    relevant_doc_ids.add('GL_006')
                elif 'pneumonia' in source_guideline.lower():
                    relevant_doc_ids.add('GL_007')
                elif 'uti' in source_guideline.lower() or 'urinary' in source_guideline.lower():
                    relevant_doc_ids.add('GL_008')
                elif 'sepsis' in source_guideline.lower():
                    relevant_doc_ids.add('GL_009')
                elif 'gastrointestinal' in source_guideline.lower() or 'gi bleed' in source_guideline.lower():
                    relevant_doc_ids.add('GL_010')
                elif 'kidney' in source_guideline.lower() or 'aki' in source_guideline.lower():
                    relevant_doc_ids.add('GL_011')
                elif 'heart failure' in source_guideline.lower():
                    relevant_doc_ids.add('GL_012')
                elif 'atrial' in source_guideline.lower() or 'afib' in source_guideline.lower():
                    relevant_doc_ids.add('GL_013')
                elif 'dvt' in source_guideline.lower() or 'deep vein' in source_guideline.lower():
                    relevant_doc_ids.add('GL_014')
                elif 'pulmonary embolism' in source_guideline.lower() or 'pe' in source_guideline.lower():
                    relevant_doc_ids.add('GL_015')
                elif 'pancreatitis' in source_guideline.lower():
                    relevant_doc_ids.add('GL_016')
                elif 'cirrhosis' in source_guideline.lower() or 'liver' in source_guideline.lower():
                    relevant_doc_ids.add('GL_017')
                elif 'rheumatoid' in source_guideline.lower() or 'arthritis' in source_guideline.lower():
                    relevant_doc_ids.add('GL_018')
                elif 'osteoporosis' in source_guideline.lower():
                    relevant_doc_ids.add('GL_019')
                elif 'depression' in source_guideline.lower():
                    relevant_doc_ids.add('GL_020')
        
        # Match by source guideline name from all_guidelines (if provided)
        if source_guideline and all_guidelines:
            for guideline in all_guidelines:
                if source_guideline.lower() in guideline.get('title', '').lower():
                    relevant_doc_ids.add(guideline.get('guideline_id', ''))
        
        # Match by category from all_guidelines (if provided)
        question_category = question.get('category', '').lower()
        if question_category and all_guidelines:
            for guideline in all_guidelines:
                guideline_category = guideline.get('category', '').lower()
                if guideline_category == question_category:
                    relevant_doc_ids.add(guideline.get('guideline_id', ''))
        
        # Extract required concepts
        # Handle both old format (case_description) and new format (question only)
        case_text = question.get('case_description', '')
        if not case_text:
            case_text = question.get('question', '')
        else:
            case_text += ' ' + question.get('question', '')
        required_concepts = self.extract_medical_concepts(case_text)
        
        return {
            'relevant_doc_ids': relevant_doc_ids,
            'expected_guideline_id': expected_guideline_id,
            'required_concepts': required_concepts,
            'question_category': question.get('category', ''),
            'source_guideline': source_guideline
        }
    
    def process_evaluation_dataset(
        self,
        clinical_cases_path: str,
        guidelines_path: Optional[str] = None,
        split: str = 'all'  # 'all', 'dev', 'test'
    ) -> List[Dict]:
        """
        Process complete evaluation dataset.
        
        Args:
            clinical_cases_path: Path to clinical_cases.json
            guidelines_path: Optional path to guidelines JSON
            split: Dataset split ('all', 'dev', 'test')
            
        Returns:
            List of processed evaluation cases with ground truth
        """
        # Load clinical cases
        cases_data = self.load_clinical_cases(clinical_cases_path)
        questions = cases_data.get('questions', [])
        
        # Filter by split
        if split == 'dev':
            questions = questions[:80]
        elif split == 'test':
            questions = questions[80:]
        
        # Load guidelines if provided
        all_guidelines = []
        if guidelines_path and Path(guidelines_path).exists():
            with open(guidelines_path, 'r', encoding='utf-8') as f:
                all_guidelines = json.load(f)
        
        # Process each question
        processed_cases = []
        for question in questions:
            # Create relevance judgments
            relevance = self.create_relevance_judgments(question, all_guidelines)
            
            # Extract medical concepts
            # Handle both old format (case_description + question) and new format (just question)
            full_question_text = question.get('question', '')
            case_description = question.get('case_description', '')
            
            # For new format: the entire case + question is in 'question' field
            # We need to keep it there for the pipeline
            if not case_description:
                # New format: question contains everything, use empty string for case_description
                case_description = ''
                case_text = full_question_text
            else:
                # Old format: combine case_description and question
                case_text = case_description + ' ' + full_question_text
            
            required_concepts = self.extract_medical_concepts(case_text)
            
            full_text = (
                case_text + ' ' +
                str(question.get('explanation', ''))
            )
            all_concepts = self.extract_medical_concepts(full_text)
            
            # Build evaluation case
            eval_case = {
                'question_id': question.get('question_id', ''),
                'case_description': case_description,  # Empty for new format
                'question': full_question_text,  # Full case + question for new format
                'options': question.get('options', {}),
                'correct_answer': question.get('correct_answer', ''),
                'explanation': question.get('explanation', ''),
                'category': question.get('category', ''),
                'difficulty': question.get('difficulty', 'medium'),
                'relevance_level': question.get('relevance_level', 'high'),
                'source_guideline': question.get('source_guideline', ''),
                'guideline_id': question.get('guideline_id', ''),
                # Ground truth for evaluation
                'ground_truth': {
                    'relevant_doc_ids': relevance['relevant_doc_ids'],
                    'expected_guideline_id': relevance['expected_guideline_id'],
                    'required_concepts': required_concepts,
                    'all_concepts': all_concepts,
                    'category': relevance['question_category']
                }
            }
            
            processed_cases.append(eval_case)
        
        return processed_cases
    
    def get_question_type(self, question_text: str) -> str:
        """
        Classify question type.
        
        Returns:
            Question type: 'diagnosis', 'treatment', 'management', 'other'
        """
        question_lower = question_text.lower()
        
        if any(word in question_lower for word in ['diagnosis', 'diagnose', 'what is', 'identify']):
            return 'diagnosis'
        elif any(word in question_lower for word in ['treatment', 'treat', 'therapy', 'medication', 'drug']):
            return 'treatment'
        elif any(word in question_lower for word in ['manage', 'management', 'monitor', 'follow-up']):
            return 'management'
        else:
            return 'other'
    
    def get_complexity_level(self, question: Dict) -> str:
        """
        Classify question complexity.
        
        Returns:
            Complexity: 'simple', 'moderate', 'complex'
        """
        difficulty = question.get('difficulty', 'medium').lower()
        
        if difficulty in ['easy', 'simple']:
            return 'simple'
        elif difficulty in ['hard', 'complex', 'difficult']:
            return 'complex'
        else:
            return 'moderate'
    
    def create_evaluation_summary(self, processed_cases: List[Dict]) -> Dict:
        """
        Create summary statistics for evaluation dataset.
        
        Args:
            processed_cases: List of processed evaluation cases
            
        Returns:
            Summary dictionary with statistics
        """
        summary = {
            'total_cases': len(processed_cases),
            'categories': defaultdict(int),
            'question_types': defaultdict(int),
            'complexity_levels': defaultdict(int),
            'relevance_levels': defaultdict(int),
            'total_concepts': 0,
            'unique_concepts': set()
        }
        
        for case in processed_cases:
            # Category
            category = case.get('category', 'unknown')
            summary['categories'][category] += 1
            
            # Question type
            q_type = self.get_question_type(case.get('question', ''))
            summary['question_types'][q_type] += 1
            
            # Complexity
            complexity = self.get_complexity_level(case)
            summary['complexity_levels'][complexity] += 1
            
            # Relevance level
            rel_level = case.get('relevance_level', 'unknown')
            summary['relevance_levels'][rel_level] += 1
            
            # Concepts
            concepts = case.get('ground_truth', {}).get('all_concepts', set())
            summary['total_concepts'] += len(concepts)
            summary['unique_concepts'].update(concepts)
        
        # Convert sets to counts
        summary['unique_concept_count'] = len(summary['unique_concepts'])
        summary['unique_concepts'] = list(summary['unique_concepts'])[:50]  # Limit for display
        
        # Convert defaultdicts to dicts
        summary['categories'] = dict(summary['categories'])
        summary['question_types'] = dict(summary['question_types'])
        summary['complexity_levels'] = dict(summary['complexity_levels'])
        summary['relevance_levels'] = dict(summary['relevance_levels'])
        
        return summary

