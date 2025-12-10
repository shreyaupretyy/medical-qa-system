"""
Semantic Evidence Matcher for Medical QA

This module implements semantic similarity-based evidence matching
to improve recognition of medical terms even when exact matches don't exist.

Key Features:
- Semantic similarity for medication names, symptoms, procedures
- Context-aware evidence scoring
- Multi-document evidence aggregation
- Missing concept detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.document_processor import Document
from models.embeddings import EmbeddingModel


@dataclass
class SemanticMatch:
    """Semantic match result with similarity score."""
    term: str
    matched_term: str
    similarity: float
    context: str
    location: str  # 'treatment', 'background', 'contraindication', etc.


class SemanticEvidenceMatcher:
    """
    Match evidence using semantic similarity instead of exact matching.
    
    This addresses the 80% of errors with confidence <0.2 by:
    1. Using semantic similarity for medical terms
    2. Context-aware scoring
    3. Multi-document aggregation
    4. Missing concept detection
    """
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """Initialize semantic evidence matcher."""
        self.embedding_model = embedding_model or EmbeddingModel()
        self._init_medical_synonym_groups()
        self._init_context_keywords()
    
    def _init_medical_synonym_groups(self):
        """Initialize medical synonym groups for semantic matching."""
        self.medication_synonyms = {
            'gentamicin': ['gentamicin', 'gentamycin', 'garamycin'],
            'ceftriaxone': ['ceftriaxone', 'rocephin'],
            'azithromycin': ['azithromycin', 'zithromax'],
            'metronidazole': ['metronidazole', 'flagyl'],
            'ampicillin': ['ampicillin', 'principen'],
            'cefotaxime': ['cefotaxime', 'claforan'],
            'aspirin': ['aspirin', 'asa', 'acetylsalicylic acid'],
            'paracetamol': ['paracetamol', 'acetaminophen', 'tylenol'],
            'morphine': ['morphine', 'morphine sulfate'],
            'insulin': ['insulin', 'regular insulin', 'human insulin'],
        }
        
        self.symptom_synonyms = {
            'chest pain': ['chest pain', 'angina', 'chest discomfort', 'precordial pain'],
            'shortness of breath': ['shortness of breath', 'dyspnea', 'sob', 'breathlessness'],
            'fever': ['fever', 'pyrexia', 'elevated temperature', 'hyperthermia'],
            'headache': ['headache', 'cephalgia', 'head pain'],
            'nausea': ['nausea', 'feeling sick', 'queasiness'],
            'vomiting': ['vomiting', 'emesis', 'throwing up'],
        }
        
        self.procedure_synonyms = {
            'surgery': ['surgery', 'operation', 'surgical procedure', 'operation'],
            'biopsy': ['biopsy', 'tissue sampling', 'specimen collection'],
            'endoscopy': ['endoscopy', 'endoscopic procedure', 'scope'],
        }
    
    def _init_context_keywords(self):
        """Initialize keywords that indicate document context."""
        self.context_keywords = {
            'treatment': ['treatment', 'therapy', 'management', 'intervention', 'medication', 'prescribe', 'administer', 'dose', 'dosage'],
            'contraindication': ['contraindicated', 'avoid', 'not recommended', 'should not', 'do not use', 'contraindication'],
            'indication': ['indicated', 'recommended', 'should be used', 'appropriate', 'indication'],
            'background': ['background', 'introduction', 'overview', 'general'],
            'complication': ['complication', 'adverse', 'side effect', 'risk'],
        }
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        return self.embedding_model.compute_similarity(text1, text2)
    
    def find_semantic_matches(
        self,
        option_text: str,
        document: Document,
        threshold: float = 0.6
    ) -> List[SemanticMatch]:
        """
        Find semantic matches for option text in document.
        
        Args:
            option_text: Answer option text
            document: Retrieved document
            threshold: Minimum similarity threshold
            
        Returns:
            List of SemanticMatch objects
        """
        matches = []
        doc_content = document.content
        doc_lower = doc_content.lower()
        option_lower = option_text.lower()
        
        # Extract medical terms from option
        option_terms = self._extract_medical_terms(option_lower)
        
        # Check each term for semantic matches
        for term in option_terms:
            # First check exact match
            if term in doc_lower:
                context = self._identify_context(doc_content, term)
                matches.append(SemanticMatch(
                    term=term,
                    matched_term=term,
                    similarity=1.0,
                    context=context,
                    location=self._find_term_location(doc_content, term)
                ))
                continue
            
            # Check synonym groups
            matched_synonym = self._check_synonym_groups(term, doc_lower)
            if matched_synonym:
                context = self._identify_context(doc_content, matched_synonym)
                matches.append(SemanticMatch(
                    term=term,
                    matched_term=matched_synonym,
                    similarity=0.9,  # High similarity for synonyms
                    context=context,
                    location=self._find_term_location(doc_content, matched_synonym)
                ))
                continue
            
            # Use semantic similarity for remaining terms
            # Check against document sentences
            sentences = self._split_into_sentences(doc_content)
            for sentence in sentences:
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                similarity = self.compute_semantic_similarity(term, sentence)
                # Day 7: More lenient threshold for low-confidence cases
                if similarity >= threshold or (threshold <= 0.5 and similarity >= threshold - 0.05):
                    context = self._identify_context(doc_content, sentence)
                    matches.append(SemanticMatch(
                        term=term,
                        matched_term=sentence[:100],  # First 100 chars
                        similarity=similarity,
                        context=context,
                        location=self._find_term_location(doc_content, sentence)
                    ))
        
        return matches
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from text."""
        terms = []
        
        # Extract medications
        for med_group in self.medication_synonyms.values():
            for med in med_group:
                if med in text:
                    terms.append(med)
                    break
        
        # Extract symptoms
        for symptom_group in self.symptom_synonyms.values():
            for symptom in symptom_group:
                if symptom in text:
                    terms.append(symptom)
                    break
        
        # Extract procedures
        for proc_group in self.procedure_synonyms.values():
            for proc in proc_group:
                if proc in text:
                    terms.append(proc)
                    break
        
        # Extract doses (e.g., "250mg", "10 days")
        dose_patterns = [
            r'\d+\s*(mg|g|ml|kg|days?|hours?|times?)',
            r'\d+\s*(stat|daily|bid|tid|qid)',
        ]
        for pattern in dose_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend([m[0] if isinstance(m, tuple) else m for m in matches])
        
        return list(set(terms))
    
    def _check_synonym_groups(self, term: str, doc_text: str) -> Optional[str]:
        """Check if term has synonyms in document."""
        # Check medication synonyms
        for med, synonyms in self.medication_synonyms.items():
            if med in term.lower():
                for synonym in synonyms:
                    if synonym in doc_text:
                        return synonym
        
        # Check symptom synonyms
        for symptom, synonyms in self.symptom_synonyms.items():
            if symptom in term.lower():
                for synonym in synonyms:
                    if synonym in doc_text:
                        return synonym
        
        # Check procedure synonyms
        for proc, synonyms in self.procedure_synonyms.items():
            if proc in term.lower():
                for synonym in synonyms:
                    if synonym in doc_text:
                        return synonym
        
        return None
    
    def _identify_context(self, doc_content: str, term_or_sentence: str) -> str:
        """Identify context where term appears (treatment, contraindication, etc.)."""
        doc_lower = doc_content.lower()
        term_lower = term_or_sentence.lower()
        
        # Find position of term
        pos = doc_lower.find(term_lower)
        if pos == -1:
            return 'unknown'
        
        # Check surrounding context (200 chars before and after)
        start = max(0, pos - 200)
        end = min(len(doc_content), pos + len(term_lower) + 200)
        context_window = doc_lower[start:end]
        
        # Check for context keywords
        for context_type, keywords in self.context_keywords.items():
            if any(keyword in context_window for keyword in keywords):
                return context_type
        
        return 'general'
    
    def _find_term_location(self, doc_content: str, term: str) -> str:
        """Find approximate location of term in document."""
        pos = doc_content.lower().find(term.lower())
        if pos == -1:
            return 'unknown'
        
        doc_length = len(doc_content)
        if pos < doc_length * 0.2:
            return 'beginning'
        elif pos < doc_length * 0.6:
            return 'middle'
        else:
            return 'end'
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def score_evidence_with_context(
        self,
        semantic_matches: List[SemanticMatch],
        option_text: str,
        document: Optional[Document] = None
    ) -> Tuple[float, List[Tuple[Document, str, float]]]:
        """
        Score evidence with context awareness.
        
        Returns:
            (total_score, list of (doc, excerpt, score) tuples)
        """
        if not semantic_matches:
            return 0.0, []
        
        scored_evidence = []
        total_score = 0.0
        
        for match in semantic_matches:
            base_score = match.similarity
            
            # Context-based weighting
            if match.context == 'treatment':
                base_score *= 1.3  # 30% boost for treatment section
            elif match.context == 'indication':
                base_score *= 1.2  # 20% boost for indication
            elif match.context == 'contraindication':
                base_score *= 0.3  # Heavy penalty for contraindication
            elif match.context == 'complication':
                base_score *= 0.7  # Slight penalty for complication section
            
            # Specificity boost (exact match vs semantic match)
            if match.similarity >= 0.95:
                base_score *= 1.1  # 10% boost for very high similarity
            
            # Location boost (beginning/middle preferred)
            if match.location in ['beginning', 'middle']:
                base_score *= 1.05  # 5% boost
            
            scored_evidence.append((document, match.matched_term, base_score))
            total_score += base_score
        
        # Normalize
        if len(scored_evidence) > 0:
            avg_score = total_score / len(scored_evidence)
        else:
            avg_score = 0.0
        
        return avg_score, scored_evidence
    
    def aggregate_multi_document_evidence(
        self,
        all_evidence: Dict[str, List[Tuple[Document, str, float]]]
    ) -> Dict[str, float]:
        """
        Aggregate evidence across multiple documents.
        
        Args:
            all_evidence: Dict mapping option_label to list of evidence tuples
            
        Returns:
            Dict mapping option_label to aggregated score
        """
        aggregated_scores = {}
        
        for option_label, evidence_list in all_evidence.items():
            if not evidence_list:
                aggregated_scores[option_label] = 0.0
                continue
            
            # Count unique documents supporting this option
            unique_docs = set()
            for doc, _, _ in evidence_list:
                if doc:
                    doc_id = (doc.metadata.get('guideline_id'), doc.metadata.get('chunk_index'))
                    unique_docs.add(doc_id)
            
            # Sum scores
            total_score = sum(score for _, _, score in evidence_list)
            
            # Boost if multiple documents support
            if len(unique_docs) > 1:
                total_score *= 1.2  # 20% boost for multiple document support
            
            # Average score
            avg_score = total_score / len(evidence_list) if evidence_list else 0.0
            
            # Cap at 1.0
            aggregated_scores[option_label] = min(1.0, avg_score)
        
        return aggregated_scores
    
    def detect_missing_concepts(
        self,
        question: str,
        case_description: str,
        retrieved_documents: List[Document]
    ) -> List[str]:
        """
        Detect medical concepts in question/case that are missing in retrieved documents.
        
        Returns:
            List of missing critical concepts
        """
        # Extract concepts from question and case
        full_text = f"{case_description} {question}".lower()
        question_concepts = self._extract_medical_terms(full_text)
        
        # Extract concepts from retrieved documents
        doc_concepts = set()
        for doc in retrieved_documents:
            doc_terms = self._extract_medical_terms(doc.content.lower())
            doc_concepts.update(doc_terms)
        
        # Find missing concepts
        missing = []
        for concept in question_concepts:
            # Check if concept or its synonym is in documents
            found = False
            if concept in doc_concepts:
                found = True
            else:
                # Check synonyms
                synonym = self._check_synonym_groups(concept, ' '.join([d.content.lower() for d in retrieved_documents]))
                if synonym:
                    found = True
            
            if not found and len(concept) > 4:  # Only significant concepts
                missing.append(concept)
        
        return missing[:3]  # Return top 3 missing concepts

