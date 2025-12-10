"""
Context Pruner

Step 2 Fix: Aggressive pruning of context before reasoning.

The model is drowning in irrelevant text. Keep only:
- Top 3 paragraphs
- That contain the disease keyword or symptom keyword

Expected gain: +8-10% accuracy
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.document_processor import Document


@dataclass
class PrunedContext:
    """Result of context pruning."""
    original_count: int
    pruned_count: int
    pruned_documents: List[Document]
    pruned_texts: List[str]
    relevance_scores: List[float]
    keywords_matched: List[List[str]]


class ContextPruner:
    """
    Aggressively prune retrieved context to keep only most relevant paragraphs.
    """
    
    def __init__(self, max_paragraphs: int = 3):
        """
        Initialize context pruner.
        
        Args:
            max_paragraphs: Maximum number of paragraphs to keep (default: 3)
        """
        self.max_paragraphs = max_paragraphs
    
    def prune(
        self,
        documents: List[Document],
        symptoms: List[str],
        diseases: List[str],
        question_keywords: List[str] = None,
        min_relevance: float = 0.1
    ) -> PrunedContext:
        """
        Prune documents to keep only top relevant paragraphs.
        
        Args:
            documents: List of retrieved documents
            symptoms: List of symptom keywords to match
            diseases: List of disease keywords to match
            question_keywords: Additional keywords from question
            min_relevance: Minimum relevance score to include
            
        Returns:
            PrunedContext with top relevant paragraphs
        """
        if not documents:
            return PrunedContext(
                original_count=0,
                pruned_count=0,
                pruned_documents=[],
                pruned_texts=[],
                relevance_scores=[],
                keywords_matched=[]
            )
        
        # Combine all keywords
        all_keywords = set()
        all_keywords.update(kw.lower() for kw in symptoms)
        all_keywords.update(kw.lower() for kw in diseases)
        if question_keywords:
            all_keywords.update(kw.lower() for kw in question_keywords)
        
        # Add treatment/management keywords (always relevant)
        treatment_keywords = {
            'treatment', 'therapy', 'management', 'medication', 'drug',
            'dose', 'dosage', 'first-line', 'recommended', 'protocol',
            'guideline', 'indication', 'contraindication'
        }
        
        # Score each document
        scored_docs = []
        for doc in documents:
            score, matched = self._score_document(
                doc, all_keywords, treatment_keywords
            )
            if score >= min_relevance:
                scored_docs.append((doc, score, matched))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_docs = scored_docs[:self.max_paragraphs]
        
        return PrunedContext(
            original_count=len(documents),
            pruned_count=len(top_docs),
            pruned_documents=[d for d, _, _ in top_docs],
            pruned_texts=[d.content for d, _, _ in top_docs],
            relevance_scores=[s for _, s, _ in top_docs],
            keywords_matched=[m for _, _, m in top_docs]
        )
    
    def _score_document(
        self,
        doc: Document,
        keywords: set,
        treatment_keywords: set
    ) -> Tuple[float, List[str]]:
        """
        Score a document based on keyword matches.
        
        Returns:
            Tuple of (score, list of matched keywords)
        """
        content_lower = doc.content.lower()
        matched_keywords = []
        score = 0.0
        
        # Score for keyword matches
        for keyword in keywords:
            if keyword in content_lower:
                matched_keywords.append(keyword)
                # Longer keywords are more specific = higher score
                score += 0.2 + (len(keyword) / 50)
        
        # Bonus for treatment keywords
        treatment_matches = 0
        for kw in treatment_keywords:
            if kw in content_lower:
                treatment_matches += 1
        
        if treatment_matches > 0:
            score += treatment_matches * 0.15
        
        # Bonus for section headers (indicates structured guideline content)
        section_patterns = [
            r'treatment:', r'management:', r'diagnosis:',
            r'first[- ]line', r'recommended', r'protocol'
        ]
        for pattern in section_patterns:
            if re.search(pattern, content_lower):
                score += 0.1
        
        # Penalty for very short content
        if len(doc.content) < 100:
            score *= 0.5
        
        # Penalty for very long content (might be too general)
        if len(doc.content) > 2000:
            score *= 0.8
        
        # Normalize to 0-1 range
        score = min(1.0, score)
        
        return score, matched_keywords
    
    def prune_to_text(
        self,
        documents: List[Document],
        symptoms: List[str],
        diseases: List[str],
        question_keywords: List[str] = None,
        max_chars: int = 3000
    ) -> str:
        """
        Prune and combine documents into a single text.
        
        Args:
            documents: List of documents
            symptoms: Symptom keywords
            diseases: Disease keywords
            question_keywords: Additional question keywords
            max_chars: Maximum characters in output
            
        Returns:
            Combined pruned text
        """
        pruned = self.prune(documents, symptoms, diseases, question_keywords)
        
        if not pruned.pruned_texts:
            # Fallback to first document
            if documents:
                return documents[0].content[:max_chars]
            return ""
        
        # Combine texts
        combined = []
        total_chars = 0
        
        for i, text in enumerate(pruned.pruned_texts):
            if total_chars + len(text) > max_chars:
                # Truncate last piece
                remaining = max_chars - total_chars
                if remaining > 100:
                    combined.append(text[:remaining] + "...")
                break
            combined.append(text)
            total_chars += len(text)
        
        return "\n\n---\n\n".join(combined)


class QuestionKeywordExtractor:
    """Extract relevant keywords from question for context pruning."""
    
    def __init__(self):
        self.question_patterns = [
            # Treatment questions
            (r'(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(?:best|recommended|first[- ]line|initial|appropriate)\s+(\w+(?:\s+\w+)?)', 'treatment'),
            (r'how\s+(?:should|would|to)\s+(?:you\s+)?(?:treat|manage)', 'treatment'),
            
            # Diagnosis questions
            (r'what\s+is\s+(?:the\s+)?(?:most\s+)?likely\s+(\w+)', 'diagnosis'),
            (r'(?:diagnos|differential)', 'diagnosis'),
            
            # Medication questions
            (r'(?:drug|medication|antibiotic)\s+of\s+choice', 'medication'),
        ]
    
    def extract(self, question: str) -> List[str]:
        """Extract keywords from question."""
        keywords = []
        question_lower = question.lower()
        
        # Pattern matching
        for pattern, keyword_type in self.question_patterns:
            match = re.search(pattern, question_lower)
            if match:
                keywords.append(keyword_type)
                if match.groups():
                    keywords.append(match.group(1))
        
        # Direct keyword extraction
        important_words = [
            'treatment', 'therapy', 'management', 'diagnosis', 'medication',
            'antibiotic', 'drug', 'dose', 'first-line', 'initial', 'recommended',
            'appropriate', 'best', 'choice', 'indicated', 'contraindicated'
        ]
        
        for word in important_words:
            if word in question_lower:
                keywords.append(word)
        
        return list(set(keywords))


def main():
    """Test context pruner."""
    print("="*70)
    print("CONTEXT PRUNER TEST")
    print("="*70)
    
    pruner = ContextPruner(max_paragraphs=3)
    keyword_extractor = QuestionKeywordExtractor()
    
    # Mock documents
    class MockDoc:
        def __init__(self, content):
            self.content = content
            self.metadata = {'title': 'Test'}
    
    docs = [
        MockDoc("General introduction to medicine. This is background information."),
        MockDoc("Treatment: For sepsis, first-line therapy is IV antibiotics. Recommended: ceftriaxone plus vancomycin."),
        MockDoc("History of medicine development. Not relevant to clinical care."),
        MockDoc("Management of fever includes antipyretics. Paracetamol 15mg/kg. Monitor temperature."),
        MockDoc("Diagnosis: Sepsis is diagnosed by qSOFA criteria. Look for fever, tachycardia, hypotension."),
    ]
    
    symptoms = ['fever', 'tachycardia']
    diseases = ['sepsis']
    
    result = pruner.prune(docs, symptoms, diseases)
    
    print(f"\nOriginal documents: {result.original_count}")
    print(f"Pruned to: {result.pruned_count}")
    print(f"\nTop documents:")
    for i, (text, score, keywords) in enumerate(zip(
        result.pruned_texts, result.relevance_scores, result.keywords_matched
    )):
        print(f"\n{i+1}. Score: {score:.2f}")
        print(f"   Matched: {keywords}")
        print(f"   Text: {text[:100]}...")
    
    # Test question keyword extraction
    print(f"\n{'='*70}")
    question = "What is the first-line antibiotic treatment for neonatal sepsis?"
    keywords = keyword_extractor.extract(question)
    print(f"Question: {question}")
    print(f"Keywords: {keywords}")


if __name__ == "__main__":
    main()

