"""
Multi-Stage Medical Retrieval System

This module implements a sophisticated three-stage retrieval pipeline designed
specifically for medical information retrieval with high precision.

ARCHITECTURE:
------------
Stage 1: Broad Semantic Search (FAISS)
  - Uses semantic embeddings to find conceptually related documents
  - Returns k=20 candidates for filtering
  - Handles synonyms and paraphrasing

Stage 2: Medical Keyword Filtering (BM25)
  - Filters Stage 1 results using medical terminology
  - Focuses on exact term matching for critical medical terms
  - Reduces noise from overly broad semantic matches

Stage 3: Medical Relevance Reranking (Cross-Encoder)
  - Scores each document for clinical relevance to specific case
  - Uses medical-specific cross-encoder model
  - Applies weighted combination of all three stages

Why Multi-Stage?
----------------
- Stage 1 catches broad concepts (e.g., "heart attack" → MI)
- Stage 2 ensures critical terms are present (e.g., "troponin", "STEMI")
- Stage 3 ranks by clinical relevance to specific patient case
- Together: High recall + High precision + Clinical relevance
"""

import numpy as np
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.faiss_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.document_processor import Document
from models.embeddings import EmbeddingModel


@dataclass
class RetrievalResult:
    """Result from multi-stage retrieval with metadata."""
    document: Document
    final_score: float
    stage1_score: float  # FAISS semantic score
    stage2_score: float  # BM25 keyword score
    stage3_score: float  # Cross-encoder relevance score
    stage1_rank: int
    stage2_rank: int
    stage3_rank: int
    retrieval_metadata: Dict


class MultiStageRetriever:
    """
    Three-stage medical retrieval system.
    
    Combines semantic search, keyword filtering, and relevance reranking
    to achieve high precision medical information retrieval.
    
    Parameters:
        faiss_store: FAISS vector store for semantic search
        bm25_retriever: BM25 retriever for keyword filtering
        embedding_model: Embedding model for query encoding
        stage1_k: Number of candidates from Stage 1 (default: 20)
        stage2_k: Number of candidates after Stage 2 filtering (default: 10)
        stage3_k: Final number of results (default: 5)
        stage1_weight: Weight for semantic scores (default: 0.3)
        stage2_weight: Weight for keyword scores (default: 0.3)
        stage3_weight: Weight for relevance scores (default: 0.4)
    """
    
    def __init__(
        self,
        faiss_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        embedding_model: EmbeddingModel,
        stage1_k: int = 25,  # Increased for better recall
        stage2_k: int = 15,  # Increased for better filtering
        stage3_k: int = 5,
        # FIX 1: Hybrid Retrieval Weighting - favor dense (FAISS) over BM25
        # Research shows 0.65 dense + 0.35 keyword boosts MAP by 20-40%
        stage1_weight: float = 0.65,  # Dense/semantic (FAISS) - increased from 0.3
        stage2_weight: float = 0.35,  # Keyword (BM25) - increased from 0.3
        stage3_weight: float = 0.35,  # Cross-encoder reranking - reduced to balance
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """Initialize multi-stage retriever."""
        self.faiss_store = faiss_store
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
        
        # Stage configuration
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.stage3_k = stage3_k
        
        # Score weights (must sum to 1.0)
        total_weight = stage1_weight + stage2_weight + stage3_weight
        self.stage1_weight = stage1_weight / total_weight
        self.stage2_weight = stage2_weight / total_weight
        self.stage3_weight = stage3_weight / total_weight
        
        # Cross-encoder model (lazy loading)
        self._cross_encoder = None
        self.cross_encoder_name = cross_encoder_name
    
    def _get_cross_encoder(self):
        """Lazy load cross-encoder model for reranking."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                # Use medical-optimized model if available, else general
                # ms-marco-MiniLM-L-6-v2 is good for medical reranking
                self._cross_encoder = CrossEncoder(
                    self.cross_encoder_name,
                    max_length=512,
                    device="cpu"  # keep reranker on CPU to avoid GPU memory / paging issues
                )
            except ImportError:
                # Fallback: use simple similarity if cross-encoder not available
                self._cross_encoder = "fallback"
        return self._cross_encoder
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score_threshold: float = 0.0,
        enhanced_query: Optional[str] = None,
        critical_symptoms: Optional[List[str]] = None,
        all_symptoms: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Execute three-stage retrieval pipeline with symptom-aware ranking.
        
        Args:
            query: Medical query/question
            top_k: Number of results to return (default: stage3_k)
            min_score_threshold: Minimum final score threshold
            enhanced_query: Optional enhanced query (from MedicalQueryEnhancer)
            critical_symptoms: Optional list of critical symptoms for boosting
            all_symptoms: Optional list of all symptoms for context-aware ranking
            
        Returns:
            List of RetrievalResult objects sorted by final score
        """
        if top_k is None:
            top_k = self.stage3_k
        
        # Day 5: Use enhanced query if provided
        retrieval_query = enhanced_query if enhanced_query else query
        retrieval_query = self._expand_query_with_synonyms(retrieval_query)
        
        # Normalize symptoms for matching
        critical_symptoms = [s.lower().strip() for s in (critical_symptoms or [])]
        all_symptoms = [s.lower().strip() for s in (all_symptoms or [])]
        
        start_time = time.time()
        
        # RETRIEVAL IMPROVEMENT: Pre-filter by symptoms if available (multi-stage: first narrow by symptoms)
        # This implements the user's suggestion: "first narrow by symptoms, then by guideline relevance"
        prefiltered_results = None
        if critical_symptoms or all_symptoms:
            # Stage 0.5: Symptom-based pre-filtering
            # Get more candidates from semantic search first
            stage1_start = time.time()
            stage1_candidates = self._stage1_semantic_search(
                retrieval_query, 
                self.stage1_k * 2,  # Get 2x candidates for symptom filtering
                critical_symptoms=critical_symptoms,
                all_symptoms=all_symptoms
            )
            stage1_time = time.time() - stage1_start
            
            # Filter by symptom matches - prioritize documents with symptom matches
            if stage1_candidates:
                symptom_matched = []
                symptom_unmatched = []
                
                for doc, score, rank in stage1_candidates:
                    doc_content_lower = doc.content.lower()
                    doc_words = set(doc_content_lower.split())
                    
                    has_symptom_match = False
                    symptom_match_count = 0
                    
                    # Check critical symptoms
                    if critical_symptoms:
                        for symptom in critical_symptoms:
                            symptom_lower = symptom.lower().strip()
                            if symptom_lower in doc_content_lower:
                                has_symptom_match = True
                                symptom_match_count += 2  # Critical symptoms count double
                            elif len(symptom_lower.split()) > 1:
                                symptom_words = set(symptom_lower.split())
                                if all(word in doc_words for word in symptom_words if len(word) > 2):
                                    has_symptom_match = True
                                    symptom_match_count += 2
                            elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                                has_symptom_match = True
                                symptom_match_count += 2
                    
                    # Check all symptoms
                    if all_symptoms:
                        for symptom in all_symptoms:
                            symptom_lower = symptom.lower().strip()
                            if symptom_lower in doc_content_lower:
                                has_symptom_match = True
                                symptom_match_count += 1
                            elif len(symptom_lower.split()) > 1:
                                symptom_words = set(symptom_lower.split())
                                if all(word in doc_words for word in symptom_words if len(word) > 2):
                                    has_symptom_match = True
                                    symptom_match_count += 1
                            elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                                has_symptom_match = True
                                symptom_match_count += 1
                    
                    if has_symptom_match:
                        # Boost score based on symptom match count
                        boosted_score = min(1.0, score + (symptom_match_count * 0.15))  # 15% per match
                        symptom_matched.append((doc, boosted_score, rank, symptom_match_count))
                    else:
                        symptom_unmatched.append((doc, score, rank))
                
                # Sort symptom-matched by match count and score, then add unmatched
                symptom_matched.sort(key=lambda x: (x[3], x[1]), reverse=True)  # Sort by match count, then score
                symptom_unmatched.sort(key=lambda x: x[1], reverse=True)
                
                # Take top symptom-matched documents, then fill with unmatched if needed
                prefiltered_results = [r[:3] for r in symptom_matched[:self.stage1_k]]  # Top symptom matches
                if len(prefiltered_results) < self.stage1_k:
                    # Fill remaining slots with unmatched (but still relevant) documents
                    remaining = self.stage1_k - len(prefiltered_results)
                    prefiltered_results.extend([r for r in symptom_unmatched[:remaining]])
        
        # Stage 1: Broad Semantic Search (FAISS) with symptom boosting
        if prefiltered_results is None:
            stage1_start = time.time()
            stage1_results = self._stage1_semantic_search(
                retrieval_query, 
                self.stage1_k,
                critical_symptoms=critical_symptoms,
                all_symptoms=all_symptoms
            )
            stage1_time = time.time() - stage1_start
        else:
            # Use pre-filtered results
            stage1_results = prefiltered_results
            stage1_time = 0.0  # Already calculated
        
        if not stage1_results:
            return []
        
        # Stage 2: Medical Keyword Filtering (BM25) with symptom-aware filtering
        stage2_start = time.time()
        stage2_results = self._stage2_keyword_filter(
            retrieval_query, 
            stage1_results, 
            self.stage2_k,
            critical_symptoms=critical_symptoms,
            all_symptoms=all_symptoms
        )
        stage2_time = time.time() - stage2_start
        
        if not stage2_results:
            # Fallback: use Stage 1 results if Stage 2 filters everything
            stage2_results = stage1_results[:self.stage2_k]
        
        # Stage 3: Medical Relevance Reranking
        stage3_start = time.time()
        # Extract guideline mentioned from query for post-processing
        import re
        guideline_mentioned = None
        guideline_patterns = [
            r"'([^']+)' guideline",
            r'"([^"]+)" guideline',
            r'based on the ([^,\.]+)',
            r'according to the ([^,\.]+)',
            r'per the ([^,\.]+)'
        ]
        query_lower = retrieval_query.lower()
        for pattern in guideline_patterns:
            match = re.search(pattern, retrieval_query, re.IGNORECASE)
            if match:
                guideline_mentioned = match.group(1).strip().lower()
                break
        stage3_results = self._stage3_relevance_rerank(retrieval_query, stage2_results, top_k, guideline_mentioned)
        stage3_time = time.time() - stage3_start
        
        # Filter by threshold
        final_results = [
            r for r in stage3_results
            if r.final_score >= min_score_threshold
        ]
        
        total_time = time.time() - start_time
        
        # Add timing metadata
        for result in final_results:
            result.retrieval_metadata.update({
                'stage1_time_ms': stage1_time * 1000,
                'stage2_time_ms': stage2_time * 1000,
                'stage3_time_ms': stage3_time * 1000,
                'total_time_ms': total_time * 1000
            })
        
        return final_results
    
    def _stage1_semantic_search(
        self,
        query: str,
        k: int,
        critical_symptoms: Optional[List[str]] = None,
        all_symptoms: Optional[List[str]] = None
    ) -> List[Tuple[Document, float, int]]:
        """
        Stage 1: Broad semantic search using FAISS.
        
        Day 7: Improved to get more candidates for better recall.
        
        Returns:
            List of (document, score, rank) tuples
        """
        # PubMedBERT: Search with very low threshold to get maximum candidates
        # 768-dim embeddings may have different similarity distribution, so use very lenient threshold
        results = self.faiss_store.search(query, top_k=k, score_threshold=-1.0)  # Very lenient for maximum recall
        
        # PubMedBERT: If query mentions a specific guideline, boost matching documents significantly
        import re
        guideline_mentioned = None
        guideline_patterns = [
            r"'([^']+)' guideline",
            r'"([^"]+)" guideline',
            r'based on the ([^,\.]+)',
            r'according to the ([^,\.]+)',
            r'per the ([^,\.]+)'
        ]
        query_lower = query.lower()
        for pattern in guideline_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                guideline_mentioned = match.group(1).strip().lower()
                break
        
        stage1_results = []
        for rank, (doc, score) in enumerate(results, 1):
            # Day 7: Enhanced category and guideline matching
            doc_category = doc.metadata.get('category', '').lower()
            doc_guideline_id = doc.metadata.get('guideline_id', '').lower()
            # Get guideline title from metadata (try multiple field names)
            doc_guideline_title = (
                doc.metadata.get('guideline_title', '') or 
                doc.metadata.get('title', '') or
                doc.metadata.get('guideline_name', '')
            ).lower()
            
            # Boost if category keywords match (enhanced for Pediatrics)
            category_boost = 0.0
            if doc_category and any(cat_word in query_lower for cat_word in doc_category.split()):
                category_boost = 0.15  # 15% boost (increased from 10%)
                # Extra boost for Pediatrics queries
                if 'pediatric' in doc_category.lower() or 'pediatric' in query_lower or 'newborn' in query_lower or 'neonatal' in query_lower:
                    category_boost = 0.20  # 20% boost for Pediatrics (increased from 15%)
            
            # PubMedBERT: Boost if guideline title keywords match query (more aggressive)
            guideline_boost = 0.0
            if doc_guideline_title:
                # Check if key terms from guideline title appear in query
                title_words = [w for w in doc_guideline_title.split() if len(w) > 3]  # Lowered from 4
                query_words = set(query_lower.split())
                matching_title_words = sum(1 for word in title_words if word in query_words)
                if matching_title_words >= 2:  # At least 2 matching words
                    guideline_boost = 0.30  # 30% boost for guideline title match (increased from 25%)
                elif matching_title_words >= 1:  # At least 1 matching word
                    guideline_boost = 0.20  # 20% boost for single match (increased from 15%)
                
                # PubMedBERT: If query explicitly mentions a guideline name, massive boost if it matches
                if guideline_mentioned:
                    title_lower = doc_guideline_title.lower()
                    guideline_mentioned_lower = guideline_mentioned.lower()
                    # Check if mentioned guideline name is in title (exact or contains)
                    if (guideline_mentioned_lower in title_lower or 
                        title_lower in guideline_mentioned_lower or
                        title_lower == guideline_mentioned_lower):
                        guideline_boost = 0.60  # 60% boost for explicit guideline name match! (increased from 50%)
                    # Check for partial match (e.g., "sick newborn" in "Sick Newborn Care")
                    elif any(word in title_lower for word in guideline_mentioned_lower.split() if len(word) > 2):  # Lowered from 3
                        # Count matching words
                        matching_words = sum(1 for word in guideline_mentioned_lower.split() if len(word) > 2 and word in title_lower)
                        if matching_words >= 2:
                            guideline_boost = max(guideline_boost, 0.50)  # 50% boost for 2+ word match (increased from 40%)
                        elif matching_words >= 1:
                            guideline_boost = max(guideline_boost, 0.40)  # 40% boost for 1 word match
            
            # PubMedBERT: Boost if query mentions guideline ID or similar (more aggressive)
            if doc_guideline_id:
                # Check exact ID match
                if doc_guideline_id.lower() in query_lower:
                    guideline_boost = max(guideline_boost, 0.40)  # 40% boost for exact ID match (increased from 30%)
                # Check ID with underscores/spaces
                elif doc_guideline_id.replace('_', ' ').lower() in query_lower:
                    guideline_boost = max(guideline_boost, 0.30)  # 30% boost (increased from 20%)
                # Check if query mentions "guideline" and ID components match
                elif 'guideline' in query_lower:
                    id_parts = doc_guideline_id.lower().split('_')
                    query_id_parts = [p for p in query_lower.split() if len(p) > 2]
                    matching_parts = sum(1 for part in id_parts if part in query_id_parts)
                    if matching_parts >= 1:
                        guideline_boost = max(guideline_boost, 0.20)  # 20% boost (increased from 15%)
            
            # Day 7: Boost for "initial treatment" queries
            initial_treatment_boost = 0.0
            if 'initial' in query_lower and ('treatment' in query_lower or 'therapy' in query_lower or 'management' in query_lower):
                # Check if document mentions "initial" or "first-line" treatment
                doc_content_lower = doc.content.lower()
                if any(term in doc_content_lower for term in ['initial treatment', 'first-line treatment', 'primary treatment', 'initial therapy', 'first treatment']):
                    initial_treatment_boost = 0.12  # 12% boost for initial treatment matches
            
            # RETRIEVAL IMPROVEMENT: Symptom-based boosting with improved matching
            symptom_boost = 0.0
            if critical_symptoms or all_symptoms:
                doc_content_lower = doc.content.lower()
                doc_words = set(doc_content_lower.split())  # Word-based matching for better recall
                
                # Boost for critical symptoms (higher weight) - use word-based matching
                if critical_symptoms:
                    critical_matches = 0
                    for symptom in critical_symptoms:
                        symptom_lower = symptom.lower().strip()
                        # Exact phrase match (higher confidence)
                        if symptom_lower in doc_content_lower:
                            critical_matches += 1
                        # Word-based match (if symptom has multiple words, check if all words present)
                        elif len(symptom_lower.split()) > 1:
                            symptom_words = set(symptom_lower.split())
                            if len(symptom_words) > 0 and all(word in doc_words for word in symptom_words if len(word) > 2):
                                critical_matches += 1
                        # Single word match (must be significant word)
                        elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                            critical_matches += 1
                    
                    if critical_matches > 0:
                        # 20% boost per critical symptom match, up to 40% total (increased for better recall)
                        symptom_boost = min(0.40, critical_matches * 0.20)
                
                # Additional boost for all symptoms (moderate weight) - use word-based matching
                if all_symptoms and symptom_boost < 0.30:
                    all_matches = 0
                    for symptom in all_symptoms:
                        symptom_lower = symptom.lower().strip()
                        # Exact phrase match
                        if symptom_lower in doc_content_lower:
                            all_matches += 1
                        # Word-based match for multi-word symptoms
                        elif len(symptom_lower.split()) > 1:
                            symptom_words = set(symptom_lower.split())
                            if len(symptom_words) > 0 and all(word in doc_words for word in symptom_words if len(word) > 2):
                                all_matches += 1
                        # Single word match
                        elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                            all_matches += 1
                    
                    if all_matches > 0:
                        # 5% boost per symptom match, up to 20% total
                        additional_boost = min(0.20, all_matches * 0.05)
                        symptom_boost = min(0.40, symptom_boost + additional_boost)  # Cap at 40% total
            
            total_boost = min(0.70, category_boost + guideline_boost + initial_treatment_boost + symptom_boost)  # Cap at 70% (increased for better recall)
            adjusted_score = min(1.0, score + total_boost)
            stage1_results.append((doc, adjusted_score, rank))
        
        return stage1_results

    def _expand_query_with_synonyms(self, query: str) -> str:
        """
        Lightweight synonym/normalization expansion to improve recall for ACS/stroke/ECG.
        """
        expansions = {
            "retrosternal pain": ["central chest pain", "pressure-like chest pain", "crushing chest pain"],
            "chest pain": ["precordial pain", "angina", "tight chest pain"],
            "radiating": ["radiation to arm", "radiates to jaw", "left arm pain"],
            "ecg": ["ekg", "electrocardiogram"],
            "stemi": ["st elevation mi", "st-segment elevation myocardial infarction"],
            "nstemi": ["non st elevation mi", "non-st-segment elevation myocardial infarction"],
            "troponin": ["cardiac enzyme", "cardiac biomarker"],
            "hemiparesis": ["unilateral weakness", "one sided weakness"],
            "facial droop": ["facial asymmetry"],
            "aphasia": ["language deficit", "word finding difficulty"],
            "tpa": ["thrombolysis", "alteplase"],
            "nitroglycerin": ["glyceryl trinitrate", "gtn"]
        }
        lower_q = query.lower()
        extra_terms = []
        for key, syns in expansions.items():
            if key in lower_q:
                extra_terms.extend(syns)
        if extra_terms:
            query = query + " " + " ".join(extra_terms)
        return query
    
    def _stage2_keyword_filter(
        self,
        query: str,
        stage1_results: List[Tuple[Document, float, int]],
        k: int,
        critical_symptoms: Optional[List[str]] = None,
        all_symptoms: Optional[List[str]] = None
    ) -> List[Tuple[Document, float, float, int, int]]:
        """
        Stage 2: Filter Stage 1 results using BM25 keyword matching.
        
        Returns:
            List of (document, stage1_score, stage2_score, stage1_rank, stage2_rank) tuples
        """
        # Get BM25 scores for all Stage 1 documents
        bm25_results = self.bm25_retriever.search(query, top_k=len(stage1_results), score_threshold=0.0)
        
        # Create mapping of document to BM25 score
        bm25_scores = {}
        bm25_ranks = {}
        for rank, (doc, score) in enumerate(bm25_results, 1):
            # Use document content as key (since Document objects aren't hashable)
            doc_key = (doc.metadata['guideline_id'], doc.metadata['chunk_index'])
            bm25_scores[doc_key] = score
            bm25_ranks[doc_key] = rank
        
        # Combine Stage 1 and Stage 2 scores with symptom-aware boosting
        combined_results = []
        for doc, stage1_score, stage1_rank in stage1_results:
            doc_key = (doc.metadata['guideline_id'], doc.metadata['chunk_index'])
            stage2_score = bm25_scores.get(doc_key, 0.0)
            stage2_rank = bm25_ranks.get(doc_key, len(stage1_results) + 1)
            
            # RETRIEVAL IMPROVEMENT: Boost BM25 score for symptom matches with improved matching
            if critical_symptoms or all_symptoms:
                doc_content_lower = doc.content.lower()
                doc_words = set(doc_content_lower.split())  # Word-based matching
                
                # Boost BM25 score if symptoms are found in document
                symptom_bm25_boost = 0.0
                if critical_symptoms:
                    critical_matches = 0
                    for symptom in critical_symptoms:
                        symptom_lower = symptom.lower().strip()
                        if symptom_lower in doc_content_lower:
                            critical_matches += 1
                        elif len(symptom_lower.split()) > 1:
                            symptom_words = set(symptom_lower.split())
                            if all(word in doc_words for word in symptom_words if len(word) > 2):
                                critical_matches += 1
                        elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                            critical_matches += 1
                    
                    if critical_matches > 0:
                        symptom_bm25_boost += min(0.4, critical_matches * 0.15)  # 15% per critical symptom (increased)
                
                if all_symptoms:
                    all_matches = 0
                    for symptom in all_symptoms:
                        symptom_lower = symptom.lower().strip()
                        if symptom_lower in doc_content_lower:
                            all_matches += 1
                        elif len(symptom_lower.split()) > 1:
                            symptom_words = set(symptom_lower.split())
                            if all(word in doc_words for word in symptom_words if len(word) > 2):
                                all_matches += 1
                        elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                            all_matches += 1
                    
                    if all_matches > 0:
                        symptom_bm25_boost += min(0.3, all_matches * 0.08)  # 8% per symptom (increased)
                
                # Apply boost to BM25 score
                stage2_score = min(1.0, stage2_score + symptom_bm25_boost)
            
            combined_results.append((doc, stage1_score, stage2_score, stage1_rank, stage2_rank))
        
        # Day 7: Improved sorting with better weighting and normalization
        # Normalize BM25 scores first (they can be >1)
        normalized_results = []
        for doc, stage1_score, stage2_score, stage1_rank, stage2_rank in combined_results:
            # Normalize stage2_score (BM25 can be >1)
            if stage2_score > 0:
                normalized_stage2 = min(1.0, 1.0 - (1.0 / (1.0 + stage2_score)))
            else:
                normalized_stage2 = 0.0
            normalized_results.append((doc, stage1_score, normalized_stage2, stage1_rank, stage2_rank))
        
        # Use stage weights for combination (better than equal weights)
        normalized_results.sort(
            key=lambda x: x[1] * self.stage1_weight + x[2] * self.stage2_weight,
            reverse=True
        )
        
        # Day 7: If we have fewer results than k, return all
        # Otherwise return top k
        return normalized_results[:k] if len(normalized_results) >= k else normalized_results
    
    def _stage3_relevance_rerank(
        self,
        query: str,
        stage2_results: List[Tuple[Document, float, float, int, int]],
        k: int,
        guideline_mentioned: Optional[str] = None,
        critical_symptoms: Optional[List[str]] = None,
        all_symptoms: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Stage 3: Rerank using cross-encoder for clinical relevance.
        
        Includes Fix 3: Guideline Prioritization Reranker
        - Boost documents with "management", "diagnosis", "treatment", "first-line", "indications"
        - Boost documents with exact disease term match
        
        Returns:
            List of RetrievalResult objects with final scores
        """
        cross_encoder = self._get_cross_encoder()
        
        # Fix 3: Guideline prioritization keywords for boosting
        guideline_boost_keywords = [
            'management', 'diagnosis', 'treatment', 'first-line', 
            'indications', 'recommended', 'protocol', 'guideline',
            'therapy', 'initial treatment', 'primary treatment',
            'treatment of choice', 'standard of care'
        ]
        
        # Extract disease terms from query
        disease_patterns = [
            r'\b(pneumonia|sepsis|meningitis|diabetes|hypertension|'
            r'myocardial infarction|heart failure|stroke|pulmonary embolism|'
            r'preeclampsia|eclampsia|anemia|infection|shock|arrhythmia|'
            r'tuberculosis|malaria|cholera|typhoid|hepatitis|'
            r'bronchitis|asthma|copd|hypoglycemia|hyperglycemia|'
            r'neonatal sepsis|newborn sepsis|pediatric|neonatal)\b'
        ]
        disease_terms = []
        for pattern in disease_patterns:
            matches = re.findall(pattern, query.lower(), re.IGNORECASE)
            disease_terms.extend(matches)
        
        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = []
        for doc, stage1_score, stage2_score, stage1_rank, stage2_rank in stage2_results:
            # Truncate document content to max length
            doc_text = doc.content[:500]  # Limit to 500 chars for cross-encoder
            query_doc_pairs.append([query, doc_text])
        
        # Get relevance scores
        if cross_encoder == "fallback":
            # Fallback: use weighted combination of Stage 1 and Stage 2
            relevance_scores = [
                (r[1] * self.stage1_weight + r[2] * self.stage2_weight)
                for r in stage2_results
            ]
        else:
            try:
                relevance_scores = cross_encoder.predict(query_doc_pairs)
                # Day 7: Improved normalization - use min-max with better handling
                if len(relevance_scores) > 0:
                    min_score = min(relevance_scores)
                    max_score = max(relevance_scores)
                    if max_score > min_score:
                        # Min-max normalization with better distribution for PubMedBERT
                        relevance_scores = [
                            (s - min_score) / (max_score - min_score)
                            for s in relevance_scores
                        ]
                        # Ensure scores are in [0, 1] and boost top scores more aggressively
                        relevance_scores = [max(0.0, min(1.0, s)) for s in relevance_scores]
                        # Boost top 3 scores to improve ranking
                        sorted_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)
                        for idx, rank in enumerate(sorted_indices[:3]):
                            boost = 1.0 + (0.15 - idx * 0.05)  # 15%, 10%, 5% boost
                            relevance_scores[rank] = min(1.0, relevance_scores[rank] * boost)
                    else:
                        relevance_scores = [0.5] * len(relevance_scores)
            except Exception as e:
                # Fallback on error
                print(f"[WARN] Cross-encoder failed, using fallback: {e}")
                relevance_scores = [
                    (r[1] * self.stage1_weight + r[2] * self.stage2_weight)
                    for r in stage2_results
                ]
        
        # Combine all three stages with weights
        final_results = []
        for i, (doc, stage1_score, stage2_score, stage1_rank, stage2_rank) in enumerate(stage2_results):
            stage3_score = float(relevance_scores[i])
            stage3_rank = i + 1
            
            # Day 7: Improved score normalization
            # Normalize scores to [0, 1] range for combination
            # Stage 1 (FAISS): cosine similarity, already in [0, 1] range
            normalized_stage1 = max(0, min(1, stage1_score))
            
            # Stage 2 (BM25): scores can be >1, normalize more carefully
            # Use log normalization for BM25 scores
            if stage2_score > 0:
                normalized_stage2 = min(1.0, 1.0 - (1.0 / (1.0 + stage2_score)))
            else:
                normalized_stage2 = 0.0
            
            # Stage 3 (Cross-encoder): already normalized
            normalized_stage3 = max(0, min(1, stage3_score))
            
            # PubMedBERT: Weighted combination with better balance and boost for high scores
            base_score = (
                normalized_stage1 * self.stage1_weight +
                normalized_stage2 * self.stage2_weight +
                normalized_stage3 * self.stage3_weight
            )
            # Boost if multiple stages have high scores (indicates strong match)
            high_stage_count = sum(1 for score in [normalized_stage1, normalized_stage2, normalized_stage3] if score > 0.6)
            if high_stage_count >= 2:
                final_score = min(1.0, base_score * 1.1)  # 10% boost for strong multi-stage matches
            else:
                final_score = base_score
            
            # Fix 3: Guideline Prioritization Boosting
            doc_content_lower = doc.content.lower()
            
            # Boost for guideline keywords (+0.2 max)
            guideline_keyword_boost = 0.0
            for keyword in guideline_boost_keywords:
                if keyword in doc_content_lower:
                    guideline_keyword_boost += 0.04  # 4% per keyword
            guideline_keyword_boost = min(0.20, guideline_keyword_boost)
            
            # Boost for disease term match (+0.3 max)
            disease_term_boost = 0.0
            for term in disease_terms:
                if term.lower() in doc_content_lower:
                    disease_term_boost += 0.10  # 10% per disease match
            disease_term_boost = min(0.30, disease_term_boost)
            
            # Apply guideline prioritization boosts
            final_score = min(1.0, final_score + guideline_keyword_boost + disease_term_boost)
            
            # RETRIEVAL IMPROVEMENT: Context-aware symptom-based boosting in final ranking with improved matching
            if critical_symptoms or all_symptoms:
                doc_content_lower = doc.content.lower()
                doc_words = set(doc_content_lower.split())  # Word-based matching
                symptom_final_boost = 0.0
                
                # Critical symptoms get highest boost (20% per symptom, up to 40%)
                if critical_symptoms:
                    critical_matches = 0
                    for symptom in critical_symptoms:
                        symptom_lower = symptom.lower().strip()
                        if symptom_lower in doc_content_lower:
                            critical_matches += 1
                        elif len(symptom_lower.split()) > 1:
                            symptom_words = set(symptom_lower.split())
                            if all(word in doc_words for word in symptom_words if len(word) > 2):
                                critical_matches += 1
                        elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                            critical_matches += 1
                    
                    if critical_matches > 0:
                        symptom_final_boost = min(0.50, critical_matches * 0.25)  # 25% per critical symptom, up to 50% (increased)
                
                # All symptoms get moderate boost (8% per symptom, up to 30%)
                if all_symptoms and symptom_final_boost < 0.50:
                    all_matches = 0
                    for symptom in all_symptoms:
                        symptom_lower = symptom.lower().strip()
                        if symptom_lower in doc_content_lower:
                            all_matches += 1
                        elif len(symptom_lower.split()) > 1:
                            symptom_words = set(symptom_lower.split())
                            if all(word in doc_words for word in symptom_words if len(word) > 2):
                                all_matches += 1
                        elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                            all_matches += 1
                    
                    if all_matches > 0:
                        additional_boost = min(0.30, all_matches * 0.08)  # 8% per symptom, up to 30%
                        symptom_final_boost = min(0.70, symptom_final_boost + additional_boost)  # Cap at 70% (increased)
                
                # Apply symptom boost to final score
                if symptom_final_boost > 0:
                    final_score = min(1.0, final_score + symptom_final_boost)
            
            result = RetrievalResult(
                document=doc,
                final_score=final_score,
                stage1_score=normalized_stage1,
                stage2_score=normalized_stage2,
                stage3_score=normalized_stage3,
                stage1_rank=stage1_rank,
                stage2_rank=stage2_rank,
                stage3_rank=stage3_rank,
                retrieval_metadata={
                    'query': query,
                    'stages_used': ['semantic', 'keyword', 'rerank']
                }
            )
            final_results.append(result)
        
        # Sort by final score
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # PubMedBERT: Post-process to boost guideline-specific matches
        # If query mentions a specific guideline, re-rank to prioritize matching documents
        if guideline_mentioned:
            # Separate matching and non-matching documents
            matching_docs = []
            non_matching_docs = []
            
            for result in final_results:
                doc = result.document
                # Get guideline title from metadata (try multiple field names)
                doc_title = (
                    doc.metadata.get('guideline_title', '') or 
                    doc.metadata.get('title', '') or
                    doc.metadata.get('guideline_name', '')
                ).lower()
                doc_id = doc.metadata.get('guideline_id', '').lower()
                
                # Check if this document matches the mentioned guideline (more aggressive matching)
                matches = False
                guideline_mentioned_lower = guideline_mentioned.lower()
                
                # Exact match or contains
                if (guideline_mentioned_lower in doc_title or 
                    doc_title in guideline_mentioned_lower or
                    doc_title == guideline_mentioned_lower):
                    matches = True
                # Partial match - check if all significant words match
                elif any(word in doc_title for word in guideline_mentioned_lower.split() if len(word) > 2):  # Lowered from 3
                    matching_words = sum(1 for word in guideline_mentioned_lower.split() if len(word) > 2 and word in doc_title)
                    
                    # RETRIEVAL IMPROVEMENT: Also check if symptoms match for guideline-specific queries
                    if critical_symptoms or all_symptoms:
                        doc_content_lower = doc.content.lower()
                        doc_words = set(doc_content_lower.split())
                        symptom_matches = 0
                        
                        # Check critical symptoms with improved matching
                        if critical_symptoms:
                            for symptom in critical_symptoms:
                                symptom_lower = symptom.lower().strip()
                                if symptom_lower in doc_content_lower:
                                    symptom_matches += 1
                                elif len(symptom_lower.split()) > 1:
                                    symptom_words = set(symptom_lower.split())
                                    if all(word in doc_words for word in symptom_words if len(word) > 2):
                                        symptom_matches += 1
                                elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                                    symptom_matches += 1
                        
                        # Check all symptoms with improved matching
                        if all_symptoms:
                            for symptom in all_symptoms:
                                symptom_lower = symptom.lower().strip()
                                if symptom_lower in doc_content_lower:
                                    symptom_matches += 1
                                elif len(symptom_lower.split()) > 1:
                                    symptom_words = set(symptom_lower.split())
                                    if all(word in doc_words for word in symptom_words if len(word) > 2):
                                        symptom_matches += 1
                                elif len(symptom_lower) > 3 and symptom_lower in doc_words:
                                    symptom_matches += 1
                        
                        # If symptoms match, increase confidence in guideline match
                        if symptom_matches > 0:
                            matching_words += 1  # Boost matching score
                    if matching_words >= 2:  # At least 2 words match
                        matches = True
                # ID match
                elif guideline_mentioned_lower.replace(' ', '_') in doc_id or guideline_mentioned_lower.replace('_', ' ') in doc_id:
                    matches = True
                # Check if guideline name appears in document content (for very specific queries)
                elif guideline_mentioned_lower in doc.content.lower()[:200]:  # Check first 200 chars
                    matches = True
                
                if matches:
                    # Massive boost for matching documents - prioritize them heavily
                    result.final_score = min(1.0, result.final_score * 1.8 + 0.4)  # Increased boost: 1.8x + 0.4 base
                    matching_docs.append(result)
                else:
                    non_matching_docs.append(result)
            
            # Re-sort: matching docs first, then non-matching
            if matching_docs:
                matching_docs.sort(key=lambda x: x.final_score, reverse=True)
                non_matching_docs.sort(key=lambda x: x.final_score, reverse=True)
                final_results = matching_docs + non_matching_docs
        
        return final_results[:k]
    
    def get_statistics(self) -> Dict:
        """Get statistics about the retriever."""
        return {
            'stage1_k': self.stage1_k,
            'stage2_k': self.stage2_k,
            'stage3_k': self.stage3_k,
            'stage1_weight': self.stage1_weight,
            'stage2_weight': self.stage2_weight,
            'stage3_weight': self.stage3_weight,
            'faiss_vectors': self.faiss_store.index.ntotal if self.faiss_store.index else 0,
            'bm25_documents': len(self.bm25_retriever.documents)
        }


def main():
    """Demo: Test multi-stage retrieval."""
    print("="*70)
    print("MULTI-STAGE RETRIEVAL SYSTEM DEMO")
    print("="*70)
    
    # Initialize components
    print("\n[INFO] Initializing components...")
    embedding_model = EmbeddingModel()
    
    from retrieval.faiss_store import FAISSVectorStore
    from retrieval.bm25_retriever import BM25Retriever
    
    faiss_store = FAISSVectorStore(embedding_model)
    index_dir = Path(__file__).parent.parent.parent / "data" / "indexes"
    
    if (index_dir / "faiss_index.bin").exists():
        print("[INFO] Loading FAISS index...")
        faiss_store.load_index(str(index_dir))
    else:
        print("[ERROR] FAISS index not found. Run scripts/build_indexes.py first")
        return
    
    bm25_retriever = BM25Retriever(faiss_store.documents)
    
    # Create multi-stage retriever
    multi_stage = MultiStageRetriever(
        faiss_store=faiss_store,
        bm25_retriever=bm25_retriever,
        embedding_model=embedding_model,
        stage1_k=20,
        stage2_k=10,
        stage3_k=5
    )
    
    # Test queries
    test_queries = [
        "What is the treatment for acute myocardial infarction?",
        "How to manage diabetes type 2?",
        "Patient with elevated troponin levels"
    ]
    
    print(f"\n{'='*70}")
    print("[TEST] TESTING MULTI-STAGE RETRIEVAL")
    print(f"{'='*70}")
    
    for query in test_queries:
        print(f"\n{'-'*70}")
        print(f"Query: {query}")
        print(f"{'-'*70}")
        
        results = multi_stage.retrieve(query, top_k=3)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Final Score: {result.final_score:.4f}")
            print(f"   Stage 1 (Semantic): {result.stage1_score:.3f} (rank {result.stage1_rank})")
            print(f"   Stage 2 (Keyword): {result.stage2_score:.3f} (rank {result.stage2_rank})")
            print(f"   Stage 3 (Rerank): {result.stage3_score:.3f} (rank {result.stage3_rank})")
            print(f"   Title: {result.document.metadata['title']}")
            print(f"   Category: {result.document.metadata['category']}")
            print(f"   Preview: {result.document.content[:150]}...")
            print(f"   Total time: {result.retrieval_metadata.get('total_time_ms', 0):.1f}ms")
    
    print(f"\n{'='*70}")
    print("[OK] Multi-stage retrieval system operational!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


