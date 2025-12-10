"""
Multi-Stage RAG Pipeline

This module integrates all components into a unified pipeline:
1. Query understanding
2. Multi-stage retrieval
3. Medical reasoning
4. Answer selection with confidence

This is the main entry point for the Day 3 multi-stage RAG system.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import json
import time
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.multi_stage_retriever import MultiStageRetriever, RetrievalResult
from retrieval.concept_first_retriever import ConceptFirstRetriever
from retrieval.faiss_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from models.embeddings import EmbeddingModel
from reasoning.query_understanding import MedicalQueryUnderstanding
from reasoning.medical_reasoning import MedicalReasoningEngine, AnswerSelection

# Day 5 improvements
try:
    from improvements.medical_query_enhancer import MedicalQueryEnhancer
    from improvements.structured_reasoner import StructuredMedicalReasoner
    from improvements.confidence_calibrator import ConfidenceCalibrator
    from improvements.specialty_adapter import SpecialtyAdapter
    IMPROVEMENTS_AVAILABLE = True
except ImportError:
    IMPROVEMENTS_AVAILABLE = False
    print("[WARN] Day 5 improvements not available, using baseline system")

# Day 6 improvements
try:
    from optimization.symptom_extractor import EnhancedSymptomExtractor
    from specialties.obgyn_handler import OBGYNHandler
    DAY6_IMPROVEMENTS_AVAILABLE = True
except ImportError:
    DAY6_IMPROVEMENTS_AVAILABLE = False
    print("[WARN] Day 6 improvements not available")

# Day 7 reasoning improvements (Fixes 1-6)
try:
    from improvements.multi_query_expander import MultiQueryExpander, merge_and_deduplicate_results
    from improvements.reasoning_improvements import (
        EnhancedReasoningPipeline,
        ForcedUncertaintyAnalyzer,
        MinimalContextEvaluator,
        DifferentialDiagnosisReasoner,
        GuidelinePrioritizationReranker
    )
    DAY7_FIXES_AVAILABLE = True
except ImportError:
    DAY7_FIXES_AVAILABLE = False
    print("[WARN] Day 7 reasoning fixes not available")

# Day 7+ Critical Fixes (Multi-Query, Symptom Injection, Guideline Reranking, Enhanced Reasoning)
try:
    from improvements.multi_query_expander import MultiQueryExpander
    from improvements.symptom_synonym_injector import SymptomSynonymInjector
    from improvements.guideline_reranker import GuidelineReranker
    from improvements.enhanced_reasoning import EnhancedMedicalReasoner
    DAY7_FIXES_AVAILABLE = True
except ImportError:
    DAY7_FIXES_AVAILABLE = False
    print("[WARN] Day 7+ critical fixes not available")

# Day 7+ Step Fixes (Clinical Feature Extraction, Context Pruning, Deterministic Reasoning)
try:
    from improvements.clinical_feature_extractor import ClinicalFeatureExtractor
    from improvements.context_pruner import ContextPruner, QuestionKeywordExtractor
    from improvements.deterministic_reasoner import DeterministicReasoner
    DAY7_STEP_FIXES_AVAILABLE = True
except ImportError:
    DAY7_STEP_FIXES_AVAILABLE = False
    print("[WARN] Day 7+ step fixes not available")

# Day 8+ Advanced Reasoning Fixes (Structured Reasoning, Safety, Hallucination Detection)
try:
    from improvements.structured_medical_reasoner_v2 import (
        StructuredMedicalReasonerV2,
        ReasoningMode,
        PatientCategory
    )
    from improvements.safety_verifier import MedicalSafetyVerifier, SafetyLevel
    from improvements.hallucination_detector import HallucinationDetector
    from improvements.terminology_normalizer import TerminologyNormalizer
    DAY8_ADVANCED_FIXES_AVAILABLE = True
except ImportError:
    DAY8_ADVANCED_FIXES_AVAILABLE = False
    print("[WARN] Day 8+ advanced reasoning fixes not available")


@dataclass
class PipelineResult:
    """Complete result from RAG pipeline."""
    question_id: str
    question: str
    case_description: str
    options: Dict[str, str]
    selected_answer: str
    correct_answer: Optional[str]
    is_correct: Optional[bool]
    confidence_score: float
    retrieval_results: List[RetrievalResult]
    reasoning: AnswerSelection
    pipeline_metadata: Dict


def merge_and_deduplicate_results(
    all_results: List[tuple],
    top_k: int = 25
) -> List[tuple]:
    """
    Helper function to merge and deduplicate retrieval results from multiple queries.
    
    Args:
        all_results: List of (RetrievalResult, score) tuples from multiple queries
        top_k: Number of results to return
        
    Returns:
        Deduplicated and sorted list of (RetrievalResult, score) tuples
    """
    seen_docs = set()
    merged = []
    
    for result, score in all_results:
        doc_id = (
            result.document.metadata.get('guideline_id', ''),
            result.document.metadata.get('chunk_index', 0)
        )
        
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            merged.append((result, score))
    
    # Sort by score descending
    merged.sort(key=lambda x: x[1], reverse=True)
    
    return merged[:top_k]


class MultiStageRAGPipeline:
    """
    Complete multi-stage RAG pipeline for medical Q&A.
    
    Integrates:
    - Query understanding
    - Multi-stage retrieval (FAISS → BM25 → Reranking)
    - Medical reasoning
    - Answer selection
    """
    
    def __init__(
        self,
        faiss_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        embedding_model: EmbeddingModel,
        retrieval_config: Optional[Dict] = None,
        use_improvements: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            faiss_store: FAISS vector store
            bm25_retriever: BM25 retriever
            embedding_model: Embedding model
            retrieval_config: Optional configuration for multi-stage retriever
            use_improvements: Use Day 5 improvements (default: True)
        """
        self.faiss_store = faiss_store
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
        self.use_improvements = use_improvements and IMPROVEMENTS_AVAILABLE
        
        # Initialize components
        self.query_understanding = MedicalQueryUnderstanding()
        
        # Initialize Ollama LLM model if configured
        llm_model = None
        try:
            from models.ollama_model import OllamaModel
            from utils.config_loader import load_config
            config_loader = load_config("config/pipeline_config.yaml")
            model_config = config_loader.get_model_config()
            llm_config = model_config.get('llm', {})
            
            if llm_config.get('use_llm', True):
                try:
                    llm_model = OllamaModel(
                        model_name=llm_config.get('model_name', 'llama3.1:8b'),
                        ollama_url=llm_config.get('ollama_url', 'http://localhost:11434/api/generate'),
                        temperature=llm_config.get('temperature', 0.1),
                        max_tokens=llm_config.get('max_tokens', 512)
                    )
                    print(f"[INFO] Ollama LLM initialized: {llm_config.get('model_name', 'llama3.1:8b')}")
                except Exception as e:
                    print(f"[WARN] Ollama initialization failed: {e}")
                    print("[INFO] Continuing without LLM enhancement")
                    llm_model = None
        except Exception as e:
            print(f"[WARN] Could not initialize LLM provider: {e}")
            llm_model = None
        
        # Day 7 Phase 1: Pass embedding model and LLM to reasoning engine
        self.reasoning_engine = MedicalReasoningEngine(embedding_model=embedding_model, llm_model=llm_model)
        
        # Day 5: Initialize improvement modules
        if self.use_improvements:
            self.query_enhancer = MedicalQueryEnhancer()
            self.structured_reasoner = StructuredMedicalReasoner()
            self.confidence_calibrator = ConfidenceCalibrator(temperature=1.0)  # Day 6: Less conservative
            self.specialty_adapter = SpecialtyAdapter()
            print("[INFO] Day 5 improvements enabled")
        else:
            self.query_enhancer = None
            self.structured_reasoner = None
            self.confidence_calibrator = None
            self.specialty_adapter = None
        
        # Day 6: Initialize optimization modules
        if DAY6_IMPROVEMENTS_AVAILABLE:
            self.symptom_extractor = EnhancedSymptomExtractor()
            self.obgyn_handler = OBGYNHandler()
            print("[INFO] Day 6 improvements enabled")
        else:
            self.symptom_extractor = None
            self.obgyn_handler = None
        
        # Day 7: Initialize reasoning improvement modules (Fixes 1-6)
        if DAY7_FIXES_AVAILABLE:
            self.multi_query_expander = MultiQueryExpander(llm_model=llm_model)
            self.enhanced_reasoning = EnhancedReasoningPipeline(llm_model=llm_model)
            self.guideline_reranker = GuidelinePrioritizationReranker()
            print("[INFO] Day 7 reasoning fixes enabled (Multi-Query, Uncertainty, Minimal Context, Differential Dx)")
        else:
            self.multi_query_expander = None
            self.enhanced_reasoning = None
            self.guideline_reranker = None
        
        # Day 7+: Initialize critical fix modules
        if DAY7_FIXES_AVAILABLE:
            self.multi_query_expander = MultiQueryExpander(llm_model=llm_model)
            self.symptom_injector = SymptomSynonymInjector()
            self.guideline_reranker = GuidelineReranker()
            self.enhanced_reasoner = EnhancedMedicalReasoner(llm_model=llm_model)
            print("[INFO] Day 7+ critical fixes enabled (multi-query, symptom injection, reranking, enhanced reasoning)")
        else:
            self.multi_query_expander = None
            self.symptom_injector = None
            self.guideline_reranker = None
            self.enhanced_reasoner = None
        
        # Day 7+ Step Fixes: Clinical feature extraction, context pruning, deterministic reasoning
        if DAY7_STEP_FIXES_AVAILABLE:
            self.clinical_feature_extractor = ClinicalFeatureExtractor()
            self.context_pruner = ContextPruner(max_paragraphs=3)  # Step 2: Top 3 only
            self.question_keyword_extractor = QuestionKeywordExtractor()
            self.deterministic_reasoner = DeterministicReasoner(llm_model=llm_model)
            print("[INFO] Day 7+ step fixes enabled (clinical extraction, context pruning, deterministic reasoning)")
        else:
            self.clinical_feature_extractor = None
            self.context_pruner = None
            self.question_keyword_extractor = None
            self.deterministic_reasoner = None
        
        # Day 8+ Advanced Reasoning: Structured 6-step reasoning, Safety, Hallucination detection
        if DAY8_ADVANCED_FIXES_AVAILABLE:
            self.structured_reasoner_v2 = StructuredMedicalReasonerV2(llm_model=llm_model)
            self.safety_verifier = MedicalSafetyVerifier()
            self.hallucination_detector = HallucinationDetector()
            self.terminology_normalizer = TerminologyNormalizer()
            print("[INFO] Day 8+ advanced fixes enabled (structured reasoning, safety, hallucination detection)")
        else:
            self.structured_reasoner_v2 = None
            self.safety_verifier = None
            self.hallucination_detector = None
            self.terminology_normalizer = None
        
        # Initialize multi-stage retriever with improved parameters for better accuracy
        retrieval_config = retrieval_config or {}
        concept_first = retrieval_config.get('concept_first', False)
        if concept_first:
            self.retriever = ConceptFirstRetriever(
                bm25=bm25_retriever,
                faiss_store=faiss_store,
                bm25_k=retrieval_config.get('stage2_k', 30),
                embed_k=retrieval_config.get('stage1_k', 50),
                stage3_k=retrieval_config.get('stage3_k', 25),
                top_k=retrieval_config.get('stage3_k', 25),
                cross_encoder_name=retrieval_config.get('cross_encoder_name', 'cross-encoder/nli-deberta-v3-base'),
            )
            print("[INFO] Concept-first retrieval enabled (BM25 → concept expansion → embedding → rerank)")
        else:
            self.retriever = MultiStageRetriever(
                faiss_store=faiss_store,
                bm25_retriever=bm25_retriever,
                embedding_model=embedding_model,
                stage1_k=retrieval_config.get('stage1_k', 50),  # Increased from 20 to 50 for better recall
                stage2_k=retrieval_config.get('stage2_k', 30),  # Increased from 10 to 30
                stage3_k=retrieval_config.get('stage3_k', 25),  # Increased from 5 to 25
                stage1_weight=retrieval_config.get('stage1_weight', 0.5),  # Higher weight for semantic search
                stage2_weight=retrieval_config.get('stage2_weight', 0.2),  # Lower weight for BM25
                stage3_weight=retrieval_config.get('stage3_weight', 0.3),  # Balanced weight for reranking
                cross_encoder_name=retrieval_config.get('cross_encoder_name', 'cross-encoder/nli-deberta-v3-base'),
            )
    
    def _is_complex_question(self, question: str, case_description: str) -> bool:
        """Check if question is complex and would benefit from Tree-of-Thought reasoning."""
        complex_indicators = [
            'with', 'and', 'complicated by', 'associated with', 'in addition',
            'multiple', 'several', 'various', 'combination', 'along with'
        ]
        full_text = f"{case_description} {question}".lower()
        return any(indicator in full_text for indicator in complex_indicators)
    
    def answer_question(
        self,
        question_id: str,
        case_description: str,
        question: str,
        options: Dict[str, str],
        correct_answer: Optional[str] = None
    ) -> PipelineResult:
        """
        Answer a medical question using the complete pipeline.
        
        Args:
            question_id: Unique identifier for the question
            case_description: Patient case description
            question: The question to answer
            options: Answer options {A: "text", B: "text", ...}
            correct_answer: Ground truth answer (for evaluation)
            
        Returns:
            PipelineResult with answer and reasoning
        """
        pipeline_start = time.time()
        
        # Step 1: Understand query
        full_query = f"{case_description} {question}"
        query_understanding = self.query_understanding.understand(full_query)
        
        # STEP 1 FIX: Clinical Feature Extraction (symptoms, vitals, risk factors, labs)
        extracted_features = None
        if DAY7_STEP_FIXES_AVAILABLE and self.clinical_feature_extractor:
            extracted_features = self.clinical_feature_extractor.extract(case_description, question)
            # Use extracted features to enhance query
            feature_query = extracted_features.to_query_string()
            if feature_query:
                full_query = f"{full_query} {feature_query}"
        
        # Day 6: Extract symptoms for better query understanding and retrieval
        symptom_terms = ""
        critical_symptoms = None
        all_symptoms = None
        if self.symptom_extractor:
            symptom_result = self.symptom_extractor.extract_symptoms(case_description, question)
            symptom_terms = self.symptom_extractor.get_symptom_query_terms(case_description, question)
            # Extract symptom lists for retrieval boosting
            critical_symptoms = symptom_result.critical_symptoms
            all_symptoms = symptom_result.all_symptom_terms
        
        # Day 6: Check for OB/GYN and apply specialized handling
        obgyn_enhancement = None
        obgyn_retrieval_params = None
        if self.obgyn_handler:
            is_obgyn, obgyn_confidence = self.obgyn_handler.detect_obgyn_query(question, case_description)
            if is_obgyn:
                obgyn_enhancement = self.obgyn_handler.enhance_obgyn_query(question, case_description)
                obgyn_retrieval_params = self.obgyn_handler.get_obgyn_retrieval_params()
                print(f"[INFO] OB/GYN query detected (confidence: {obgyn_confidence:.2f})")
        
        # Day 5: Enhance query for better retrieval
        enhanced_query = None
        if self.use_improvements and self.query_enhancer:
            enhanced_query_obj = self.query_enhancer.enhance(full_query)
            enhanced_query = self.query_enhancer.get_enhanced_query_for_retrieval(full_query)
            
            # Day 6: Add symptom terms to enhanced query
            if symptom_terms:
                enhanced_query = f"{enhanced_query} {symptom_terms}"
            
            # PubMedBERT: Add guideline-specific terms if query mentions a guideline (more aggressive)
            if 'guideline' in question.lower() or 'according to' in question.lower():
                # Extract potential guideline name from question
                guideline_patterns = [
                    r"'([^']+)' guideline",
                    r'"([^"]+)" guideline',
                    r'guideline[,\s]+([^,\.]+)',
                    r'according to the ([^,\.]+)',
                    r'based on the ([^,\.]+) guideline',
                    r'per the ([^,\.]+) guideline'
                ]
                for pattern in guideline_patterns:
                    match = re.search(pattern, question, re.IGNORECASE)
                    if match:
                        guideline_name = match.group(1).strip()
                        enhanced_query = f"{enhanced_query} {guideline_name}"
                        # Also add variations
                        enhanced_query = f"{enhanced_query} {guideline_name.replace(' ', '_')} {guideline_name.replace('_', ' ')}"
                        break
            
            # Day 7 Phase 3: Medical concept expansion
            try:
                from improvements.medical_concept_expander import MedicalConceptExpander
                concept_expander = MedicalConceptExpander()
                enhanced_query = concept_expander.expand(enhanced_query, max_expansions=3)
            except Exception:
                pass
        
        # Day 6: Apply OB/GYN-specific query enhancement
        if obgyn_enhancement and obgyn_enhancement.specialty_confidence > 0.3:
            enhanced_query = obgyn_enhancement.enhanced_query
        
        # Day 5: Detect specialty and adapt
        specialty_adaptation = None
        if self.use_improvements and self.specialty_adapter:
            specialty_adaptation = self.specialty_adapter.detect_specialty(question, case_description)
            if enhanced_query and not obgyn_enhancement:
                # Further enhance with specialty-specific terms (unless OB/GYN handled separately)
                enhanced_query = specialty_adaptation.adapted_query
        
        # Step 2: Multi-stage retrieval
        retrieval_start = time.time()
        
        # Improved retrieval parameters - adaptive based on query complexity
        # For queries mentioning specific guidelines, use smaller top_k (more precise)
        # For general queries, use larger top_k (better recall)
        guideline_mentioned = None
        if 'guideline' in question.lower() or 'according to' in question.lower():
            # Extract guideline name to check if specific guideline is mentioned
            guideline_patterns = [
                r"'([^']+)' guideline",
                r'"([^"]+)" guideline',
                r'according to the ([^,\.]+)',
                r'based on the ([^,\.]+) guideline'
            ]
            for pattern in guideline_patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    guideline_mentioned = match.group(1).strip()
                    break
        
        # Adaptive top_k: smaller for specific guideline queries (faster, more precise)
        # Larger for general queries (better recall)
        if guideline_mentioned:
            retrieval_top_k = 20  # Smaller for specific guideline queries
        else:
            retrieval_top_k = 25  # Moderate for general queries (balance between recall and performance)
        
        retrieval_threshold = -1.0  # Very lenient for maximum recall
        if obgyn_retrieval_params:
            retrieval_top_k = obgyn_retrieval_params.get('stage3_k', 10)
            retrieval_threshold = obgyn_retrieval_params.get('min_score_threshold', -1.0)
        
        # FIX 2: Symptom Keyword Injection - expand symptoms with synonyms
        if DAY7_FIXES_AVAILABLE and self.symptom_injector:
            try:
                symptom_injection = self.symptom_injector.get_injection_string(case_description, question)
                if symptom_injection:
                    if enhanced_query:
                        enhanced_query = f"{enhanced_query} {symptom_injection}"
                    else:
                        enhanced_query = f"{full_query} {symptom_injection}"
            except Exception as e:
                print(f"[WARN] Symptom injection failed: {e}")
        
        # Day 7 Phase 3: Query decomposition for complex queries
        use_decomposition = False
        if self.use_improvements:
            # Check if query is complex (multiple conditions, complications)
            complex_indicators = ['with', 'and', 'complicated by', 'associated with']
            if any(indicator in full_query.lower() for indicator in complex_indicators):
                try:
                    from improvements.query_decomposer import QueryDecomposer
                    decomposer = QueryDecomposer()
                    sub_queries = decomposer.decompose(question, case_description)
                    
                    if len(sub_queries) > 1:
                        # Retrieve for each sub-query
                        all_results = []
                        for sub_query in sub_queries[:3]:  # Limit to 3 sub-queries
                            sub_results = self.retriever.retrieve(
                                query=sub_query,
                                top_k=retrieval_top_k + 2,  # Get more for merging
                                min_score_threshold=retrieval_threshold - 0.1,
                                enhanced_query=sub_query
                            )
                            all_results.extend(sub_results)
                        
                        # Merge and deduplicate
                        seen_docs = set()
                        merged_results = []
                        for result in all_results:
                            doc_id = (
                                result.document.metadata.get('guideline_id'),
                                result.document.metadata.get('chunk_index')
                            )
                            if doc_id not in seen_docs:
                                seen_docs.add(doc_id)
                                merged_results.append(result)
                        
                        # Sort by score and take top k
                        merged_results.sort(key=lambda x: x.final_score, reverse=True)
                        retrieval_results = merged_results[:retrieval_top_k]
                        use_decomposition = True
                except Exception:
                    pass
        
        if not use_decomposition:
            # Fix 1: Multi-Query Expansion
            if DAY7_FIXES_AVAILABLE and self.multi_query_expander:
                try:
                    expanded = self.multi_query_expander.expand(question, case_description)
                    all_results = []
                    
                    # Retrieve for each query variant
                    for variant_query in expanded.all_queries[:4]:  # Use up to 4 query variants
                        variant_results = self.retriever.retrieve(
                            query=variant_query,
                            top_k=retrieval_top_k + 5,  # Get extra for merging
                            min_score_threshold=retrieval_threshold,
                            enhanced_query=variant_query,
                            critical_symptoms=critical_symptoms,
                            all_symptoms=all_symptoms
                        )
                        for r in variant_results:
                            all_results.append((r, r.final_score))
                    
                    # Merge and deduplicate results
                    if all_results:
                        merged = merge_and_deduplicate_results(all_results, retrieval_top_k)
                        # Convert back to RetrievalResult objects
                        retrieval_results = [r for r, score in merged]
                    else:
                        # Fallback to single query
                        retrieval_results = self.retriever.retrieve(
                            query=full_query,
                            top_k=retrieval_top_k,
                            min_score_threshold=retrieval_threshold,
                            enhanced_query=enhanced_query,
                            critical_symptoms=critical_symptoms,
                            all_symptoms=all_symptoms
                        )
                except Exception as e:
                    print(f"[WARN] Multi-query expansion failed: {e}")
                    retrieval_results = self.retriever.retrieve(
                        query=full_query,
                        top_k=retrieval_top_k,
                        min_score_threshold=retrieval_threshold,
                        enhanced_query=enhanced_query,
                        critical_symptoms=critical_symptoms,
                        all_symptoms=all_symptoms
                    )
            else:
                retrieval_results = self.retriever.retrieve(
                    query=full_query,
                    top_k=retrieval_top_k,
                    min_score_threshold=retrieval_threshold,
                    enhanced_query=enhanced_query,
                    critical_symptoms=critical_symptoms,
                    all_symptoms=all_symptoms
                )
        
        # FIX 3: Guideline Prioritization Reranker - boost documents with treatment/management terms
        if DAY7_FIXES_AVAILABLE and self.guideline_reranker and retrieval_results:
            try:
                retrieval_results = self.guideline_reranker.rerank_retrieval_results(
                    retrieval_results, full_query, case_description
                )
            except Exception as e:
                print(f"[WARN] Guideline reranking failed: {e}")
        
        retrieval_time = time.time() - retrieval_start
        
        # Day 7: Improve OB/GYN retrieval - lower threshold and better integration
        if obgyn_enhancement and obgyn_enhancement.specialty_confidence > 0.2:  # Lowered from 0.5
            # Check if we need to boost OB/GYN retrieval
            obgyn_docs_found = any(
                'obstetrics' in doc.metadata.get('category', '').lower() or
                'gynecology' in doc.metadata.get('category', '').lower() or
                'obgyn' in doc.metadata.get('guideline_id', '').lower()
                for r in retrieval_results
                for doc in [r.document]
            )
            
            # If no OB/GYN docs found or low scores, try to inject/boost
            if not obgyn_docs_found or all(r.final_score < 0.4 for r in retrieval_results):
                injected_guidelines = self.obgyn_handler.inject_obgyn_knowledge(question, case_description)
                if injected_guidelines:
                    print(f"[INFO] OB/GYN detected (confidence: {obgyn_enhancement.specialty_confidence:.2f}) - {len(injected_guidelines)} guidelines available")
                    # Boost OB/GYN-related documents in retrieval results
                    # Re-retrieve with OB/GYN-specific parameters if available
                    if hasattr(self.retriever, 'retrieve') and obgyn_enhancement.specialty_confidence > 0.3:
                        # Try retrieval with OB/GYN-enhanced query
                        obgyn_params = self.obgyn_handler.get_obgyn_retrieval_params()
                        enhanced_query_obgyn = obgyn_enhancement.enhanced_query
                        if enhanced_query_obgyn != full_query:
                            # Re-retrieve with OB/GYN-enhanced query
                            obgyn_results = self.retriever.retrieve(
                                query=enhanced_query_obgyn,
                                top_k=retrieval_top_k + 5,  # Get more candidates
                                min_score_threshold=0.0,  # Lower threshold for OB/GYN
                                enhanced_query=enhanced_query_obgyn
                            )
                            # Merge and deduplicate, prioritizing OB/GYN docs
                            obgyn_doc_ids = {id(r.document) for r in obgyn_results}
                            existing_doc_ids = {id(r.document) for r in retrieval_results}
                            
                            # Add OB/GYN docs that aren't already in results
                            for obgyn_result in obgyn_results:
                                if id(obgyn_result.document) not in existing_doc_ids:
                                    # Boost score for OB/GYN documents
                                    obgyn_result.final_score *= 1.2  # 20% boost
                                    retrieval_results.append(obgyn_result)
                            
                            # Re-sort by score
                            retrieval_results.sort(key=lambda x: x.final_score, reverse=True)
                            retrieval_results = retrieval_results[:retrieval_top_k]
        
        # Day 5: Adapt retrieval results for specialty
        if self.use_improvements and self.specialty_adapter and specialty_adaptation:
            retrieved_docs = [r.document for r in retrieval_results]
            adapted_docs = self.specialty_adapter.adapt_retrieval(
                full_query,
                specialty_adaptation.detected_specialty,
                retrieved_docs
            )
            # Rebuild retrieval results with adapted order
            doc_map = {id(doc): (doc, result) for doc, result in zip(retrieved_docs, retrieval_results)}
            retrieval_results = [doc_map[id(doc)][1] for doc in adapted_docs if id(doc) in doc_map]
        
        # Extract documents from retrieval results
        retrieved_documents = [r.document for r in retrieval_results]
        
        # STEP 2 FIX: Aggressive Context Pruning - keep only top 3 relevant paragraphs
        pruned_context_text = None
        if DAY7_STEP_FIXES_AVAILABLE and self.context_pruner and extracted_features:
            # Get symptoms and diseases from extracted features
            symptoms = extracted_features.symptoms if extracted_features else []
            diseases = extracted_features.chronic_diseases if extracted_features else []
            
            # Get question keywords
            question_keywords = []
            if self.question_keyword_extractor:
                question_keywords = self.question_keyword_extractor.extract(question)
            
            # Prune context to top 3 most relevant paragraphs
            pruned = self.context_pruner.prune(
                retrieved_documents, symptoms, diseases, question_keywords
            )
            
            # Use pruned documents for reasoning
            if pruned.pruned_documents:
                retrieved_documents = pruned.pruned_documents
                pruned_context_text = self.context_pruner.prune_to_text(
                    retrieved_documents, symptoms, diseases, question_keywords, max_chars=3000
                )
        
        # Step 3: Medical reasoning
        reasoning_start = time.time()
        
        # === 3-STAGE HYBRID REASONING PIPELINE ===
        # Stage 1: CoT (Primary) - Fast, accurate for most questions
        # Stage 2: ToT (Complex) - Multi-branch reasoning for complex cases  
        # Stage 3: Structured (Fallback) - Rule-based + LLM for edge cases
        
        reasoning_method_tag = "Baseline"
        answer_selection = None
        
        # Import necessary classes
        from reasoning.medical_reasoning import AnswerSelection, ReasoningStep, EvidenceMatch
        
        # Determine question complexity
        is_complex = self._is_complex_question(question, case_description)
        
        # Try Stage 1: Chain-of-Thought (Primary method)
        try:
            # Use deterministic CoT if available (most optimized)
            if DAY7_STEP_FIXES_AVAILABLE and self.deterministic_reasoner and pruned_context_text:
                det_critical_symptoms = []
                if extracted_features:
                    det_critical_symptoms = extracted_features.symptoms[:5]
                    det_critical_symptoms.extend(extracted_features.get_critical_findings()[:3])
                
                det_result = self.deterministic_reasoner.reason(
                    question=question,
                    case_description=case_description,
                    options=options,
                    context=pruned_context_text,
                    critical_symptoms=det_critical_symptoms,
                    extracted_features=extracted_features.__dict__ if extracted_features else None
                )
                
                det_reasoning_steps = [
                    ReasoningStep(
                        step_number=1,
                        description="Chain-of-Thought Reasoning",
                        reasoning=det_result.reasoning_trace[:500] if det_result.reasoning_trace else "CoT reasoning applied",
                        evidence_used=["Pruned context (top paragraphs)"]
                    )
                ]
                
                supporting_guidelines_list = list(set([
                    doc.metadata.get('guideline_id', '')
                    for doc in retrieved_documents
                    if doc.metadata.get('guideline_id')
                ]))
                
                answer_selection = AnswerSelection(
                    selected_answer=det_result.selected_answer,
                    confidence_score=det_result.confidence,
                    reasoning_steps=det_reasoning_steps,
                    evidence_matches={},
                    rationale=f"CoT reasoning. Verification: {'passed' if det_result.verification_passed else 'failed'}",
                    supporting_guidelines=supporting_guidelines_list
                )
                reasoning_method_tag = "CoT"
                
                # If complex question detected AND CoT confidence is not very high, escalate to ToT
                if is_complex and det_result.confidence < 0.75:
                    print(f"[INFO] Complex question detected with moderate confidence ({det_result.confidence:.2f}), escalating to Tree-of-Thought")
                    answer_selection = None  # Clear to trigger ToT
            else:
                # Fallback to standard reasoning engine CoT
                answer_selection = self.reasoning_engine.reason_and_select_answer(
                    question=question,
                    case_description=case_description,
                    options=options,
                    retrieved_contexts=retrieved_documents,
                    correct_answer=correct_answer
                )
                reasoning_method_tag = "CoT"
                
                # Escalate to ToT if complex and confidence not high
                if is_complex and answer_selection.confidence_score < 0.75:
                    print(f"[INFO] Complex question detected, escalating to Tree-of-Thought")
                    answer_selection = None
        except Exception as e:
            print(f"[WARN] CoT reasoning failed: {e}, will try ToT or Structured")
            answer_selection = None
        
        # Try Stage 2: Tree-of-Thought (for complex questions or CoT failures)
        if answer_selection is None and self.reasoning_engine.llm_model:
            try:
                print(f"[INFO] Using Tree-of-Thought reasoning for complex case")
                
                # Use structured reasoner to get evidence, then enhance with ToT
                if self.structured_reasoner:
                    structured_result = self.structured_reasoner.reason(
                        question=question,
                        case_description=case_description,
                        options=options,
                        retrieved_contexts=retrieved_documents,
                        query_understanding=query_understanding
                    )
                    
                    # Convert evidence_scores to evidence_matches format
                    evidence_matches = {}
                    for label, evidence_score in structured_result.evidence_scores.items():
                        evidence_matches[label] = EvidenceMatch(
                            option_label=label,
                            option_text=evidence_score.option_text,
                            supporting_evidence=evidence_score.evidence_sources,
                            contradicting_evidence=[],
                            evidence_strength=evidence_score.total_score,
                            match_type='direct' if evidence_score.direct_evidence_score > 0 else 'inferred'
                        )
                    
                    # Apply Tree-of-Thought enhancement
                    tot_answer, tot_confidence = self.reasoning_engine._llm_enhance_answer_selection(
                        question=question,
                        case_description=case_description,
                        options=options,
                        retrieved_contexts=retrieved_documents,
                        evidence_matches=evidence_matches,
                        current_answer=structured_result.selected_answer,
                        current_confidence=structured_result.confidence_score,
                        use_tot=True  # Force ToT mode
                    )
                    
                    supporting_guidelines_list = structured_result.supporting_guidelines
                    
                    answer_selection = AnswerSelection(
                        selected_answer=tot_answer,
                        confidence_score=tot_confidence,
                        reasoning_steps=[
                            ReasoningStep(
                                step_number=1,
                                description="Tree-of-Thought Multi-Branch Reasoning",
                                reasoning=f"ToT explored multiple reasoning paths for complex case. Selected: {tot_answer}",
                                evidence_used=[f"ToT with {len(retrieved_documents[:5])} documents"]
                            )
                        ],
                        evidence_matches=evidence_matches,
                        rationale="Tree-of-Thought multi-branch reasoning for complex medical case",
                        supporting_guidelines=supporting_guidelines_list
                    )
                    reasoning_method_tag = "ToT"
            except Exception as e:
                print(f"[WARN] ToT reasoning failed: {e}, falling back to Structured")
                answer_selection = None
        
        # Stage 3: Structured Medical Reasoning (Fallback with LLM enhancement)
        if answer_selection is None and self.use_improvements and self.structured_reasoner:
            print(f"[INFO] Using Structured Medical reasoning as fallback")
            structured_result = self.structured_reasoner.reason(
                question=question,
                case_description=case_description,
                options=options,
                retrieved_contexts=retrieved_documents,
                query_understanding=query_understanding
            )
            
            # Convert to AnswerSelection format
            reasoning_steps = [
                ReasoningStep(
                    step_number=step['step'],
                    description=step['name'],
                    reasoning=step.get('description', ''),
                    evidence_used=[]
                )
                for step in structured_result.reasoning_steps
            ]
            
            # Calibrate confidence if available
            final_confidence = structured_result.confidence_score
            if self.confidence_calibrator and structured_result.selected_answer in structured_result.evidence_scores:
                evidence_strength = structured_result.evidence_scores[structured_result.selected_answer].total_score
                evidence_count = len(structured_result.evidence_scores[structured_result.selected_answer].evidence_sources)
                calibration_result = self.confidence_calibrator.calibrate(
                    raw_confidence=structured_result.confidence_score,
                    evidence_strength=evidence_strength,
                    evidence_count=evidence_count,
                    uncertainty=structured_result.uncertainty_estimate
                )
                final_confidence = calibration_result.calibrated_confidence
            
            answer_selection = AnswerSelection(
                selected_answer=structured_result.selected_answer,
                confidence_score=final_confidence,
                reasoning_steps=reasoning_steps,
                evidence_matches={},
                rationale=structured_result.rationale,
                supporting_guidelines=structured_result.supporting_guidelines
            )
            reasoning_method_tag = "Structured"
        
        # Ultimate fallback if all stages failed
        if answer_selection is None:
            answer_selection = self.reasoning_engine.reason_and_select_answer(
                question=question,
                case_description=case_description,
                options=options,
                retrieved_contexts=retrieved_documents,
                correct_answer=correct_answer
            )
            reasoning_method_tag = "Fallback"
        
        # === END 3-STAGE HYBRID REASONING ===
        
        reasoning_time = time.time() - reasoning_start
        
        # ORIGINAL CODE CONTINUES BELOW (Day 7+ enhancements)
        # STEP 3 FIX: Use Deterministic Reasoner (no ToT) if available
        use_deterministic_old = False  # Disabled since we use new 3-stage approach
        
        reasoning_method_tag_old = reasoning_method_tag  # Keep track
        
        # STEP 3 & 4 FIX: Deterministic CoT reasoning with symptom importance ranking
        if use_deterministic_old:
            # Get critical symptoms from extracted features
            det_critical_symptoms = []
            if extracted_features:
                det_critical_symptoms = extracted_features.symptoms[:5]
                det_critical_symptoms.extend(extracted_features.get_critical_findings()[:3])
            
            # Run deterministic reasoning
            det_result = self.deterministic_reasoner.reason(
                question=question,
                case_description=case_description,
                options=options,
                context=pruned_context_text,
                critical_symptoms=det_critical_symptoms,
                extracted_features=extracted_features.__dict__ if extracted_features else None
            )
            
            # Convert to AnswerSelection format
            from reasoning.medical_reasoning import AnswerSelection, ReasoningStep
            
            det_reasoning_steps = [
                ReasoningStep(
                    step_number=1,
                    description="Deterministic Chain-of-Thought Reasoning",
                    reasoning=det_result.reasoning_trace[:500] if det_result.reasoning_trace else "Deterministic reasoning applied",
                    evidence_used=["Pruned context (top 3 paragraphs)"]
                )
            ]
            
            # Add symptom ranking step if available
            if det_result.symptom_ranking:
                det_reasoning_steps.append(ReasoningStep(
                    step_number=2,
                    description="Symptom Importance Ranking",
                    reasoning=f"Top symptoms: {', '.join(det_result.symptom_ranking.top_symptoms[:3])}. Missed critical: {det_result.symptom_ranking.missed_critical}",
                    evidence_used=["Clinical feature extraction"]
                ))
            
            # FIX: Extract supporting_guidelines from retrieved_documents
            supporting_guidelines_list = list(set([
                doc.metadata.get('guideline_id', '')
                for doc in retrieved_documents
                if doc.metadata.get('guideline_id')
            ]))
            
            answer_selection = AnswerSelection(
                selected_answer=det_result.selected_answer,
                confidence_score=det_result.confidence,
                reasoning_steps=det_reasoning_steps,
                evidence_matches={},
                rationale=f"Deterministic CoT reasoning with verification. Retries: {det_result.retry_count}. Verification: {'passed' if det_result.verification_passed else 'failed'}",
                supporting_guidelines=supporting_guidelines_list
            )
            reasoning_method_tag = "Deterministic"
        
        # Day 5: Use structured reasoning if available (fallback if deterministic not used)
        elif self.use_improvements and self.structured_reasoner:
            structured_result = self.structured_reasoner.reason(
                question=question,
                case_description=case_description,
                options=options,
                retrieved_contexts=retrieved_documents,
                query_understanding=query_understanding
            )
            reasoning_method_tag = "Structured"
            
            # Check if answer is "Cannot answer from the provided context."
            if structured_result.selected_answer == "Cannot answer from the provided context.":
                # Convert to AnswerSelection format
                from reasoning.medical_reasoning import AnswerSelection, ReasoningStep
                reasoning_steps = [
                    ReasoningStep(
                        step_number=step['step'],
                        description=step['name'],
                        reasoning=step.get('description', ''),
                        evidence_used=[]
                    )
                    for step in structured_result.reasoning_steps
                ]
                
                # FIX: Extract supporting_guidelines from retrieved_documents
                supporting_guidelines_list = list(set([
                    doc.metadata.get('guideline_id', '')
                    for doc in retrieved_documents
                    if doc.metadata.get('guideline_id')
                ]))
                
                answer_selection = AnswerSelection(
                    selected_answer=structured_result.selected_answer,
                    confidence_score=0.0,
                    reasoning_steps=reasoning_steps,
                    evidence_matches={},
                    rationale=structured_result.rationale,
                    supporting_guidelines=supporting_guidelines_list
                )
            else:
                # Day 5: Calibrate confidence
                if self.confidence_calibrator:
                    # Check if selected_answer exists in evidence_scores
                    if structured_result.selected_answer in structured_result.evidence_scores:
                        evidence_strength = structured_result.evidence_scores[
                            structured_result.selected_answer
                        ].total_score
                        evidence_count = len(structured_result.evidence_scores[
                            structured_result.selected_answer
                        ].evidence_sources)
                    else:
                        # Fallback if answer not in evidence_scores
                        evidence_strength = 0.0
                        evidence_count = 0
                    
                    calibration_result = self.confidence_calibrator.calibrate(
                        raw_confidence=structured_result.confidence_score,
                        evidence_strength=evidence_strength,
                        evidence_count=evidence_count,
                        uncertainty=structured_result.uncertainty_estimate
                    )
                    
                    # Use calibrated confidence
                    final_confidence = calibration_result.calibrated_confidence
                else:
                    final_confidence = structured_result.confidence_score
                
                # Convert to AnswerSelection format for compatibility
                from reasoning.medical_reasoning import AnswerSelection, ReasoningStep
                reasoning_steps = [
                    ReasoningStep(
                        step_number=step['step'],
                        description=step['name'],
                        reasoning=step.get('description', ''),
                        evidence_used=[]
                    )
                    for step in structured_result.reasoning_steps
                ]
                
                # Day 5: Add specialty-specific reasoning
                if self.specialty_adapter and specialty_adaptation:
                    reasoning_steps = self.specialty_adapter.adapt_reasoning(
                        specialty_adaptation.detected_specialty,
                        [{'step': s.step_number, 'name': s.description, 'description': s.reasoning}
                         for s in reasoning_steps]
                    )
                    reasoning_steps = [
                        ReasoningStep(
                            step_number=s['step'],
                            description=s['name'],
                            reasoning=s.get('description', ''),
                            evidence_used=[]
                        )
                        for s in reasoning_steps
                    ]
                
                answer_selection = AnswerSelection(
                    selected_answer=structured_result.selected_answer,
                    confidence_score=final_confidence,
                    reasoning_steps=reasoning_steps,
                    evidence_matches={},  # Structured result has different format
                    rationale=structured_result.rationale,
                    supporting_guidelines=structured_result.supporting_guidelines
                )
                
                # CRITICAL: Apply LLM enhancement with chain-of-thought reasoning
                # Use LLM for ALL cases (not just moderate confidence) to leverage chain-of-thought
                if self.reasoning_engine.llm_model:
                    # Convert evidence_scores to evidence_matches format for LLM
                    evidence_matches = {}
                    for label, evidence_score in structured_result.evidence_scores.items():
                        from reasoning.medical_reasoning import EvidenceMatch
                        evidence_matches[label] = EvidenceMatch(
                            option_label=label,
                            option_text=evidence_score.option_text,
                            supporting_evidence=evidence_score.evidence_sources,
                            contradicting_evidence=[],
                            evidence_strength=evidence_score.total_score,
                            match_type='direct' if evidence_score.direct_evidence_score > 0 else 'inferred'
                        )
                    
                    # IMPROVED: Use LLM enhancement more frequently with all improvements
                    # Use LLM for most cases to leverage enhanced chain-of-thought reasoning
                    # Expanded range to catch errors and improve accuracy
                    if 0.15 < final_confidence < 0.98:  # Wider range to use LLM more often
                        # Check if question is complex and would benefit from Tree-of-Thought
                        is_complex = self._is_complex_question(question, case_description)
                        use_tot = is_complex  # Use ToT for complex questions
                        
                        llm_enhanced_answer, llm_confidence = self.reasoning_engine._llm_enhance_answer_selection(
                            question=question,
                            case_description=case_description,
                            options=options,
                            retrieved_contexts=retrieved_documents,
                            evidence_matches=evidence_matches,
                            current_answer=structured_result.selected_answer,
                            current_confidence=final_confidence,
                            use_tot=use_tot
                        )
                        # IMPROVED: Use LLM answer more aggressively for better accuracy
                        # Use LLM answer if:
                        # 1. LLM confidence is higher, OR
                        # 2. Current confidence is high but might be wrong (let LLM verify), OR
                        # 3. LLM answer is different (might have caught an error)
                        should_use_llm = (
                            llm_confidence > final_confidence or
                            (final_confidence > 0.7 and llm_confidence > 0.5) or
                            (llm_enhanced_answer != structured_result.selected_answer and llm_confidence > 0.4)
                        )
                        
                        if should_use_llm:
                            answer_selection.selected_answer = llm_enhanced_answer
                            answer_selection.confidence_score = llm_confidence
                            reasoning_method = "Tree-of-Thought" if use_tot else "Enhanced Chain-of-Thought"
                            reasoning_method_tag = "ToT" if use_tot else "CoT"
                            answer_selection.reasoning_steps.append(
                                ReasoningStep(
                                    step_number=6,
                                    description=f"LLM {reasoning_method} Enhancement",
                                    reasoning=f"LLM enhanced answer using {reasoning_method} with multi-document synthesis, critical symptom weighting, and step-by-step reasoning. Original: {structured_result.selected_answer}, Enhanced: {llm_enhanced_answer}",
                                    evidence_used=[f"LLM {reasoning_method} reasoning with {len(retrieved_documents[:5])} document synthesis"]
                                )
                            )
        else:
            # Fallback to original reasoning
            answer_selection = self.reasoning_engine.reason_and_select_answer(
                question=question,
                case_description=case_description,
                options=options,
                retrieved_contexts=retrieved_documents,
                correct_answer=correct_answer
            )
            reasoning_method_tag = "CoT"
        
        reasoning_time = time.time() - reasoning_start
        
        # Day 7+: Apply enhanced reasoning improvements (Fixes 4, 5, 6)
        if DAY7_FIXES_AVAILABLE and self.enhanced_reasoner:
            try:
                from reasoning.medical_reasoning import ReasoningStep
                
                initial_answer = answer_selection.selected_answer
                initial_confidence = answer_selection.confidence_score
                
                # FIX 4: Forced Uncertainty Step - analyze potential errors
                uncertainty = self.enhanced_reasoner.analyze_uncertainty(
                    question=question,
                    case_description=case_description,
                    options=options,
                    selected_answer=initial_answer,
                    confidence=initial_confidence,
                    retrieved_contexts=retrieved_documents
                )
                
                # Adjust confidence based on uncertainty analysis
                adjusted_confidence = self.enhanced_reasoner.adjust_confidence_for_uncertainty(
                    initial_confidence, uncertainty
                )
                
                # FIX 5: Minimal Context Re-evaluation - verify with top 2 paragraphs only
                minimal_result = self.enhanced_reasoner.reevaluate_with_minimal_context(
                    question=question,
                    case_description=case_description,
                    options=options,
                    original_answer=initial_answer,
                    retrieved_contexts=retrieved_documents,
                    top_k=2
                )
                
                # Apply minimal context confidence adjustment
                adjusted_confidence *= minimal_result.confidence_adjustment
                
                # FIX 6: Stepwise Differential Diagnosis (for complex cases)
                is_complex = self._is_complex_question(question, case_description)
                differential_answer = initial_answer
                
                if is_complex or not minimal_result.answers_agree:
                    differential_result = self.enhanced_reasoner.stepwise_differential_diagnosis(
                        question=question,
                        case_description=case_description,
                        options=options,
                        retrieved_contexts=retrieved_documents
                    )
                    differential_answer = differential_result.selected_answer
                
                # Determine final answer based on agreement
                final_answer = initial_answer
                if minimal_result.answers_agree and differential_answer == initial_answer:
                    # All methods agree - high confidence
                    final_answer = initial_answer
                    adjusted_confidence = min(0.95, adjusted_confidence * 1.1)
                elif minimal_result.answers_agree:
                    # Minimal context agrees but differential differs - use original with slight reduction
                    final_answer = initial_answer
                    adjusted_confidence *= 0.9
                elif differential_answer == minimal_result.minimal_context_answer:
                    # Minimal context and differential agree but differ from original - consider switching
                    if adjusted_confidence < 0.6:
                        final_answer = minimal_result.minimal_context_answer
                        adjusted_confidence = min(0.7, adjusted_confidence + 0.1)
                
                # Update answer selection
                answer_selection.selected_answer = final_answer
                answer_selection.confidence_score = max(0.05, min(0.95, adjusted_confidence))
                
                # Add reasoning step documenting the enhancements
                enhancement_details = []
                if uncertainty.potential_errors:
                    enhancement_details.append(f"Potential errors identified: {len(uncertainty.potential_errors)}")
                if not minimal_result.answers_agree:
                    enhancement_details.append(f"Minimal context suggests: {minimal_result.minimal_context_answer}")
                enhancement_details.append(f"Confidence adjusted: {initial_confidence:.2%} → {answer_selection.confidence_score:.2%}")
                
                answer_selection.reasoning_steps.append(ReasoningStep(
                    step_number=7,
                    description="Enhanced Reasoning (Uncertainty + Minimal Context + Differential)",
                    reasoning=" | ".join(enhancement_details),
                    evidence_used=["Uncertainty analysis", "Minimal context verification", "Differential diagnosis"]
                ))
                
                reasoning_method_tag = "Enhanced"
                
            except Exception as e:
                print(f"[WARN] Enhanced reasoning failed: {e}")
        
        # Day 8+: Apply Advanced Reasoning with Structured 6-Step, Safety, and Hallucination Detection
        if DAY8_ADVANCED_FIXES_AVAILABLE and self.structured_reasoner_v2:
            try:
                from reasoning.medical_reasoning import ReasoningStep
                
                # A. Apply Structured 6-Step Medical Reasoning as verification
                structured_v2_result = self.structured_reasoner_v2.reason(
                    question=question,
                    case_description=case_description,
                    options=list(options.values()),
                    retrieved_contexts=retrieval_results,
                    extracted_features=extracted_features
                )
                
                # Determine reasoning mode used
                mode_name = structured_v2_result.reasoning_mode.value
                
                # B. Check if answer is evidence-grounded
                if not structured_v2_result.evidence_grounded:
                    # If not grounded, reduce confidence significantly
                    answer_selection.confidence_score *= 0.6
                    answer_selection.reasoning_steps.append(ReasoningStep(
                        step_number=8,
                        description="Evidence Grounding Check",
                        reasoning=f"WARNING: Answer may not be fully grounded in evidence. Mode: {mode_name}",
                        evidence_used=["Structured reasoning verification"]
                    ))
                
                # C. Check for hallucination
                if self.hallucination_detector:
                    rationale_text = answer_selection.rationale if hasattr(answer_selection, 'rationale') else ""
                    hallucination_result = self.hallucination_detector.detect(
                        reasoning_text=rationale_text,
                        final_answer=answer_selection.selected_answer,
                        retrieved_contexts=retrieval_results,
                        options=list(options.values())
                    )
                    
                    if hallucination_result.is_hallucinated:
                        # Major hallucination detected - regenerate with grounded evidence
                        grounded_answer = self.hallucination_detector.generate_grounded_response(
                            question=question,
                            options=list(options.values()),
                            retrieved_contexts=retrieval_results
                        )
                        
                        if grounded_answer != "Cannot answer from context":
                            answer_selection.selected_answer = grounded_answer
                            answer_selection.confidence_score = min(0.5, answer_selection.confidence_score)
                        
                        answer_selection.reasoning_steps.append(ReasoningStep(
                            step_number=9,
                            description="Hallucination Detection & Correction",
                            reasoning=f"Hallucination detected. {hallucination_result.recommendation}. Corrected answer applied.",
                            evidence_used=hallucination_result.grounded_elements[:3]
                        ))
                
                # D. Safety verification
                if self.safety_verifier:
                    patient_category = structured_v2_result.step2_patient_category.value
                    safety_result = self.safety_verifier.verify(
                        answer=answer_selection.selected_answer,
                        question=question,
                        case_description=case_description,
                        options=list(options.values()),
                        retrieved_contexts=retrieval_results,
                        patient_category=patient_category
                    )
                    
                    if not safety_result.is_safe:
                        # Safety violation detected
                        if safety_result.safety_level.value == 'dangerous':
                            # Critical safety issue - mark with very low confidence
                            answer_selection.confidence_score = min(0.2, answer_selection.confidence_score)
                            
                        answer_selection.reasoning_steps.append(ReasoningStep(
                            step_number=10,
                            description="Safety Verification",
                            reasoning=f"SAFETY {safety_result.safety_level.value.upper()}: {'; '.join(safety_result.recommendations[:2])}",
                            evidence_used=safety_result.emergency_flags[:3] if safety_result.emergency_flags else []
                        ))
                    else:
                        # Safety verified - slight confidence boost
                        answer_selection.confidence_score = min(0.95, answer_selection.confidence_score * 1.05)
                
                # E. Confidence Calibration with Uncertainty Trigger
                if answer_selection.confidence_score < 0.65:
                    # Low confidence - consider if structured reasoning agrees
                    if structured_v2_result.step6_final_answer != answer_selection.selected_answer:
                        # Disagreement - further reduce confidence
                        answer_selection.confidence_score *= 0.8
                    else:
                        # Agreement - slight confidence boost
                        answer_selection.confidence_score = min(0.7, answer_selection.confidence_score * 1.1)
                
                reasoning_method_tag = f"Advanced-{mode_name}"
                
            except Exception as e:
                print(f"[WARN] Advanced reasoning (Day 8+) failed: {e}")
        
        # Step 4: Check correctness (if ground truth provided)
        is_correct = None
        if correct_answer:
            is_correct = answer_selection.selected_answer.upper() == correct_answer.upper()
        
        total_time = time.time() - pipeline_start
        
        # Build metadata
        pipeline_metadata = {
            'query_understanding': {
                'specialty': query_understanding.likely_specialty,
                'acuity': query_understanding.acuity_level,
                'query_type': query_understanding.query_type
            },
            'retrieval': {
                'num_results': len(retrieval_results),
                'retrieval_time_ms': retrieval_time * 1000,
                'stage1_k': self.retriever.stage1_k,
                'stage2_k': self.retriever.stage2_k,
                'stage3_k': self.retriever.stage3_k
            },
            'reasoning': {
                'reasoning_time_ms': reasoning_time * 1000,
                'num_reasoning_steps': len(answer_selection.reasoning_steps),
                'supporting_guidelines': answer_selection.supporting_guidelines,
                'method': reasoning_method_tag
            },
            'pipeline': {
                'total_time_ms': total_time * 1000
            }
        }
        
        return PipelineResult(
            question_id=question_id,
            question=question,
            case_description=case_description,
            options=options,
            selected_answer=answer_selection.selected_answer,
            correct_answer=correct_answer,
            is_correct=is_correct,
            confidence_score=answer_selection.confidence_score,
            retrieval_results=retrieval_results,
            reasoning=answer_selection,
            pipeline_metadata=pipeline_metadata
        )
    
    def process_batch(
        self,
        questions: List[Dict],
        show_progress: bool = True
    ) -> List[PipelineResult]:
        """
        Process a batch of questions.
        
        Args:
            questions: List of question dictionaries with keys:
                - question_id
                - case_description
                - question
                - options
                - correct_answer (optional)
            show_progress: Show progress bar
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(questions, desc="Processing questions")
        else:
            iterator = questions
        
        for q in iterator:
            result = self.answer_question(
                question_id=q.get('question_id', 'unknown'),
                case_description=q.get('case_description', ''),
                question=q.get('question', ''),
                options=q.get('options', {}),
                correct_answer=q.get('correct_answer')
            )
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'retriever': self.retriever.get_statistics(),
            'faiss_vectors': self.faiss_store.index.ntotal if self.faiss_store.index else 0,
            'bm25_documents': len(self.bm25_retriever.documents)
        }


def load_pipeline(
    index_dir: Optional[str] = None,
    retrieval_config: Optional[Dict] = None,
    config_path: Optional[str] = None
) -> MultiStageRAGPipeline:
    """
    Load and initialize the complete pipeline.
    
    Args:
        index_dir: Directory containing FAISS index (default: from config or data/indexes)
        retrieval_config: Configuration for multi-stage retriever (overrides config file)
        config_path: Path to YAML config file (default: config/pipeline_config.yaml)
        
    Returns:
        Initialized MultiStageRAGPipeline
    """
    # Load configuration
    try:
        from utils.config_loader import load_config
        config = load_config(config_path)
        paths_config = config.get_paths_config()
        
        # Use config values if not provided
        if index_dir is None:
            base_dir = Path(__file__).parent.parent.parent
            index_dir = str(base_dir / paths_config.get('index_dir', 'data/indexes'))
        
        # Get retrieval config from file if not provided
        if retrieval_config is None:
            retrieval_config = config.get_retrieval_config()
    except Exception as e:
        print(f"[WARN] Failed to load config: {e}, using defaults")
        base_dir = Path(__file__).parent.parent.parent
        if index_dir is None:
            index_dir = str(base_dir / "data" / "indexes")
        if retrieval_config is None:
            retrieval_config = {}
    
    # Initialize embedding model from config
    print("[INFO] Loading embedding model...")
    model_config = config.get_model_config()
    embedding_config = model_config.get('embedding', {})
    
    embedding_model = EmbeddingModel(
        model_name=embedding_config.get('model_name'),
        device=embedding_config.get('device', 'auto'),
        use_medical_model=embedding_config.get('use_medical_model', True),
        precision=embedding_config.get('precision', 'auto')
    )
    
    # Load FAISS index with GPU support
    print("[INFO] Loading FAISS index...")
    faiss_store = FAISSVectorStore(
        embedding_model,
        use_gpu=embedding_model.device == "cuda"
    )
    faiss_store.load_index(index_dir)
    
    # Build BM25 index
    print("[INFO] Building BM25 index...")
    bm25_retriever = BM25Retriever(faiss_store.documents)
    
    # Create pipeline
    print("[INFO] Initializing multi-stage RAG pipeline...")
    pipeline = MultiStageRAGPipeline(
        faiss_store=faiss_store,
        bm25_retriever=bm25_retriever,
        embedding_model=embedding_model,
        retrieval_config=retrieval_config
    )
    
    print("[OK] Pipeline ready!")
    return pipeline


def main():
    """Demo: Test complete pipeline."""
    print("="*70)
    print("MULTI-STAGE RAG PIPELINE DEMO")
    print("="*70)
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Test question
    test_question = {
        'question_id': 'TEST_001',
        'case_description': 'A 65-year-old male presents with chest pain and elevated troponin levels.',
        'question': 'What is the recommended first-line treatment?',
        'options': {
            'A': 'Aspirin 325mg',
            'B': 'Morphine',
            'C': 'Beta blocker',
            'D': 'ACE inhibitor'
        },
        'correct_answer': 'A'
    }
    
    print(f"\n[TEST] Processing test question...")
    result = pipeline.answer_question(**test_question)
    
    print(f"\n{'='*70}")
    print("PIPELINE RESULT")
    print(f"{'='*70}")
    print(f"\nQuestion: {result.question}")
    print(f"Selected Answer: {result.selected_answer}")
    print(f"Correct Answer: {result.correct_answer}")
    print(f"Is Correct: {result.is_correct}")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"\nRationale: {result.reasoning.rationale}")
    print(f"\nSupporting Guidelines: {', '.join(result.reasoning.supporting_guidelines)}")
    print(f"\nTotal Time: {result.pipeline_metadata['pipeline']['total_time_ms']:.1f}ms")
    print(f"  - Retrieval: {result.pipeline_metadata['retrieval']['retrieval_time_ms']:.1f}ms")
    print(f"  - Reasoning: {result.pipeline_metadata['reasoning']['reasoning_time_ms']:.1f}ms")


if __name__ == "__main__":
    main()


