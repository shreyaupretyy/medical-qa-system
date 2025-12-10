"""
Day 5+ System Improvements

This package contains improvements to address critical issues:
- Phase 1: Retrieval precision fixes
- Phase 2: Reasoning error reduction
- Phase 3: Specialty-specific improvements
- Day 7+ Critical Fixes:
  * Fix 1: Multi-Query Expansion (symptom, guideline, disease, keyword focused)
  * Fix 2: Symptom Keyword Injection with synonym dictionary
  * Fix 3: Guideline Prioritization Reranker
  * Fix 4: Forced Uncertainty Step
  * Fix 5: Minimal Context Re-evaluation
  * Fix 6: Stepwise Differential Diagnosis Template
- Day 8+ Advanced Fixes:
  * A: Structured 6-Step Medical Reasoning Template
  * B: Context-Grounded Reasoning with Evidence Linking
  * C: Safety Verification (contraindications, red flags)
  * D: Hallucination Detection & Suppression
  * E: Terminology Normalization
"""

try:
    from improvements.medical_query_enhancer import MedicalQueryEnhancer
    from improvements.structured_reasoner import StructuredMedicalReasoner
    from improvements.confidence_calibrator import ConfidenceCalibrator
    from improvements.specialty_adapter import SpecialtyAdapter
    # Day 7+ Critical Fixes
    from improvements.multi_query_expander import MultiQueryExpander
    from improvements.symptom_synonym_injector import SymptomSynonymInjector
    from improvements.guideline_reranker import GuidelineReranker
    from improvements.enhanced_reasoning import EnhancedMedicalReasoner
    # Day 7+ Step Fixes
    from improvements.clinical_feature_extractor import ClinicalFeatureExtractor
    from improvements.context_pruner import ContextPruner, QuestionKeywordExtractor
    from improvements.deterministic_reasoner import DeterministicReasoner
    # Day 8+ Advanced Fixes
    from improvements.structured_medical_reasoner_v2 import StructuredMedicalReasonerV2, ReasoningMode, PatientCategory
    from improvements.safety_verifier import MedicalSafetyVerifier, SafetyLevel
    from improvements.hallucination_detector import HallucinationDetector
    from improvements.terminology_normalizer import TerminologyNormalizer
    
    __all__ = [
        'MedicalQueryEnhancer',
        'StructuredMedicalReasoner',
        'ConfidenceCalibrator',
        'SpecialtyAdapter',
        # Day 7+ Critical Fixes
        'MultiQueryExpander',
        'SymptomSynonymInjector',
        'GuidelineReranker',
        'EnhancedMedicalReasoner',
        # Day 7+ Step Fixes
        'ClinicalFeatureExtractor',
        'ContextPruner',
        'QuestionKeywordExtractor',
        'DeterministicReasoner',
        # Day 8+ Advanced Fixes
        'StructuredMedicalReasonerV2',
        'ReasoningMode',
        'PatientCategory',
        'MedicalSafetyVerifier',
        'SafetyLevel',
        'HallucinationDetector',
        'TerminologyNormalizer',
    ]
except ImportError:
    # Handle relative imports
    from .medical_query_enhancer import MedicalQueryEnhancer
    from .structured_reasoner import StructuredMedicalReasoner
    from .confidence_calibrator import ConfidenceCalibrator
    from .specialty_adapter import SpecialtyAdapter
    # Day 7+ Critical Fixes
    from .multi_query_expander import MultiQueryExpander
    from .symptom_synonym_injector import SymptomSynonymInjector
    from .guideline_reranker import GuidelineReranker
    from .enhanced_reasoning import EnhancedMedicalReasoner
    # Day 7+ Step Fixes
    from .clinical_feature_extractor import ClinicalFeatureExtractor
    from .context_pruner import ContextPruner, QuestionKeywordExtractor
    from .deterministic_reasoner import DeterministicReasoner
    # Day 8+ Advanced Fixes
    from .structured_medical_reasoner_v2 import StructuredMedicalReasonerV2, ReasoningMode, PatientCategory
    from .safety_verifier import MedicalSafetyVerifier, SafetyLevel
    from .hallucination_detector import HallucinationDetector
    from .terminology_normalizer import TerminologyNormalizer
    
    __all__ = [
        'MedicalQueryEnhancer',
        'StructuredMedicalReasoner',
        'ConfidenceCalibrator',
        'SpecialtyAdapter',
        # Day 7+ Critical Fixes
        'MultiQueryExpander',
        'SymptomSynonymInjector',
        'GuidelineReranker',
        'EnhancedMedicalReasoner',
        # Day 7+ Step Fixes
        'ClinicalFeatureExtractor',
        'ContextPruner',
        'QuestionKeywordExtractor',
        'DeterministicReasoner',
        # Day 8+ Advanced Fixes
        'StructuredMedicalReasonerV2',
        'ReasoningMode',
        'PatientCategory',
        'MedicalSafetyVerifier',
        'SafetyLevel',
        'HallucinationDetector',
        'TerminologyNormalizer',
    ]

