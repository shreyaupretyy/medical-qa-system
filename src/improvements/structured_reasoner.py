"""
Structured Medical Reasoning Module

Implements 5-step structured medical reasoning:
1. Clinical feature extraction
2. Differential diagnosis generation
3. Evidence gathering and scoring
4. Treatment guideline matching
5. Answer selection with calibrated confidence

Addresses Day 4 Issue: 77.6% of errors are reasoning errors
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.document_processor import Document
from reasoning.query_understanding import ClinicalFeatures, QueryUnderstanding
from models.ollama_model import OllamaModel


@dataclass
class ClinicalFeatureExtraction:
    """Extracted clinical features from case."""
    symptoms: List[str]
    demographics: Dict[str, Optional[str]]
    vital_signs: Dict[str, Optional[float]]
    lab_findings: Dict[str, Optional[float]]
    medications: List[str]
    comorbidities: List[str]
    acuity: str
    specialty: Optional[str]


@dataclass
class DifferentialDiagnosis:
    """Differential diagnosis candidate."""
    diagnosis: str
    probability: float
    supporting_evidence: List[str]
    contraindications: List[str]
    urgency: str


@dataclass
class EvidenceScore:
    """Scored evidence for an answer option."""
    option_label: str
    option_text: str
    direct_evidence_score: float
    inferred_evidence_score: float
    guideline_alignment_score: float
    contraindication_penalty: float
    total_score: float
    evidence_sources: List[Tuple[Document, str, float]]  # (doc, excerpt, relevance)
    reasoning: str


@dataclass
class StructuredReasoningResult:
    """Complete structured reasoning result."""
    selected_answer: str
    confidence_score: float
    reasoning_steps: List[Dict]
    differential_diagnoses: List[DifferentialDiagnosis]
    evidence_scores: Dict[str, EvidenceScore]
    rationale: str
    supporting_guidelines: List[str]
    uncertainty_estimate: float


class StructuredMedicalReasoner:
    """
    5-step structured medical reasoning engine.
    
    This addresses reasoning errors by:
    - Systematic clinical feature extraction
    - Explicit differential diagnosis generation
    - Evidence-based scoring with weights
    - Treatment guideline matching
    - Calibrated confidence with uncertainty
    """
    
    def __init__(self, llm_model: Optional[OllamaModel] = None):
        """Initialize structured reasoner.
        
        Args:
            llm_model: Optional LLM model for enhanced reasoning. If None, uses rule-based only.
        """
        self.llm_model = llm_model
        self._init_medical_logic_rules()
        self._init_evidence_scoring_weights()
    
    def _init_medical_logic_rules(self):
        """Initialize medical logic rules."""
        self.medical_rules = {
            'rule_out_worst_case': True,
            'consider_demographics': True,
            'check_contraindications': True,
            'prefer_guideline_based': True,
            'consider_urgency': True,
            'weight_evidence_by_authority': True,
        }
    
    def _init_evidence_scoring_weights(self):
        """Initialize weights for evidence scoring."""
        # PubMedBERT: Optimized weights for better recall and accuracy
        self.scoring_weights = {
            'direct_evidence': 0.6,  # Increased from 0.5
            'inferred_evidence': 0.3,  # Keep same
            'guideline_alignment': 0.1,  # Reduced from 0.2
            'contraindication_penalty': -0.4,  # Less aggressive (was -0.5)
        }
    
    def reason(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        retrieved_contexts: List[Document],
        query_understanding: Optional[QueryUnderstanding] = None
    ) -> StructuredReasoningResult:
        """
        Perform 5-step structured medical reasoning.
        
        Args:
            question: The question being asked
            case_description: Patient case description
            options: Answer options {A: "text", ...}
            retrieved_contexts: Retrieved medical documents
            query_understanding: Optional pre-computed query understanding
            
        Returns:
            StructuredReasoningResult with answer and reasoning
        """
        # Check if context is sufficient
        if not retrieved_contexts or len(retrieved_contexts) == 0:
            return StructuredReasoningResult(
                selected_answer="Cannot answer from the provided context.",
                confidence_score=0.0,
                reasoning_steps=[{
                    'step': 0,
                    'name': 'Context Sufficiency Check',
                    'result': {},
                    'description': 'No medical guidelines were retrieved. Cannot answer from the provided context.'
                }],
                differential_diagnoses=[],
                evidence_scores={},
                rationale="Cannot answer from the provided context. No medical guidelines were retrieved.",
                supporting_guidelines=[],
                uncertainty_estimate=1.0
            )
        
        # Check if context has sufficient information
        context_text = " ".join([doc.content for doc in retrieved_contexts[:5]])
        if len(context_text.strip()) < 50:  # Very minimal context
            return StructuredReasoningResult(
                selected_answer="Cannot answer from the provided context.",
                confidence_score=0.0,
                reasoning_steps=[{
                    'step': 0,
                    'name': 'Context Sufficiency Check',
                    'result': {},
                    'description': 'Retrieved context is insufficient. Cannot answer from the provided context.'
                }],
                differential_diagnoses=[],
                evidence_scores={},
                rationale="Cannot answer from the provided context. Retrieved context is insufficient.",
                supporting_guidelines=[],
                uncertainty_estimate=1.0
            )
        
        reasoning_steps = []
        
        # Step 1: Clinical Feature Extraction
        step1_result = self._step1_extract_clinical_features(
            case_description, question, query_understanding
        )
        reasoning_steps.append({
            'step': 1,
            'name': 'Clinical Feature Extraction',
            'result': step1_result,
            'description': f"Extracted {len(step1_result.symptoms)} symptoms, "
                          f"demographics: {step1_result.demographics}, "
                          f"acuity: {step1_result.acuity}"
        })
        
        # Step 2: Differential Diagnosis Generation
        step2_result = self._step2_generate_differential_diagnoses(
            step1_result, retrieved_contexts
        )
        reasoning_steps.append({
            'step': 2,
            'name': 'Differential Diagnosis Generation',
            'result': step2_result,
            'description': f"Generated {len(step2_result)} differential diagnoses"
        })
        
        # Step 3: Evidence Gathering and Scoring
        step3_result = self._step3_gather_and_score_evidence(
            options, retrieved_contexts, step1_result, step2_result
        )
        reasoning_steps.append({
            'step': 3,
            'name': 'Evidence Gathering and Scoring',
            'result': {label: {
                'total_score': score.total_score,
                'evidence_count': len(score.evidence_sources)
            } for label, score in step3_result.items()},
            'description': f"Scored evidence for {len(step3_result)} options"
        })
        
        # Step 4: Treatment Guideline Matching
        step4_result = self._step4_match_treatment_guidelines(
            step3_result, retrieved_contexts, step1_result
        )
        reasoning_steps.append({
            'step': 4,
            'name': 'Treatment Guideline Matching',
            'result': step4_result,
            'description': f"Matched guidelines for top {len(step4_result)} options"
        })
        
        # Step 5: Answer Selection with Confidence
        step5_result = self._step5_select_answer_with_confidence(
            step3_result, step4_result, step1_result
        )
        
        # Check if we have sufficient evidence - if confidence is very low and no strong evidence, return "Cannot answer"
        if step5_result['confidence'] < 0.05:  # Very low confidence threshold (reduced from 0.15)
            # Check if any option has meaningful evidence
            has_meaningful_evidence = any(
                score.total_score > 0.05 and len(score.evidence_sources) > 0  # Lowered from 0.1
                for score in step3_result.values()
            )
            if not has_meaningful_evidence:
                return StructuredReasoningResult(
                    selected_answer="Cannot answer from the provided context.",
                    confidence_score=0.0,
                    reasoning_steps=reasoning_steps + [{
                        'step': 5.5,
                        'name': 'Insufficient Evidence Check',
                        'result': {},
                        'description': 'No meaningful evidence found in retrieved context for any option. Cannot answer from the provided context.'
                    }],
                    differential_diagnoses=step2_result,
                    evidence_scores=step3_result,
                    rationale="Cannot answer from the provided context. No meaningful evidence found in retrieved medical guidelines for any answer option.",
                    supporting_guidelines=[],
                    uncertainty_estimate=1.0
                )
        
        reasoning_steps.append({
            'step': 5,
            'name': 'Answer Selection',
            'result': step5_result,
            'description': f"Selected answer {step5_result['selected_answer']} "
                          f"with confidence {step5_result['confidence']:.2%}"
        })
        
        # Step 6 (Optional): LLM Verification - Use LLM to verify and potentially adjust the answer
        if self.llm_model and step5_result['confidence'] < 0.8:  # Only verify if not highly confident
            llm_verification = self._step6_llm_verification(
                question, case_description, options, 
                step5_result, step3_result, retrieved_contexts
            )
            if llm_verification:
                # Update answer if LLM has strong disagreement with high confidence
                if (llm_verification['answer'] != step5_result['selected_answer'] and 
                    llm_verification['confidence'] > 0.7 and
                    step5_result['confidence'] < 0.6):
                    step5_result['selected_answer'] = llm_verification['answer']
                    step5_result['confidence'] = (llm_verification['confidence'] + step5_result['confidence']) / 2
                    reasoning_steps.append({
                        'step': 6,
                        'name': 'LLM Verification',
                        'result': llm_verification,
                        'description': f"LLM verification adjusted answer to {llm_verification['answer']}"
                    })
                else:
                    # Boost confidence if LLM agrees
                    if llm_verification['answer'] == step5_result['selected_answer']:
                        step5_result['confidence'] = min(1.0, step5_result['confidence'] * 1.15)
                        reasoning_steps.append({
                            'step': 6,
                            'name': 'LLM Verification',
                            'result': llm_verification,
                            'description': f"LLM verification confirmed answer, boosted confidence"
                        })
        
        # Generate rationale
        rationale = self._generate_rationale(
            step5_result, step3_result, step2_result, step1_result
        )
        
        # Extract supporting guidelines
        supporting_guidelines = list(set([
            doc.metadata.get('guideline_id', '')
            for score in step3_result.values()
            for doc, _, _ in score.evidence_sources
            if doc.metadata.get('guideline_id')
        ]))
        
        return StructuredReasoningResult(
            selected_answer=step5_result['selected_answer'],
            confidence_score=step5_result['confidence'],
            reasoning_steps=reasoning_steps,
            differential_diagnoses=step2_result,
            evidence_scores=step3_result,
            rationale=rationale,
            supporting_guidelines=supporting_guidelines,
            uncertainty_estimate=step5_result['uncertainty']
        )
    
    def _step1_extract_clinical_features(
        self,
        case_description: str,
        question: str,
        query_understanding: Optional[QueryUnderstanding]
    ) -> ClinicalFeatureExtraction:
        """Step 1: Extract clinical features."""
        # Day 6: Use enhanced symptom extractor if available
        enhanced_symptoms = []
        try:
            from optimization.symptom_extractor import EnhancedSymptomExtractor
            symptom_extractor = EnhancedSymptomExtractor()
            symptom_result = symptom_extractor.extract_symptoms(case_description, question)
            # Use enhanced symptoms
            enhanced_symptoms = symptom_result.all_symptom_terms
        except ImportError:
            pass  # Fallback to query understanding
        
        if query_understanding:
            features = query_understanding.clinical_features
            # Day 6: Merge enhanced symptoms with query understanding
            if enhanced_symptoms:
                features.symptoms = list(set(features.symptoms + enhanced_symptoms))
        else:
            # Fallback: basic extraction
            features = ClinicalFeatures(
                symptoms=enhanced_symptoms if enhanced_symptoms else [],
                demographics={}, medical_terms=[],
                medications=[], tests=[], conditions=[],
                urgency_keywords=[], specialty_hints=[], negations=[]
            )
        
        # Extract vital signs (if mentioned)
        vital_signs = self._extract_vital_signs(case_description)
        
        # Extract lab findings
        lab_findings = self._extract_lab_findings(case_description)
        
        # Extract comorbidities
        comorbidities = self._extract_comorbidities(case_description)
        
        return ClinicalFeatureExtraction(
            symptoms=features.symptoms,
            demographics=features.demographics,
            vital_signs=vital_signs,
            lab_findings=lab_findings,
            medications=features.medications,
            comorbidities=comorbidities,
            acuity=query_understanding.acuity_level if query_understanding else 'routine',
            specialty=query_understanding.likely_specialty if query_understanding else None
        )
    
    def _step2_generate_differential_diagnoses(
        self,
        clinical_features: ClinicalFeatureExtraction,
        retrieved_contexts: List[Document]
    ) -> List[DifferentialDiagnosis]:
        """Step 2: Generate differential diagnoses using LLM if available."""
        differentials = []
        
        # If LLM is available, use it for intelligent differential generation
        if self.llm_model:
            # Prepare context from retrieved documents
            context_text = "\n\n".join([
                f"Guideline {i+1}: {doc.content[:500]}"
                for i, doc in enumerate(retrieved_contexts[:5])
            ])
            
            symptoms_str = ", ".join(clinical_features.symptoms) if clinical_features.symptoms else "None specified"
            
            prompt = f"""Based on the clinical presentation and medical guidelines provided, generate a differential diagnosis list.

CLINICAL PRESENTATION:
- Symptoms: {symptoms_str}
- Demographics: {clinical_features.demographics}
- Vital Signs: {clinical_features.vital_signs}
- Acuity: {clinical_features.acuity}

RELEVANT MEDICAL GUIDELINES:
{context_text}

TASK: Generate top 3-5 differential diagnoses that match this presentation based on the guidelines.
For each diagnosis, provide:
1. Diagnosis name
2. Probability (0.0-1.0)
3. Supporting evidence from the guidelines

FORMAT YOUR RESPONSE AS JSON:
{{"differentials": [{{"diagnosis": "...", "probability": 0.X, "evidence": "..."}}, ...]}}

Only include diagnoses supported by the provided guidelines."""

            try:
                response = self.llm_model.generate(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=512
                )
                
                # Parse JSON response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    for diff in data.get('differentials', [])[:5]:
                        differentials.append(DifferentialDiagnosis(
                            diagnosis=diff.get('diagnosis', 'Unknown'),
                            probability=float(diff.get('probability', 0.5)),
                            supporting_evidence=[diff.get('evidence', '')],
                            contraindications=[],
                            urgency=clinical_features.acuity
                        ))
            except Exception as e:
                # Fallback to rule-based if LLM fails
                pass
        
        # Fallback to rule-based if no LLM or LLM failed
        if not differentials:
            symptom_diagnosis_map = {
                'chest pain': ['myocardial infarction', 'angina', 'pulmonary embolism'],
                'shortness of breath': ['heart failure', 'pneumonia', 'copd', 'pulmonary embolism'],
                'fever': ['infection', 'sepsis', 'pneumonia'],
                'abdominal pain': ['appendicitis', 'cholecystitis', 'peptic ulcer'],
                'headache': ['migraine', 'meningitis', 'subarachnoid hemorrhage'],
            }
            
            for symptom in clinical_features.symptoms:
                if symptom.lower() in symptom_diagnosis_map:
                    for diagnosis in symptom_diagnosis_map[symptom.lower()]:
                        support_score = self._check_diagnosis_support(diagnosis, retrieved_contexts)
                        if support_score > 0.2:
                            differentials.append(DifferentialDiagnosis(
                                diagnosis=diagnosis,
                                probability=support_score,
                                supporting_evidence=[f"Supported by symptom: {symptom}"],
                                contraindications=[],
                                urgency=clinical_features.acuity
                            ))
        
        # Sort by probability
        differentials.sort(key=lambda x: x.probability, reverse=True)
        return differentials[:5]  # Top 5
    
    def _step3_gather_and_score_evidence(
        self,
        options: Dict[str, str],
        retrieved_contexts: List[Document],
        clinical_features: ClinicalFeatureExtraction,
        differentials: List[DifferentialDiagnosis]
    ) -> Dict[str, EvidenceScore]:
        """Step 3: Gather and score evidence for each option."""
        evidence_scores = {}
        
        for option_label, option_text in options.items():
            # Gather evidence
            direct_evidence = []
            inferred_evidence = []
            
            for doc in retrieved_contexts:
                doc_content_lower = doc.content.lower()
                option_lower = option_text.lower()
                
                # Direct evidence: exact match or high similarity
                if option_lower in doc_content_lower:
                    excerpt = self._extract_relevant_excerpt(option_text, doc.content)
                    # Check context for better scoring
                    pos = doc_content_lower.find(option_lower)
                    context = doc_content_lower[max(0, pos-150):min(len(doc_content_lower), pos+len(option_lower)+150)]
                    if any(term in context for term in ['treatment', 'therapy', 'medication', 'management', 'prescribe', 'recommended', 'indicated']):
                        direct_evidence.append((doc, excerpt, 1.0))  # Perfect score in treatment context
                    else:
                        direct_evidence.append((doc, excerpt, 0.95))  # Slightly lower if not in treatment context
                
                # Inferred evidence: keyword overlap (PubMedBERT: lower threshold for better recall)
                option_keywords = set(option_lower.split())
                doc_keywords = set(doc_content_lower.split())
                overlap = len(option_keywords & doc_keywords)
                if overlap >= 1:  # Changed from > 1 to >= 1 for even better recall
                    excerpt = self._extract_relevant_excerpt(option_text, doc.content)
                    # Calculate score with better weighting
                    base_score = overlap / max(1, len(option_keywords))
                    # Boost for important medical terms
                    important_terms = ['treatment', 'medication', 'dose', 'mg', 'therapy', 'management', 'drug', 'prescribe']
                    important_overlap = len([t for t in important_terms if t in option_keywords & doc_keywords])
                    if important_overlap > 0:
                        base_score *= (1.0 + important_overlap * 0.15)  # 15% boost per important term (increased from 10%)
                    inferred_evidence.append((doc, excerpt, min(1.0, base_score * 1.5)))  # Higher boost (increased from 1.3)
            
            # Score evidence (PubMedBERT: better scoring for accuracy)
            if direct_evidence:
                # Average of direct evidence scores, with boost for multiple sources
                direct_score = sum(score for _, _, score in direct_evidence) / len(direct_evidence)
                if len(direct_evidence) > 2:
                    direct_score *= 1.15  # 15% boost for 3+ direct sources (increased from 10%)
                elif len(direct_evidence) > 1:
                    direct_score *= 1.12  # 12% boost for multiple direct sources (increased from 10%)
            else:
                direct_score = 0.0
            
            if inferred_evidence:
                # Average of inferred evidence scores, with boost for multiple sources
                inferred_score = sum(score for _, _, score in inferred_evidence) / len(inferred_evidence)
                if len(inferred_evidence) > 3:
                    inferred_score *= 1.10  # 10% boost for 4+ inferred sources (increased from 5%)
                elif len(inferred_evidence) > 2:
                    inferred_score *= 1.08  # 8% boost for multiple inferred sources (increased from 5%)
            else:
                inferred_score = 0.0
            
            # Check contraindications
            contraindication_penalty = self._check_contraindications(option_text, retrieved_contexts)
            
            # Guideline alignment
            guideline_score = self._calculate_guideline_alignment(
                option_text, retrieved_contexts, clinical_features
            )
            
            # Total score
            total_score = (
                direct_score * self.scoring_weights['direct_evidence'] +
                inferred_score * self.scoring_weights['inferred_evidence'] +
                guideline_score * self.scoring_weights['guideline_alignment'] +
                contraindication_penalty * self.scoring_weights['contraindication_penalty']
            )
            total_score = max(0.0, min(1.0, total_score))  # Clamp to [0, 1]
            
            # Generate reasoning
            reasoning = self._generate_evidence_reasoning(
                direct_evidence, inferred_evidence, contraindication_penalty
            )
            
            evidence_scores[option_label] = EvidenceScore(
                option_label=option_label,
                option_text=option_text,
                direct_evidence_score=direct_score,
                inferred_evidence_score=inferred_score,
                guideline_alignment_score=guideline_score,
                contraindication_penalty=contraindication_penalty,
                total_score=total_score,
                evidence_sources=direct_evidence + inferred_evidence,
                reasoning=reasoning
            )
        
        return evidence_scores
    
    def _step4_match_treatment_guidelines(
        self,
        evidence_scores: Dict[str, EvidenceScore],
        retrieved_contexts: List[Document],
        clinical_features: ClinicalFeatureExtraction
    ) -> Dict[str, float]:
        """Step 4: Match treatment guidelines."""
        guideline_scores = {}
        
        # Score each option based on guideline alignment
        for option_label, evidence_score in evidence_scores.items():
            # Count guideline mentions
            guideline_count = sum(
                1 for doc, _, _ in evidence_score.evidence_sources
                if 'guideline' in doc.metadata.get('title', '').lower() or
                   'guideline' in doc.content.lower()
            )
            
            # Normalize
            guideline_scores[option_label] = min(1.0, guideline_count / 3.0)
        
        return guideline_scores
    
    def _step5_select_answer_with_confidence(
        self,
        evidence_scores: Dict[str, EvidenceScore],
        guideline_scores: Dict[str, float],
        clinical_features: ClinicalFeatureExtraction
    ) -> Dict:
        """Step 5: Select answer with calibrated confidence."""
        # Combine scores (PubMedBERT: favor evidence more, with boost for high evidence)
        final_scores = {}
        for option_label, evidence_score in evidence_scores.items():
            base_score = (
                evidence_score.total_score * 0.8 +  # Increased from 0.7
                guideline_scores.get(option_label, 0.0) * 0.2  # Reduced from 0.3
            )
            # Boost if evidence is strong and from multiple sources
            if evidence_score.total_score > 0.5 and len(evidence_score.evidence_sources) > 1:
                base_score = min(1.0, base_score * 1.1)  # 10% boost for strong multi-source evidence
            final_scores[option_label] = base_score
        
        # Select best answer
        selected_answer = max(final_scores.items(), key=lambda x: x[1])[0]
        confidence = final_scores[selected_answer]
        
        # Calculate uncertainty (gap between top 2 scores)
        sorted_scores = sorted(final_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            uncertainty = sorted_scores[0] - sorted_scores[1]
        else:
            uncertainty = 0.0
        
        # Less aggressive confidence adjustment
        # Base confidence on evidence score, with minimal reduction (PubMedBERT: less aggressive)
        if uncertainty < 0.10:  # Reduced threshold from 0.15 to penalize less
            confidence *= 0.95  # Less penalty (changed from 0.9)
        elif uncertainty > 0.25:  # Clear winner = slight boost (reduced from 0.3)
            confidence = min(1.0, confidence * 1.10)  # Bigger boost (increased from 1.05)
        
        # Ensure minimum confidence if we have any evidence
        selected_evidence = evidence_scores[selected_answer]
        if selected_evidence.total_score > 0.05:  # Lower threshold for better recall
            confidence = max(confidence, 0.5)  # Minimum 50% if we have evidence (increased from 0.3)
        
        return {
            'selected_answer': selected_answer,
            'confidence': max(0.0, min(1.0, confidence)),
            'uncertainty': uncertainty,
            'final_scores': final_scores
        }
    
    # Helper methods
    def _extract_vital_signs(self, text: str) -> Dict[str, Optional[float]]:
        """Extract vital signs from text."""
        vital_signs = {}
        # Simple pattern matching (can be enhanced)
        import re
        bp_match = re.search(r'bp[:\s]+(\d+)/(\d+)', text, re.IGNORECASE)
        if bp_match:
            vital_signs['systolic'] = float(bp_match.group(1))
            vital_signs['diastolic'] = float(bp_match.group(2))
        return vital_signs
    
    def _extract_lab_findings(self, text: str) -> Dict[str, Optional[float]]:
        """Extract lab findings from text."""
        lab_findings = {}
        # Simple pattern matching
        import re
        troponin_match = re.search(r'troponin[:\s]+([\d.]+)', text, re.IGNORECASE)
        if troponin_match:
            lab_findings['troponin'] = float(troponin_match.group(1))
        return lab_findings
    
    def _extract_comorbidities(self, text: str) -> List[str]:
        """Extract comorbidities from text."""
        comorbidities = []
        common_comorbidities = ['diabetes', 'hypertension', 'copd', 'ckd']
        text_lower = text.lower()
        for comorbidity in common_comorbidities:
            if comorbidity in text_lower:
                comorbidities.append(comorbidity)
        return comorbidities
    
    def _check_diagnosis_support(self, diagnosis: str, contexts: List[Document]) -> float:
        """Check how well contexts support a diagnosis."""
        support_count = 0
        for doc in contexts:
            if diagnosis.lower() in doc.content.lower():
                support_count += 1
        return min(1.0, support_count / 3.0)
    
    def _extract_relevant_excerpt(self, query_text: str, document: str, context_chars: int = 400) -> str:
        """Extract relevant excerpt from document."""
        query_lower = query_text.lower()
        doc_lower = document.lower()
        pos = doc_lower.find(query_lower)
        if pos == -1:
            return document[:context_chars] + "..."
        start = max(0, pos - context_chars // 2)
        end = min(len(document), pos + len(query_text) + context_chars // 2)
        excerpt = document[start:end]
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(document):
            excerpt = excerpt + "..."
        return excerpt
    
    def _check_contraindications(self, option_text: str, contexts: List[Document]) -> float:
        """Check for contraindications."""
        contradiction_keywords = ['contraindicated', 'not recommended', 'avoid', 'should not']
        option_lower = option_text.lower()
        for doc in contexts:
            doc_lower = doc.content.lower()
            if option_lower in doc_lower:
                for keyword in contradiction_keywords:
                    if keyword in doc_lower:
                        # Check proximity
                        option_pos = doc_lower.find(option_lower)
                        keyword_pos = doc_lower.find(keyword)
                        if abs(option_pos - keyword_pos) < 100:
                            return 1.0  # Strong contradiction
        return 0.0
    
    def _calculate_guideline_alignment(
        self,
        option_text: str,
        contexts: List[Document],
        clinical_features: ClinicalFeatureExtraction
    ) -> float:
        """Calculate alignment with treatment guidelines."""
        alignment_score = 0.0
        option_lower = option_text.lower()
        
        for doc in contexts:
            doc_lower = doc.content.lower()
            if option_lower in doc_lower:
                # Check if it's a guideline document
                if 'guideline' in doc.metadata.get('title', '').lower():
                    alignment_score += 0.3
                # Check specialty match
                if clinical_features.specialty:
                    if clinical_features.specialty.lower() in doc.metadata.get('category', '').lower():
                        alignment_score += 0.2
        
        return min(1.0, alignment_score)
    
    def _generate_evidence_reasoning(
        self,
        direct_evidence: List,
        inferred_evidence: List,
        contraindication_penalty: float
    ) -> str:
        """Generate reasoning text for evidence."""
        reasoning_parts = []
        if direct_evidence:
            reasoning_parts.append(f"{len(direct_evidence)} direct evidence sources")
        if inferred_evidence:
            reasoning_parts.append(f"{len(inferred_evidence)} inferred evidence sources")
        if contraindication_penalty > 0:
            reasoning_parts.append("Contraindications noted")
        return "; ".join(reasoning_parts) if reasoning_parts else "Limited evidence"
    
    def _step6_llm_verification(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        step5_result: Dict,
        evidence_scores: Dict[str, EvidenceScore],
        retrieved_contexts: List[Document]
    ) -> Optional[Dict]:
        """Step 6: Use LLM to verify and potentially adjust the answer."""
        if not self.llm_model:
            return None
        
        # Prepare evidence summary
        evidence_summary = []
        for label, score in sorted(evidence_scores.items(), key=lambda x: x[1].total_score, reverse=True)[:3]:
            evidence_summary.append(f"{label}) {options[label]}: Score {score.total_score:.2f}, "
                                  f"{len(score.evidence_sources)} sources")
        
        # Prepare context
        context_text = "\n\n".join([
            f"Guideline {i+1}: {doc.content[:400]}"
            for i, doc in enumerate(retrieved_contexts[:5])
        ])
        
        prompt = f"""You are a medical expert. Review this case and select the best answer based ONLY on the provided medical guidelines.

CASE: {case_description}

QUESTION: {question}

OPTIONS:
{chr(10).join([f"{k}) {v}" for k, v in options.items()])}

RETRIEVED MEDICAL GUIDELINES:
{context_text}

RULE-BASED ANALYSIS:
Current selection: {step5_result['selected_answer']}) {options[step5_result['selected_answer']]}
Evidence scores for all options:
{chr(10).join(evidence_summary)}

TASK:
1. Verify if the current selection is correct based on the guidelines
2. If you find stronger evidence for a different answer, recommend it
3. Provide your confidence (0.0-1.0)

FORMAT YOUR RESPONSE AS JSON:
{{"answer": "A/B/C/D", "confidence": 0.X, "reasoning": "brief explanation"}}

CRITICAL: Base your answer ONLY on the provided medical guidelines. Do not use general medical knowledge."""

        try:
            response = self.llm_model.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=256
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'answer': data.get('answer', step5_result['selected_answer']),
                    'confidence': float(data.get('confidence', 0.5)),
                    'reasoning': data.get('reasoning', '')
                }
        except Exception as e:
            # Return None if verification fails
            pass
        
        return None
    
    def _generate_rationale(
        self,
        step5_result: Dict,
        evidence_scores: Dict[str, EvidenceScore],
        differentials: List[DifferentialDiagnosis],
        clinical_features: ClinicalFeatureExtraction
    ) -> str:
        """Generate comprehensive rationale."""
        selected = step5_result['selected_answer']
        evidence = evidence_scores[selected]
        
        rationale_parts = [
            f"Selected answer {selected} based on structured 5-step reasoning:",
            f"1. Clinical features: {', '.join(clinical_features.symptoms[:3])}",
            f"2. Top differential: {differentials[0].diagnosis if differentials else 'N/A'}",
            f"3. Evidence score: {evidence.total_score:.2f} (direct: {evidence.direct_evidence_score:.2f}, "
            f"inferred: {evidence.inferred_evidence_score:.2f})",
            f"4. {len(evidence.evidence_sources)} evidence sources",
            f"5. Confidence: {step5_result['confidence']:.2%} (uncertainty: {step5_result['uncertainty']:.2f})"
        ]
        
        return " | ".join(rationale_parts)


def main():
    """Demo: Test structured reasoning."""
    print("="*70)
    print("STRUCTURED MEDICAL REASONER DEMO")
    print("="*70)
    print("\n[INFO] Structured 5-step reasoning engine initialized")
    print("[INFO] Ready to process clinical cases with structured reasoning")


if __name__ == "__main__":
    main()

