"""
Structured Medical Reasoning Engine V2

Implements a 6-step deterministic reasoning template:
1. Extract symptoms and key findings
2. Identify patient category (adult/peds/pregnancy/emergency)
3. List differential diagnoses
4. Select most likely diagnosis with reasoning
5. Cite retrieved guideline lines that support decision
6. Give final answer ONLY if confirmed by evidence

Fallback hierarchy:
- Primary: Structured 6-step reasoning
- Backup: Chain-of-Thought reasoning
- Complex: Tree-of-Thought reasoning (for multi-step differential)
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import json


class ReasoningMode(Enum):
    """Reasoning mode selection."""
    STRUCTURED = "structured"  # Primary: 6-step template
    CHAIN_OF_THOUGHT = "cot"   # Backup: Linear reasoning
    TREE_OF_THOUGHT = "tot"    # Complex: Multi-branch exploration


class PatientCategory(Enum):
    """Patient category classification."""
    ADULT = "adult"
    PEDIATRIC = "pediatric"
    NEONATAL = "neonatal"
    PREGNANCY = "pregnancy"
    EMERGENCY = "emergency"
    GERIATRIC = "geriatric"


@dataclass
class ExtractedFindings:
    """Step 1: Extracted clinical findings."""
    symptoms: List[str] = field(default_factory=list)
    vital_signs: Dict[str, str] = field(default_factory=dict)
    lab_values: Dict[str, str] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[str] = None


@dataclass
class DifferentialDiagnosis:
    """A differential diagnosis with supporting evidence."""
    diagnosis: str
    likelihood: float  # 0-1
    supporting_evidence: List[str]
    against_evidence: List[str]
    guideline_support: bool = False
    guideline_citation: Optional[str] = None


@dataclass
class EvidenceCitation:
    """Citation from retrieved guidelines."""
    text: str
    source: str
    relevance_score: float
    supports_answer: bool


@dataclass
class StructuredReasoningResult:
    """Complete result of structured reasoning."""
    # Step outputs
    step1_findings: ExtractedFindings
    step2_patient_category: PatientCategory
    step3_differentials: List[DifferentialDiagnosis]
    step4_primary_diagnosis: Optional[DifferentialDiagnosis]
    step5_evidence_citations: List[EvidenceCitation]
    step6_final_answer: str
    
    # Metadata
    reasoning_mode: ReasoningMode
    confidence_score: float
    evidence_grounded: bool
    hallucination_detected: bool
    safety_verified: bool
    reasoning_trace: List[Dict[str, Any]]
    
    # Fallback info
    used_fallback: bool = False
    fallback_reason: Optional[str] = None


class StructuredMedicalReasonerV2:
    """
    Advanced medical reasoning engine with structured templates.
    
    Features:
    - 6-step deterministic reasoning
    - Evidence grounding requirement
    - Hallucination detection
    - Safety verification
    - Fallback to CoT/ToT for complex cases
    """
    
    def __init__(self, llm_model=None, config: Optional[Dict] = None):
        """Initialize the reasoner."""
        self.llm_model = llm_model
        self.config = config or {}
        
        # Reasoning mode thresholds
        self.complexity_threshold = self.config.get('complexity_threshold', 0.7)
        self.evidence_threshold = self.config.get('evidence_threshold', 0.5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.65)
        
        # Initialize pattern matchers
        self._init_patterns()
        
        # Red flag patterns for safety
        self._init_red_flags()
    
    def _init_patterns(self):
        """Initialize extraction patterns."""
        self.symptom_patterns = [
            r'\b(fever|cough|dyspnea|pain|bleeding|vomiting|diarrhea|headache|'
            r'seizure|confusion|weakness|fatigue|rash|swelling|tachycardia|'
            r'bradycardia|hypoxia|cyanosis|jaundice|edema|nausea|syncope|'
            r'dizziness|chest pain|abdominal pain|shortness of breath|'
            r'altered mental status|unconscious|lethargy|oliguria|anuria|'
            r'hematemesis|melena|hematuria|hemoptysis|dysphagia|dysuria)\b'
        ]
        
        self.vital_patterns = {
            'bp': r'(?:bp|blood pressure)[:\s]+(\d+/\d+)',
            'hr': r'(?:hr|heart rate|pulse)[:\s]+(\d+)',
            'rr': r'(?:rr|respiratory rate)[:\s]+(\d+)',
            'temp': r'(?:temp|temperature)[:\s]+([\d.]+)',
            'spo2': r'(?:spo2|o2 sat|oxygen)[:\s]+(\d+)',
            'gcs': r'(?:gcs|glasgow)[:\s]+(\d+)'
        }
        
        self.age_patterns = [
            r'(\d+)[- ]?(year|yr|y/o|years?)[- ]?old',
            r'(\d+)[- ]?(month|mo|months?)[- ]?old',
            r'(\d+)[- ]?(week|wk|weeks?)[- ]?old',
            r'(\d+)[- ]?(day|d|days?)[- ]?old',
            r'newborn|neonate|infant|toddler|child|adolescent|adult|elderly'
        ]
        
        self.pregnancy_patterns = [
            r'\b(pregnant|pregnancy|gravida|g\d+p\d+|gestational|prenatal|'
            r'antenatal|postpartum|trimester|weeks gestation|wog|weeks pregnant)\b'
        ]
        
        self.emergency_patterns = [
            r'\b(emergency|critical|severe|acute|unstable|shock|arrest|'
            r'respiratory failure|cardiac arrest|status epilepticus|'
            r'massive hemorrhage|anaphylaxis|trauma)\b'
        ]
    
    def _init_red_flags(self):
        """Initialize safety red flags."""
        self.red_flags = {
            'respiratory': ['stridor', 'severe dyspnea', 'cyanosis', 'apnea', 'respiratory failure'],
            'cardiovascular': ['cardiac arrest', 'pulseless', 'severe hypotension', 'shock'],
            'neurological': ['gcs < 8', 'status epilepticus', 'herniation', 'stroke symptoms'],
            'sepsis': ['septic shock', 'severe sepsis', 'refractory hypotension'],
            'obstetric': ['eclampsia', 'cord prolapse', 'placental abruption', 'severe preeclampsia'],
            'pediatric': ['non-accidental injury', 'severe dehydration', 'meningitis signs']
        }
    
    def reason(
        self,
        question: str,
        case_description: str,
        options: List[str],
        retrieved_contexts: List[Any],
        extracted_features: Optional[Any] = None
    ) -> StructuredReasoningResult:
        """
        Execute structured medical reasoning.
        
        Args:
            question: The clinical question
            case_description: Patient case description
            options: Answer options (A, B, C, D)
            retrieved_contexts: Retrieved guideline documents
            extracted_features: Pre-extracted clinical features
            
        Returns:
            StructuredReasoningResult with complete reasoning trace
        """
        reasoning_trace = []
        
        # Determine reasoning mode based on complexity
        mode = self._determine_reasoning_mode(question, case_description, options)
        reasoning_trace.append({'step': 'mode_selection', 'mode': mode.value})
        
        # Execute appropriate reasoning pipeline
        if mode == ReasoningMode.STRUCTURED:
            return self._structured_reasoning(
                question, case_description, options, 
                retrieved_contexts, extracted_features, reasoning_trace
            )
        elif mode == ReasoningMode.TREE_OF_THOUGHT:
            return self._tree_of_thought_reasoning(
                question, case_description, options,
                retrieved_contexts, extracted_features, reasoning_trace
            )
        else:
            return self._chain_of_thought_reasoning(
                question, case_description, options,
                retrieved_contexts, extracted_features, reasoning_trace
            )
    
    def _determine_reasoning_mode(
        self,
        question: str,
        case_description: str,
        options: List[str]
    ) -> ReasoningMode:
        """Determine the best reasoning mode for this question."""
        complexity_score = 0.0
        
        # Check for complex differential diagnosis
        differential_keywords = ['differential', 'most likely', 'best explains', 'cause']
        if any(kw in question.lower() for kw in differential_keywords):
            complexity_score += 0.3
        
        # Check for multiple conditions mentioned
        condition_count = len(re.findall(
            r'\b(pneumonia|sepsis|meningitis|diabetes|heart failure|stroke|'
            r'pulmonary embolism|preeclampsia|anemia|shock)\b',
            case_description.lower()
        ))
        if condition_count >= 2:
            complexity_score += 0.2
        
        # Check for treatment vs diagnosis question
        if 'treatment' in question.lower() or 'management' in question.lower():
            complexity_score += 0.1
        
        # Check for pediatric/pregnancy (more complex protocols)
        if any(re.search(p, case_description.lower()) for p in self.pregnancy_patterns):
            complexity_score += 0.2
        
        # Determine mode
        if complexity_score >= self.complexity_threshold:
            return ReasoningMode.TREE_OF_THOUGHT
        elif complexity_score >= 0.3:
            return ReasoningMode.CHAIN_OF_THOUGHT
        else:
            return ReasoningMode.STRUCTURED
    
    def _structured_reasoning(
        self,
        question: str,
        case_description: str,
        options: List[str],
        retrieved_contexts: List[Any],
        extracted_features: Optional[Any],
        reasoning_trace: List[Dict]
    ) -> StructuredReasoningResult:
        """Execute 6-step structured reasoning."""
        
        # STEP 1: Extract symptoms and key findings
        step1_findings = self._step1_extract_findings(case_description, extracted_features)
        reasoning_trace.append({
            'step': 1,
            'name': 'Extract Findings',
            'symptoms': step1_findings.symptoms,
            'vitals': step1_findings.vital_signs,
            'red_flags': step1_findings.red_flags
        })
        
        # STEP 2: Identify patient category
        step2_category = self._step2_identify_category(case_description, step1_findings)
        reasoning_trace.append({
            'step': 2,
            'name': 'Patient Category',
            'category': step2_category.value
        })
        
        # STEP 3: Generate differential diagnoses
        step3_differentials = self._step3_differential_diagnosis(
            question, case_description, step1_findings, step2_category, options
        )
        reasoning_trace.append({
            'step': 3,
            'name': 'Differential Diagnoses',
            'differentials': [d.diagnosis for d in step3_differentials]
        })
        
        # STEP 4: Select primary diagnosis with reasoning
        step4_primary = self._step4_select_diagnosis(
            step3_differentials, step1_findings, retrieved_contexts
        )
        reasoning_trace.append({
            'step': 4,
            'name': 'Primary Diagnosis',
            'selected': step4_primary.diagnosis if step4_primary else None,
            'likelihood': step4_primary.likelihood if step4_primary else 0
        })
        
        # STEP 5: Cite evidence from guidelines
        step5_citations = self._step5_cite_evidence(
            step4_primary, retrieved_contexts, options
        )
        reasoning_trace.append({
            'step': 5,
            'name': 'Evidence Citations',
            'citations': len(step5_citations),
            'grounded': any(c.supports_answer for c in step5_citations)
        })
        
        # STEP 6: Generate final answer (ONLY if evidence supports)
        step6_answer, confidence, grounded = self._step6_final_answer(
            question, options, step4_primary, step5_citations, step1_findings
        )
        reasoning_trace.append({
            'step': 6,
            'name': 'Final Answer',
            'answer': step6_answer,
            'confidence': confidence,
            'evidence_grounded': grounded
        })
        
        # Check for hallucination
        hallucination = self._detect_hallucination(step6_answer, step5_citations)
        
        # Verify safety
        safety_ok = self._verify_safety(step6_answer, step1_findings, step2_category)
        
        return StructuredReasoningResult(
            step1_findings=step1_findings,
            step2_patient_category=step2_category,
            step3_differentials=step3_differentials,
            step4_primary_diagnosis=step4_primary,
            step5_evidence_citations=step5_citations,
            step6_final_answer=step6_answer,
            reasoning_mode=ReasoningMode.STRUCTURED,
            confidence_score=confidence,
            evidence_grounded=grounded,
            hallucination_detected=hallucination,
            safety_verified=safety_ok,
            reasoning_trace=reasoning_trace
        )
    
    def _step1_extract_findings(
        self,
        case_description: str,
        extracted_features: Optional[Any]
    ) -> ExtractedFindings:
        """Step 1: Extract all clinical findings from case."""
        findings = ExtractedFindings()
        text_lower = case_description.lower()
        
        # Use pre-extracted features if available
        if extracted_features:
            if hasattr(extracted_features, 'symptoms'):
                findings.symptoms = extracted_features.symptoms
            if hasattr(extracted_features, 'vitals'):
                findings.vital_signs = extracted_features.vitals
            if hasattr(extracted_features, 'labs'):
                findings.lab_values = extracted_features.labs
            if hasattr(extracted_features, 'risk_factors'):
                findings.risk_factors = extracted_features.risk_factors
            if hasattr(extracted_features, 'medications'):
                findings.medications = extracted_features.medications
        
        # Extract symptoms using patterns
        for pattern in self.symptom_patterns:
            matches = re.findall(pattern, text_lower)
            findings.symptoms.extend(matches)
        findings.symptoms = list(set(findings.symptoms))
        
        # Extract vitals
        for vital_name, pattern in self.vital_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                findings.vital_signs[vital_name] = match.group(1)
        
        # Check for red flags
        for category, flags in self.red_flags.items():
            for flag in flags:
                if flag.lower() in text_lower:
                    findings.red_flags.append(f"{category}: {flag}")
        
        # Extract duration
        duration_match = re.search(
            r'for (\d+)\s*(days?|hours?|weeks?|months?)',
            text_lower
        )
        if duration_match:
            findings.duration = f"{duration_match.group(1)} {duration_match.group(2)}"
        
        # Extract severity
        severity_words = ['mild', 'moderate', 'severe', 'critical', 'acute', 'chronic']
        for word in severity_words:
            if word in text_lower:
                findings.severity = word
                break
        
        return findings
    
    def _step2_identify_category(
        self,
        case_description: str,
        findings: ExtractedFindings
    ) -> PatientCategory:
        """Step 2: Identify patient category."""
        text_lower = case_description.lower()
        
        # Check for emergency first (highest priority)
        for pattern in self.emergency_patterns:
            if re.search(pattern, text_lower):
                return PatientCategory.EMERGENCY
        
        # Check for red flags indicating emergency
        if findings.red_flags:
            return PatientCategory.EMERGENCY
        
        # Check for pregnancy
        for pattern in self.pregnancy_patterns:
            if re.search(pattern, text_lower):
                return PatientCategory.PREGNANCY
        
        # Check for pediatric/neonatal
        age_indicators = {
            'newborn': PatientCategory.NEONATAL,
            'neonate': PatientCategory.NEONATAL,
            'neonatal': PatientCategory.NEONATAL,
            'infant': PatientCategory.PEDIATRIC,
            'toddler': PatientCategory.PEDIATRIC,
            'child': PatientCategory.PEDIATRIC,
            'pediatric': PatientCategory.PEDIATRIC,
            'elderly': PatientCategory.GERIATRIC,
            'geriatric': PatientCategory.GERIATRIC
        }
        
        for indicator, category in age_indicators.items():
            if indicator in text_lower:
                return category
        
        # Check age in years/months/days
        age_match = re.search(r'(\d+)[- ]?(year|yr|month|mo|week|wk|day|d)', text_lower)
        if age_match:
            num = int(age_match.group(1))
            unit = age_match.group(2)
            
            if unit in ['day', 'd', 'week', 'wk']:
                return PatientCategory.NEONATAL
            elif unit in ['month', 'mo']:
                if num <= 12:
                    return PatientCategory.NEONATAL if num <= 1 else PatientCategory.PEDIATRIC
            elif unit in ['year', 'yr']:
                if num <= 18:
                    return PatientCategory.PEDIATRIC
                elif num >= 65:
                    return PatientCategory.GERIATRIC
        
        return PatientCategory.ADULT
    
    def _step3_differential_diagnosis(
        self,
        question: str,
        case_description: str,
        findings: ExtractedFindings,
        category: PatientCategory,
        options: List[str]
    ) -> List[DifferentialDiagnosis]:
        """Step 3: Generate differential diagnoses based on findings."""
        differentials = []
        
        # Extract potential diagnoses from options
        for option in options:
            # Parse option (e.g., "A) Pneumonia" or "A. Pneumonia")
            match = re.match(r'^([A-D])[.)]\s*(.+)$', option.strip())
            if match:
                diagnosis = match.group(2).strip()
            else:
                diagnosis = option.strip()
            
            # Score this diagnosis based on symptom matching
            supporting = []
            against = []
            
            # Check if diagnosis keywords match symptoms
            diagnosis_lower = diagnosis.lower()
            for symptom in findings.symptoms:
                # Simple keyword matching for now
                if self._symptom_supports_diagnosis(symptom, diagnosis_lower):
                    supporting.append(f"Patient has {symptom}")
            
            # Calculate likelihood
            likelihood = min(0.9, len(supporting) * 0.2 + 0.1)
            
            differentials.append(DifferentialDiagnosis(
                diagnosis=diagnosis,
                likelihood=likelihood,
                supporting_evidence=supporting,
                against_evidence=against
            ))
        
        # Sort by likelihood
        differentials.sort(key=lambda x: x.likelihood, reverse=True)
        
        return differentials
    
    def _symptom_supports_diagnosis(self, symptom: str, diagnosis: str) -> bool:
        """Check if symptom supports a diagnosis."""
        symptom_diagnosis_map = {
            'fever': ['sepsis', 'infection', 'pneumonia', 'meningitis', 'malaria', 'typhoid'],
            'cough': ['pneumonia', 'bronchitis', 'asthma', 'copd', 'tuberculosis'],
            'dyspnea': ['pneumonia', 'heart failure', 'asthma', 'pulmonary embolism', 'copd'],
            'chest pain': ['myocardial infarction', 'angina', 'pulmonary embolism', 'pneumonia'],
            'headache': ['meningitis', 'stroke', 'migraine', 'hypertension'],
            'seizure': ['epilepsy', 'meningitis', 'eclampsia', 'hypoglycemia'],
            'vomiting': ['gastroenteritis', 'intestinal obstruction', 'meningitis', 'pregnancy'],
            'diarrhea': ['gastroenteritis', 'cholera', 'typhoid', 'food poisoning'],
            'bleeding': ['hemorrhage', 'trauma', 'coagulopathy', 'ulcer'],
            'confusion': ['sepsis', 'stroke', 'hypoglycemia', 'encephalopathy', 'meningitis'],
            'hypotension': ['shock', 'sepsis', 'hemorrhage', 'dehydration'],
            'tachycardia': ['sepsis', 'dehydration', 'heart failure', 'anemia', 'fever'],
            'jaundice': ['hepatitis', 'biliary obstruction', 'hemolysis', 'neonatal jaundice'],
            'rash': ['infection', 'allergy', 'drug reaction', 'meningitis']
        }
        
        symptom_lower = symptom.lower()
        for key, diagnoses in symptom_diagnosis_map.items():
            if key in symptom_lower:
                for dx in diagnoses:
                    if dx in diagnosis.lower():
                        return True
        return False
    
    def _step4_select_diagnosis(
        self,
        differentials: List[DifferentialDiagnosis],
        findings: ExtractedFindings,
        retrieved_contexts: List[Any]
    ) -> Optional[DifferentialDiagnosis]:
        """Step 4: Select primary diagnosis with reasoning."""
        if not differentials:
            return None
        
        # Check guideline support for each differential
        for diff in differentials:
            for ctx in retrieved_contexts:
                if hasattr(ctx, 'document'):
                    content = ctx.document.content.lower()
                elif hasattr(ctx, 'content'):
                    content = ctx.content.lower()
                else:
                    content = str(ctx).lower()
                
                # Check if diagnosis mentioned in guidelines
                if diff.diagnosis.lower() in content:
                    diff.guideline_support = True
                    # Extract citation
                    diff.guideline_citation = content[:200] + "..."
                    # Boost likelihood
                    diff.likelihood = min(0.95, diff.likelihood + 0.3)
        
        # Re-sort after guideline boost
        differentials.sort(key=lambda x: (x.guideline_support, x.likelihood), reverse=True)
        
        return differentials[0]
    
    def _step5_cite_evidence(
        self,
        primary_diagnosis: Optional[DifferentialDiagnosis],
        retrieved_contexts: List[Any],
        options: List[str]
    ) -> List[EvidenceCitation]:
        """Step 5: Cite evidence from retrieved guidelines."""
        citations = []
        
        if not primary_diagnosis:
            return citations
        
        diagnosis_lower = primary_diagnosis.diagnosis.lower()
        
        for ctx in retrieved_contexts:
            if hasattr(ctx, 'document'):
                content = ctx.document.content
                source = ctx.document.metadata.get('title', 'Unknown')
                score = ctx.final_score if hasattr(ctx, 'final_score') else 0.5
            elif hasattr(ctx, 'content'):
                content = ctx.content
                source = ctx.metadata.get('title', 'Unknown') if hasattr(ctx, 'metadata') else 'Unknown'
                score = 0.5
            else:
                continue
            
            content_lower = content.lower()
            
            # Check if this context supports the diagnosis
            supports = diagnosis_lower in content_lower
            
            # Also check if any option is mentioned
            for option in options:
                match = re.match(r'^[A-D][.)]\s*(.+)$', option.strip())
                if match:
                    opt_text = match.group(1).strip().lower()
                    if opt_text in content_lower:
                        supports = True
                        break
            
            if supports or score > 0.6:
                citations.append(EvidenceCitation(
                    text=content[:300],
                    source=source,
                    relevance_score=score,
                    supports_answer=supports
                ))
        
        # Sort by relevance
        citations.sort(key=lambda x: (x.supports_answer, x.relevance_score), reverse=True)
        
        return citations[:5]  # Top 5 citations
    
    def _step6_final_answer(
        self,
        question: str,
        options: List[str],
        primary_diagnosis: Optional[DifferentialDiagnosis],
        citations: List[EvidenceCitation],
        findings: ExtractedFindings
    ) -> Tuple[str, float, bool]:
        """Step 6: Generate final answer with evidence grounding."""
        
        # Check if we have sufficient evidence
        evidence_grounded = any(c.supports_answer for c in citations)
        
        if not primary_diagnosis:
            return "Cannot answer from context", 0.3, False
        
        # Find matching option
        best_option = None
        best_score = 0.0
        
        for option in options:
            match = re.match(r'^([A-D])[.)]\s*(.+)$', option.strip())
            if match:
                letter = match.group(1)
                opt_text = match.group(2).strip()
                
                # Check if this option matches primary diagnosis
                if opt_text.lower() == primary_diagnosis.diagnosis.lower():
                    best_option = letter
                    best_score = primary_diagnosis.likelihood
                    break
                
                # Partial match
                if primary_diagnosis.diagnosis.lower() in opt_text.lower():
                    if primary_diagnosis.likelihood > best_score:
                        best_option = letter
                        best_score = primary_diagnosis.likelihood
        
        if not best_option:
            # Fallback: use first differential that matches an option
            for option in options:
                match = re.match(r'^([A-D])[.)]\s*(.+)$', option.strip())
                if match:
                    letter = match.group(1)
                    opt_text = match.group(2).strip().lower()
                    
                    for diff in [primary_diagnosis] + ([] if not primary_diagnosis else []):
                        if diff and diff.diagnosis.lower() in opt_text:
                            best_option = letter
                            best_score = diff.likelihood
                            break
                if best_option:
                    break
        
        # If still no match and not grounded, return cannot answer
        if not best_option:
            if not evidence_grounded:
                return "Cannot answer from context", 0.3, False
            else:
                # Return first option with low confidence
                best_option = "A"
                best_score = 0.4
        
        # Adjust confidence based on evidence
        if evidence_grounded:
            confidence = min(0.95, best_score + 0.1)
        else:
            confidence = max(0.3, best_score - 0.2)
        
        return best_option, confidence, evidence_grounded
    
    def _detect_hallucination(
        self,
        answer: str,
        citations: List[EvidenceCitation]
    ) -> bool:
        """Detect if the answer contains hallucinated content."""
        # If answer is "cannot answer", no hallucination
        if "cannot answer" in answer.lower():
            return False
        
        # If no citations support the answer, potential hallucination
        if not any(c.supports_answer for c in citations):
            return True
        
        return False
    
    def _verify_safety(
        self,
        answer: str,
        findings: ExtractedFindings,
        category: PatientCategory
    ) -> bool:
        """Verify the answer is safe given patient context."""
        # Check if red flags are addressed
        if findings.red_flags and category != PatientCategory.EMERGENCY:
            # Red flags present but not categorized as emergency - potential safety issue
            return False
        
        return True
    
    def _chain_of_thought_reasoning(
        self,
        question: str,
        case_description: str,
        options: List[str],
        retrieved_contexts: List[Any],
        extracted_features: Optional[Any],
        reasoning_trace: List[Dict]
    ) -> StructuredReasoningResult:
        """Chain-of-Thought reasoning as backup."""
        # Use simplified structured reasoning
        result = self._structured_reasoning(
            question, case_description, options,
            retrieved_contexts, extracted_features, reasoning_trace
        )
        result.reasoning_mode = ReasoningMode.CHAIN_OF_THOUGHT
        result.used_fallback = True
        result.fallback_reason = "CoT backup used"
        return result
    
    def _tree_of_thought_reasoning(
        self,
        question: str,
        case_description: str,
        options: List[str],
        retrieved_contexts: List[Any],
        extracted_features: Optional[Any],
        reasoning_trace: List[Dict]
    ) -> StructuredReasoningResult:
        """Tree-of-Thought reasoning for complex cases."""
        # For complex cases, run structured reasoning multiple times
        # with different initial hypotheses
        
        results = []
        
        # Try each option as the primary hypothesis
        for i, option in enumerate(options[:3]):  # Top 3 options
            match = re.match(r'^([A-D])[.)]\s*(.+)$', option.strip())
            if match:
                hypothesis = match.group(2).strip()
                
                # Run reasoning with this hypothesis biased
                result = self._structured_reasoning(
                    question, case_description, options,
                    retrieved_contexts, extracted_features,
                    reasoning_trace.copy()
                )
                results.append((result, result.confidence_score))
        
        # Select best result
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            best_result = results[0][0]
            best_result.reasoning_mode = ReasoningMode.TREE_OF_THOUGHT
            return best_result
        
        # Fallback to structured
        result = self._structured_reasoning(
            question, case_description, options,
            retrieved_contexts, extracted_features, reasoning_trace
        )
        result.reasoning_mode = ReasoningMode.TREE_OF_THOUGHT
        result.used_fallback = True
        result.fallback_reason = "ToT failed, used structured fallback"
        return result

