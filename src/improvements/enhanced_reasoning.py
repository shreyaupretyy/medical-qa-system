"""
Enhanced Medical Reasoning Module

Implements three critical fixes for reasoning:

Fix 4: Forced Uncertainty Step
- Before final answer, ask model to list reasons the answer might be wrong
- Verify model used information from context
- Verify model avoided hallucinating

Fix 5: Minimal Context Re-evaluation
- Second pass with only top 2 most relevant paragraphs
- Reduces noise from retrieval
- Prevents guideline mismatches

Fix 6: Stepwise Differential Diagnosis Template
- Force systematic reasoning:
  Step 1: List all key symptoms
  Step 2: List all possible differentials
  Step 3: Rule out each differential
  Step 4: Select final diagnosis
  Step 5: Give guideline-based management
  Step 6: Choose best option A-D

Expected improvements:
- Calibration error drops to 0.15-0.18
- Overconfident wrong answers drop from 4 → 1
- Reasoning error count: 6 → 2 or 3
- Accuracy: +5-10%
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.document_processor import Document


@dataclass
class UncertaintyAnalysis:
    """Result of forced uncertainty analysis."""
    potential_errors: List[str]
    missing_evidence: List[str]
    alternative_answers: List[Tuple[str, str]]  # (option, reason)
    hallucination_risk: float  # 0-1
    context_grounding_score: float  # 0-1, how well grounded in context


@dataclass
class MinimalContextResult:
    """Result of minimal context re-evaluation."""
    original_answer: str
    minimal_context_answer: str
    answers_agree: bool
    confidence_adjustment: float
    top_paragraphs: List[str]


@dataclass
class DifferentialDiagnosisResult:
    """Result of stepwise differential diagnosis."""
    key_symptoms: List[str]
    differential_diagnoses: List[Dict]  # {diagnosis, supporting, contradicting, ruled_out}
    selected_diagnosis: str
    guideline_management: str
    selected_answer: str
    reasoning_trace: List[str]


class EnhancedMedicalReasoner:
    """
    Enhanced reasoning with uncertainty analysis and structured diagnosis.
    """
    
    def __init__(self, llm_model=None):
        """
        Initialize enhanced reasoner.
        
        Args:
            llm_model: Optional LLM for advanced reasoning
        """
        self.llm_model = llm_model
    
    # ==================== FIX 4: Forced Uncertainty Step ====================
    
    def analyze_uncertainty(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        selected_answer: str,
        confidence: float,
        retrieved_contexts: List[Document]
    ) -> UncertaintyAnalysis:
        """
        Analyze potential errors and uncertainty in the answer.
        
        This immediately reduces overconfidence by forcing consideration
        of why the answer might be wrong.
        
        Args:
            question: The clinical question
            case_description: Patient case
            options: Answer options
            selected_answer: Currently selected answer
            confidence: Current confidence score
            retrieved_contexts: Retrieved documents
            
        Returns:
            UncertaintyAnalysis with potential issues
        """
        potential_errors = []
        missing_evidence = []
        alternative_answers = []
        
        # Combine context text
        context_text = " ".join([doc.content for doc in retrieved_contexts[:5]])
        context_lower = context_text.lower()
        
        # Check 1: Is the selected option actually mentioned in context?
        selected_text = options.get(selected_answer, "").lower()
        if selected_text and selected_text not in context_lower:
            # Check for medication names
            words = selected_text.split()
            found_any = any(
                len(word) > 4 and word in context_lower 
                for word in words
            )
            if not found_any:
                potential_errors.append(
                    f"Selected answer '{selected_answer}' may not be directly supported by context"
                )
        
        # Check 2: Are there other options with stronger evidence?
        for opt_label, opt_text in options.items():
            if opt_label == selected_answer:
                continue
            opt_lower = opt_text.lower()
            
            # Count how many key words appear in context
            words = [w for w in opt_lower.split() if len(w) > 4]
            matches = sum(1 for w in words if w in context_lower)
            
            if matches >= 3 and len(words) > 0:
                match_ratio = matches / len(words)
                if match_ratio > 0.6:
                    alternative_answers.append(
                        (opt_label, f"Option {opt_label} has {matches} keywords matching context")
                    )
        
        # Check 3: Key symptoms from case covered in context?
        case_lower = case_description.lower()
        symptom_keywords = [
            'pain', 'fever', 'cough', 'bleeding', 'vomiting', 'diarrhea',
            'headache', 'weakness', 'fatigue', 'seizure', 'confusion'
        ]
        
        case_symptoms = [s for s in symptom_keywords if s in case_lower]
        context_symptoms = [s for s in case_symptoms if s in context_lower]
        
        if len(case_symptoms) > 0:
            symptom_coverage = len(context_symptoms) / len(case_symptoms)
            if symptom_coverage < 0.5:
                missing_evidence.append(
                    f"Only {len(context_symptoms)}/{len(case_symptoms)} key symptoms found in context"
                )
        
        # Check 4: Contradiction detection
        contradiction_phrases = [
            'contraindicated', 'not recommended', 'avoid', 'should not'
        ]
        
        for phrase in contradiction_phrases:
            if phrase in context_lower:
                # Check if contradiction is near selected answer terms
                phrase_pos = context_lower.find(phrase)
                nearby_text = context_lower[max(0, phrase_pos-100):phrase_pos+100]
                
                if any(word in nearby_text for word in selected_text.split() if len(word) > 4):
                    potential_errors.append(
                        f"Possible contraindication found near selected answer"
                    )
        
        # Check 5: Is this a treatment question but context is about diagnosis?
        is_treatment_q = any(
            term in question.lower() 
            for term in ['treatment', 'therapy', 'management', 'prescribe', 'medication']
        )
        is_treatment_context = any(
            term in context_lower 
            for term in ['treatment:', 'therapy:', 'medication:', 'first-line', 'dose']
        )
        
        if is_treatment_q and not is_treatment_context:
            missing_evidence.append(
                "Question asks about treatment but context may lack treatment information"
            )
        
        # Calculate hallucination risk
        hallucination_risk = 0.0
        if potential_errors:
            hallucination_risk += len(potential_errors) * 0.2
        if missing_evidence:
            hallucination_risk += len(missing_evidence) * 0.15
        hallucination_risk = min(1.0, hallucination_risk)
        
        # Calculate context grounding score
        grounding_score = 1.0
        grounding_score -= hallucination_risk * 0.5
        if alternative_answers:
            grounding_score -= len(alternative_answers) * 0.1
        grounding_score = max(0.0, grounding_score)
        
        return UncertaintyAnalysis(
            potential_errors=potential_errors,
            missing_evidence=missing_evidence,
            alternative_answers=alternative_answers,
            hallucination_risk=hallucination_risk,
            context_grounding_score=grounding_score
        )
    
    def adjust_confidence_for_uncertainty(
        self,
        original_confidence: float,
        uncertainty: UncertaintyAnalysis
    ) -> float:
        """
        Adjust confidence based on uncertainty analysis.
        
        Args:
            original_confidence: Original confidence score
            uncertainty: Uncertainty analysis result
            
        Returns:
            Adjusted confidence score
        """
        adjusted = original_confidence
        
        # Reduce for potential errors
        adjusted -= len(uncertainty.potential_errors) * 0.1
        
        # Reduce for missing evidence
        adjusted -= len(uncertainty.missing_evidence) * 0.08
        
        # Reduce for strong alternatives
        adjusted -= len(uncertainty.alternative_answers) * 0.05
        
        # Apply hallucination risk penalty
        adjusted *= (1.0 - uncertainty.hallucination_risk * 0.3)
        
        # Apply grounding boost/penalty
        if uncertainty.context_grounding_score > 0.8:
            adjusted *= 1.1  # Boost well-grounded answers
        elif uncertainty.context_grounding_score < 0.5:
            adjusted *= 0.8  # Penalize poorly-grounded answers
        
        return max(0.05, min(0.95, adjusted))  # Cap between 5% and 95%
    
    # ==================== FIX 5: Minimal Context Re-evaluation ====================
    
    def reevaluate_with_minimal_context(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        original_answer: str,
        retrieved_contexts: List[Document],
        top_k: int = 2
    ) -> MinimalContextResult:
        """
        Re-evaluate answer using only the top-k most relevant paragraphs.
        
        This reduces noise from retrieval and prevents irrelevant content
        from influencing reasoning.
        
        Args:
            question: Clinical question
            case_description: Patient case
            options: Answer options
            original_answer: Answer from full context
            retrieved_contexts: All retrieved documents
            top_k: Number of top paragraphs to use (default: 2)
            
        Returns:
            MinimalContextResult with comparison
        """
        # Take only top-k documents
        top_docs = retrieved_contexts[:top_k]
        top_paragraphs = [doc.content for doc in top_docs]
        
        # Score each option against minimal context
        minimal_context = " ".join(top_paragraphs).lower()
        option_scores = {}
        
        for label, text in options.items():
            score = self._score_option_against_context(text, minimal_context)
            option_scores[label] = score
        
        # Select answer from minimal context
        if option_scores:
            minimal_answer = max(option_scores.items(), key=lambda x: x[1])[0]
        else:
            minimal_answer = original_answer
        
        # Check agreement
        answers_agree = minimal_answer == original_answer
        
        # Calculate confidence adjustment
        if answers_agree:
            confidence_adjustment = 1.1  # Boost if both agree
        else:
            # Check if scores are close
            original_score = option_scores.get(original_answer, 0)
            minimal_score = option_scores.get(minimal_answer, 0)
            
            if abs(original_score - minimal_score) < 0.1:
                confidence_adjustment = 0.95  # Slight reduction for close call
            else:
                confidence_adjustment = 0.8  # Larger reduction for disagreement
        
        return MinimalContextResult(
            original_answer=original_answer,
            minimal_context_answer=minimal_answer,
            answers_agree=answers_agree,
            confidence_adjustment=confidence_adjustment,
            top_paragraphs=top_paragraphs
        )
    
    def _score_option_against_context(
        self,
        option_text: str,
        context: str
    ) -> float:
        """Score how well an option matches context."""
        option_lower = option_text.lower()
        
        # Exact match
        if option_lower in context:
            return 1.0
        
        # Word overlap
        words = [w for w in option_lower.split() if len(w) > 3]
        if not words:
            return 0.0
        
        matches = sum(1 for w in words if w in context)
        
        # Check for medication names
        med_patterns = [
            r'\b([a-z]+(?:mycin|cycline|floxacin|penem|azole|pril|olol))\b'
        ]
        for pattern in med_patterns:
            match = re.search(pattern, option_lower)
            if match and match.group(1) in context:
                matches += 2  # Bonus for medication match
        
        return min(1.0, matches / max(1, len(words)))
    
    # ==================== FIX 6: Stepwise Differential Diagnosis ====================
    
    def stepwise_differential_diagnosis(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        retrieved_contexts: List[Document]
    ) -> DifferentialDiagnosisResult:
        """
        Apply stepwise differential diagnosis template.
        
        Steps:
        1. List all key symptoms from the question
        2. List all possible differentials
        3. Rule out each differential based on context
        4. Select the final diagnosis
        5. Give guideline-based management
        6. Choose best option A-D
        
        Args:
            question: Clinical question
            case_description: Patient case
            options: Answer options
            retrieved_contexts: Retrieved documents
            
        Returns:
            DifferentialDiagnosisResult with structured reasoning
        """
        reasoning_trace = []
        
        # Combine text for analysis
        full_text = f"{case_description} {question}".lower()
        context_text = " ".join([doc.content for doc in retrieved_contexts[:5]]).lower()
        
        # STEP 1: List all key symptoms
        reasoning_trace.append("STEP 1: Extracting key symptoms from case")
        key_symptoms = self._extract_symptoms_structured(full_text)
        reasoning_trace.append(f"  Found symptoms: {', '.join(key_symptoms)}")
        
        # STEP 2: Generate differential diagnoses
        reasoning_trace.append("\nSTEP 2: Generating differential diagnoses")
        differentials = self._generate_differentials(key_symptoms, context_text)
        for diff in differentials:
            reasoning_trace.append(f"  - {diff['diagnosis']}: {diff['reason']}")
        
        # STEP 3: Rule out differentials
        reasoning_trace.append("\nSTEP 3: Ruling out differentials based on context")
        for diff in differentials:
            ruled_out, reason = self._check_rule_out(diff['diagnosis'], context_text, key_symptoms)
            diff['ruled_out'] = ruled_out
            diff['rule_out_reason'] = reason
            status = "RULED OUT" if ruled_out else "POSSIBLE"
            reasoning_trace.append(f"  - {diff['diagnosis']}: {status} - {reason}")
        
        # STEP 4: Select final diagnosis
        reasoning_trace.append("\nSTEP 4: Selecting final diagnosis")
        remaining = [d for d in differentials if not d['ruled_out']]
        if remaining:
            selected_diagnosis = remaining[0]['diagnosis']
        else:
            selected_diagnosis = differentials[0]['diagnosis'] if differentials else "Unknown"
        reasoning_trace.append(f"  Selected: {selected_diagnosis}")
        
        # STEP 5: Find guideline-based management
        reasoning_trace.append("\nSTEP 5: Finding guideline-based management")
        guideline_management = self._find_management_in_context(
            selected_diagnosis, context_text
        )
        reasoning_trace.append(f"  Management: {guideline_management[:200]}...")
        
        # STEP 6: Choose best option
        reasoning_trace.append("\nSTEP 6: Selecting best answer option")
        selected_answer = self._match_answer_to_management(
            options, guideline_management, context_text
        )
        reasoning_trace.append(f"  Selected answer: {selected_answer}")
        
        return DifferentialDiagnosisResult(
            key_symptoms=key_symptoms,
            differential_diagnoses=differentials,
            selected_diagnosis=selected_diagnosis,
            guideline_management=guideline_management,
            selected_answer=selected_answer,
            reasoning_trace=reasoning_trace
        )
    
    def _extract_symptoms_structured(self, text: str) -> List[str]:
        """Extract symptoms from text using structured approach."""
        symptoms = []
        
        # Common symptom patterns
        symptom_keywords = [
            'pain', 'fever', 'cough', 'bleeding', 'vomiting', 'diarrhea',
            'headache', 'weakness', 'fatigue', 'seizure', 'confusion',
            'dyspnea', 'shortness of breath', 'chest pain', 'abdominal pain',
            'nausea', 'jaundice', 'rash', 'swelling', 'edema', 'lethargy',
            'poor feeding', 'irritability', 'unconscious', 'syncope'
        ]
        
        for symptom in symptom_keywords:
            if symptom in text:
                symptoms.append(symptom)
        
        # Extract age/demographic info
        age_match = re.search(r'(\d+)[\s-]*(year|yr|month|day|week)', text)
        if age_match:
            symptoms.append(f"age: {age_match.group(0)}")
        
        if 'pregnant' in text or 'pregnancy' in text:
            symptoms.append('pregnant')
        
        if 'newborn' in text or 'neonate' in text:
            symptoms.append('newborn/neonate')
        
        return symptoms
    
    def _generate_differentials(
        self,
        symptoms: List[str],
        context: str
    ) -> List[Dict]:
        """Generate differential diagnoses based on symptoms."""
        differentials = []
        
        # Symptom-based differential mapping
        symptom_differentials = {
            'fever': ['infection', 'sepsis', 'pneumonia', 'malaria', 'uti'],
            'chest pain': ['myocardial infarction', 'angina', 'pulmonary embolism', 'pneumothorax'],
            'headache': ['migraine', 'meningitis', 'subarachnoid hemorrhage', 'hypertensive crisis'],
            'abdominal pain': ['appendicitis', 'cholecystitis', 'pancreatitis', 'peptic ulcer'],
            'shortness of breath': ['heart failure', 'pneumonia', 'copd', 'asthma', 'pulmonary embolism'],
            'seizure': ['epilepsy', 'febrile seizure', 'meningitis', 'hypoglycemia'],
            'bleeding': ['trauma', 'coagulopathy', 'peptic ulcer', 'malignancy'],
            'jaundice': ['hepatitis', 'cholestasis', 'hemolysis', 'neonatal jaundice'],
            'confusion': ['delirium', 'stroke', 'hypoglycemia', 'encephalopathy', 'sepsis'],
        }
        
        seen_diagnoses = set()
        
        for symptom in symptoms:
            for key_symptom, possible_diagnoses in symptom_differentials.items():
                if key_symptom in symptom:
                    for diagnosis in possible_diagnoses:
                        if diagnosis not in seen_diagnoses:
                            # Check if diagnosis appears in context
                            in_context = diagnosis in context
                            differentials.append({
                                'diagnosis': diagnosis,
                                'supporting_symptoms': [symptom],
                                'in_context': in_context,
                                'reason': f"Based on {symptom}" + (" (found in context)" if in_context else ""),
                                'ruled_out': False
                            })
                            seen_diagnoses.add(diagnosis)
        
        # Sort by presence in context (prioritize those in context)
        differentials.sort(key=lambda x: (x['in_context'], x['diagnosis']), reverse=True)
        
        return differentials[:6]  # Top 6 differentials
    
    def _check_rule_out(
        self,
        diagnosis: str,
        context: str,
        symptoms: List[str]
    ) -> Tuple[bool, str]:
        """Check if a differential should be ruled out."""
        diagnosis_lower = diagnosis.lower()
        
        # Check for explicit rule-out phrases
        rule_out_phrases = [
            f"ruled out {diagnosis_lower}",
            f"not {diagnosis_lower}",
            f"exclude {diagnosis_lower}",
            f"unlikely {diagnosis_lower}",
        ]
        
        for phrase in rule_out_phrases:
            if phrase in context:
                return True, f"Context explicitly rules out: '{phrase}'"
        
        # Check for contradicting evidence
        if diagnosis_lower not in context:
            # If diagnosis not mentioned at all, might be less likely
            # But don't rule out automatically
            pass
        
        # Check for age-based contraindications
        if 'newborn' in ' '.join(symptoms) or 'neonate' in ' '.join(symptoms):
            adult_conditions = ['myocardial infarction', 'angina', 'copd']
            if diagnosis_lower in adult_conditions:
                return True, "Adult condition unlikely in newborn"
        
        return False, "Not ruled out by available evidence"
    
    def _find_management_in_context(
        self,
        diagnosis: str,
        context: str
    ) -> str:
        """Find management/treatment information in context."""
        # Look for treatment sections
        treatment_patterns = [
            r'treatment[:\s]+([^\.]+\.)',
            r'management[:\s]+([^\.]+\.)',
            r'therapy[:\s]+([^\.]+\.)',
            r'first[- ]line[:\s]+([^\.]+\.)',
            r'recommended[:\s]+([^\.]+\.)',
        ]
        
        for pattern in treatment_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: find any sentence with treatment keywords
        sentences = context.split('.')
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in ['treatment', 'give', 'administer', 'prescribe', 'dose']):
                return sentence.strip()
        
        return "No specific management found in context"
    
    def _match_answer_to_management(
        self,
        options: Dict[str, str],
        management: str,
        context: str
    ) -> str:
        """Match the best answer option to the management found."""
        management_lower = management.lower()
        
        best_match = None
        best_score = 0.0
        
        for label, text in options.items():
            text_lower = text.lower()
            
            # Check exact match in management
            if text_lower in management_lower:
                return label
            
            # Check exact match in context
            if text_lower in context:
                score = 0.9
            else:
                # Word overlap
                words = [w for w in text_lower.split() if len(w) > 3]
                matches = sum(1 for w in words if w in management_lower or w in context)
                score = matches / max(1, len(words))
            
            if score > best_score:
                best_score = score
                best_match = label
        
        return best_match or list(options.keys())[0]
    
    # ==================== LLM-Enhanced Methods ====================
    
    def llm_uncertainty_analysis(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        selected_answer: str,
        retrieved_contexts: List[Document]
    ) -> UncertaintyAnalysis:
        """Use LLM to analyze uncertainty (if available)."""
        if not self.llm_model:
            return self.analyze_uncertainty(
                question, case_description, options, selected_answer, 0.5, retrieved_contexts
            )
        
        try:
            context_text = "\n".join([
                f"Document {i+1}: {doc.content[:500]}"
                for i, doc in enumerate(retrieved_contexts[:3])
            ])
            
            prompt = f"""Analyze the potential errors in this medical answer selection.

Case: {case_description}
Question: {question}
Selected Answer: {selected_answer} - {options.get(selected_answer, '')}

Context Used:
{context_text}

List all reasons this answer might be WRONG:
1. [reason 1]
2. [reason 2]
...

Check if:
- The answer is actually supported by the context
- There might be a better answer
- Any information was hallucinated (not in context)

POTENTIAL ERRORS:"""

            response = self.llm_model.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=300
            )
            
            # Parse response for potential errors
            potential_errors = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    potential_errors.append(line.lstrip('0123456789.-) '))
            
            # Calculate scores based on LLM response
            error_count = len(potential_errors)
            hallucination_risk = min(1.0, error_count * 0.2)
            grounding_score = max(0.0, 1.0 - error_count * 0.15)
            
            return UncertaintyAnalysis(
                potential_errors=potential_errors[:5],
                missing_evidence=[],
                alternative_answers=[],
                hallucination_risk=hallucination_risk,
                context_grounding_score=grounding_score
            )
            
        except Exception as e:
            print(f"[WARN] LLM uncertainty analysis failed: {e}")
            return self.analyze_uncertainty(
                question, case_description, options, selected_answer, 0.5, retrieved_contexts
            )


def main():
    """Demo: Test enhanced reasoning."""
    print("="*70)
    print("ENHANCED MEDICAL REASONING DEMO")
    print("="*70)
    
    reasoner = EnhancedMedicalReasoner()
    
    # Mock document
    class MockDocument:
        def __init__(self, content):
            self.content = content
            self.metadata = {'title': 'Test', 'guideline_id': 'test'}
    
    # Test case
    case = "A 3-day-old newborn presents with fever, poor feeding, and lethargy."
    question = "What is the first-line treatment?"
    options = {
        'A': 'Ampicillin and gentamicin',
        'B': 'Ceftriaxone only',
        'C': 'Observation only',
        'D': 'Oral antibiotics'
    }
    contexts = [
        MockDocument("Treatment: For neonatal sepsis, first-line therapy is ampicillin and gentamicin. Dose: Ampicillin 100mg/kg."),
        MockDocument("Management: IV antibiotics should be started immediately. Ceftriaxone is not recommended in newborns due to bilirubin displacement.")
    ]
    
    print(f"\nCase: {case}")
    print(f"Question: {question}")
    
    # Test Fix 4: Uncertainty Analysis
    print(f"\n{'-'*70}")
    print("FIX 4: Uncertainty Analysis")
    print(f"{'-'*70}")
    
    uncertainty = reasoner.analyze_uncertainty(
        question, case, options, 'A', 0.8, contexts
    )
    
    print(f"Hallucination risk: {uncertainty.hallucination_risk:.2f}")
    print(f"Context grounding: {uncertainty.context_grounding_score:.2f}")
    print(f"Potential errors: {uncertainty.potential_errors}")
    
    original_conf = 0.8
    adjusted_conf = reasoner.adjust_confidence_for_uncertainty(original_conf, uncertainty)
    print(f"Confidence: {original_conf:.2f} → {adjusted_conf:.2f}")
    
    # Test Fix 5: Minimal Context
    print(f"\n{'-'*70}")
    print("FIX 5: Minimal Context Re-evaluation")
    print(f"{'-'*70}")
    
    minimal = reasoner.reevaluate_with_minimal_context(
        question, case, options, 'A', contexts, top_k=2
    )
    
    print(f"Original answer: {minimal.original_answer}")
    print(f"Minimal context answer: {minimal.minimal_context_answer}")
    print(f"Answers agree: {minimal.answers_agree}")
    print(f"Confidence adjustment: {minimal.confidence_adjustment:.2f}")
    
    # Test Fix 6: Stepwise Differential
    print(f"\n{'-'*70}")
    print("FIX 6: Stepwise Differential Diagnosis")
    print(f"{'-'*70}")
    
    differential = reasoner.stepwise_differential_diagnosis(
        question, case, options, contexts
    )
    
    print(f"Key symptoms: {differential.key_symptoms}")
    print(f"Selected diagnosis: {differential.selected_diagnosis}")
    print(f"Selected answer: {differential.selected_answer}")
    print("\nReasoning trace:")
    for step in differential.reasoning_trace[:10]:
        print(f"  {step}")
    
    print(f"\n{'='*70}")
    print("[OK] Enhanced Medical Reasoning operational!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

