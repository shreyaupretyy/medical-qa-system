"""
Reasoning Improvements for Medical QA System

Fix 4: Forced Uncertainty Step - Ask model to list reasons the answer might be wrong
Fix 5: Minimal Context Re-evaluation - Second pass with only top 2 paragraphs
Fix 6: Stepwise Differential Diagnosis Template - Structured reasoning

Expected improvements:
- Calibration error drops from 0.30 to 0.15-0.18
- Overconfident wrong answers drop from 4 to 1
- Accuracy improves 4-10%
- Reasoning errors drop from 6 to 2-3
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class UncertaintyAnalysis:
    """Result of uncertainty analysis."""
    original_answer: str
    original_confidence: float
    potential_errors: List[str]
    verified_answer: str
    adjusted_confidence: float
    used_context: bool
    no_hallucination: bool


@dataclass
class MinimalContextResult:
    """Result of minimal context re-evaluation."""
    top_paragraphs: List[str]
    original_answer: str
    minimal_context_answer: str
    answers_match: bool
    final_answer: str
    final_confidence: float


@dataclass
class DifferentialDiagnosisResult:
    """Result of differential diagnosis reasoning."""
    step1_symptoms: List[str]
    step2_differentials: List[str]
    step3_ruled_out: Dict[str, str]  # diagnosis -> reason ruled out
    step4_final_diagnosis: str
    step5_management: str
    step6_selected_option: str
    confidence: float
    reasoning_chain: str


class ForcedUncertaintyAnalyzer:
    """
    Fix 4: Force the model to consider why the answer might be wrong.
    
    Before final answer, ask model to list all reasons the answer might be wrong.
    This immediately reduces overconfidence.
    """
    
    def __init__(self, llm_model=None):
        """Initialize with LLM model."""
        self.llm_model = llm_model
        if llm_model is None:
            try:
                from models.ollama_model import OllamaModel
                self.llm_model = OllamaModel(
                    model_name="llama3.1:8b",
                    temperature=0.2,
                    max_tokens=512
                )
            except Exception as e:
                print(f"[WARN] Could not initialize LLM for uncertainty analysis: {e}")
                self.llm_model = None
    
    def analyze_uncertainty(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        selected_answer: str,
        confidence: float,
        context: List[str],
        reasoning: str = ""
    ) -> UncertaintyAnalysis:
        """
        Analyze potential errors and adjust confidence.
        
        Args:
            question: The clinical question
            case_description: Patient case description
            options: Answer options
            selected_answer: Currently selected answer
            confidence: Current confidence score
            context: Retrieved context documents
            reasoning: Current reasoning chain
            
        Returns:
            UncertaintyAnalysis with adjusted confidence
        """
        if not self.llm_model:
            return UncertaintyAnalysis(
                original_answer=selected_answer,
                original_confidence=confidence,
                potential_errors=[],
                verified_answer=selected_answer,
                adjusted_confidence=confidence,
                used_context=True,
                no_hallucination=True
            )
        
        try:
            return self._llm_uncertainty_analysis(
                question, case_description, options, selected_answer,
                confidence, context, reasoning
            )
        except Exception as e:
            print(f"[WARN] Uncertainty analysis failed: {e}")
            return UncertaintyAnalysis(
                original_answer=selected_answer,
                original_confidence=confidence,
                potential_errors=[],
                verified_answer=selected_answer,
                adjusted_confidence=confidence,
                used_context=True,
                no_hallucination=True
            )
    
    def _llm_uncertainty_analysis(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        selected_answer: str,
        confidence: float,
        context: List[str],
        reasoning: str
    ) -> UncertaintyAnalysis:
        """Use LLM to analyze uncertainty."""
        context_text = "\n".join(context[:3])  # Use top 3 contexts
        option_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        prompt = f"""You are a medical expert reviewing an answer selection. Your task is to critically evaluate the selected answer and identify potential errors.

Clinical Case:
{case_description}

Question: {question}

Options:
{option_text}

Selected Answer: {selected_answer}
Current Confidence: {confidence:.0%}

Context Used:
{context_text[:1500]}

Previous Reasoning:
{reasoning[:500]}

CRITICAL ANALYSIS TASK:
List ALL reasons why the selected answer "{selected_answer}" might be WRONG. Be thorough and skeptical.

Consider:
1. Does the context ACTUALLY support this answer, or was it assumed?
2. Could another option be more appropriate for this patient?
3. Are there contraindications or patient-specific factors that weren't considered?
4. Is there information from the context that was ignored?
5. Could the reasoning have jumped to conclusions?

Format your response exactly as:
POTENTIAL_ERRORS:
- [error 1]
- [error 2]
- [error 3]

CONTEXT_VERIFICATION: [YES if answer is supported by context, NO if assumed]
HALLUCINATION_CHECK: [YES if all facts come from context, NO if external knowledge used]
SHOULD_CHANGE_ANSWER: [YES/NO]
BETTER_ANSWER: [if YES, specify which option]
CONFIDENCE_ADJUSTMENT: [increase/decrease/maintain]
"""
        
        response = self.llm_model.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=512
        )
        
        # Parse response
        potential_errors = []
        error_section = re.search(r'POTENTIAL_ERRORS:(.*?)(?:CONTEXT_VERIFICATION|$)', response, re.DOTALL | re.IGNORECASE)
        if error_section:
            errors = re.findall(r'-\s*(.+)', error_section.group(1))
            potential_errors = [e.strip() for e in errors if e.strip()]
        
        # Check context verification
        context_match = re.search(r'CONTEXT_VERIFICATION:\s*(YES|NO)', response, re.IGNORECASE)
        used_context = context_match and context_match.group(1).upper() == 'YES' if context_match else True
        
        # Check hallucination
        halluc_match = re.search(r'HALLUCINATION_CHECK:\s*(YES|NO)', response, re.IGNORECASE)
        no_hallucination = halluc_match and halluc_match.group(1).upper() == 'YES' if halluc_match else True
        
        # Check if should change answer
        change_match = re.search(r'SHOULD_CHANGE_ANSWER:\s*(YES|NO)', response, re.IGNORECASE)
        should_change = change_match and change_match.group(1).upper() == 'YES' if change_match else False
        
        # Get better answer if needed
        verified_answer = selected_answer
        if should_change:
            better_match = re.search(r'BETTER_ANSWER:\s*([A-D])', response, re.IGNORECASE)
            if better_match:
                verified_answer = better_match.group(1).upper()
        
        # Adjust confidence
        adjusted_confidence = confidence
        conf_match = re.search(r'CONFIDENCE_ADJUSTMENT:\s*(increase|decrease|maintain)', response, re.IGNORECASE)
        if conf_match:
            adjustment = conf_match.group(1).lower()
            if adjustment == 'decrease':
                # More errors = larger decrease
                error_penalty = min(0.3, len(potential_errors) * 0.08)
                adjusted_confidence = max(0.2, confidence - error_penalty)
                # Additional penalty if context not used properly
                if not used_context:
                    adjusted_confidence = max(0.15, adjusted_confidence - 0.15)
                if not no_hallucination:
                    adjusted_confidence = max(0.15, adjusted_confidence - 0.20)
            elif adjustment == 'increase':
                adjusted_confidence = min(0.95, confidence + 0.1)
        
        return UncertaintyAnalysis(
            original_answer=selected_answer,
            original_confidence=confidence,
            potential_errors=potential_errors,
            verified_answer=verified_answer,
            adjusted_confidence=adjusted_confidence,
            used_context=used_context,
            no_hallucination=no_hallucination
        )


class MinimalContextEvaluator:
    """
    Fix 5: Re-evaluate with only the top 2 most relevant paragraphs.
    
    Reduces noise from irrelevant retrieved documents.
    Prevents guideline mismatches.
    Prevents irrelevant content from influencing reasoning.
    """
    
    def __init__(self, llm_model=None):
        """Initialize with LLM model."""
        self.llm_model = llm_model
        if llm_model is None:
            try:
                from models.ollama_model import OllamaModel
                self.llm_model = OllamaModel(
                    model_name="llama3.1:8b",
                    temperature=0.1,
                    max_tokens=384
                )
            except Exception as e:
                print(f"[WARN] Could not initialize LLM for minimal context: {e}")
                self.llm_model = None
    
    def evaluate(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        original_answer: str,
        original_confidence: float,
        all_contexts: List[str]
    ) -> MinimalContextResult:
        """
        Re-evaluate answer using only top 2 paragraphs.
        
        Args:
            question: The clinical question
            case_description: Patient case description
            options: Answer options
            original_answer: Initially selected answer
            original_confidence: Original confidence score
            all_contexts: All retrieved context documents
            
        Returns:
            MinimalContextResult with final answer
        """
        if not self.llm_model or len(all_contexts) < 2:
            return MinimalContextResult(
                top_paragraphs=all_contexts[:2] if all_contexts else [],
                original_answer=original_answer,
                minimal_context_answer=original_answer,
                answers_match=True,
                final_answer=original_answer,
                final_confidence=original_confidence
            )
        
        try:
            return self._llm_minimal_context_eval(
                question, case_description, options, original_answer,
                original_confidence, all_contexts
            )
        except Exception as e:
            print(f"[WARN] Minimal context evaluation failed: {e}")
            return MinimalContextResult(
                top_paragraphs=all_contexts[:2],
                original_answer=original_answer,
                minimal_context_answer=original_answer,
                answers_match=True,
                final_answer=original_answer,
                final_confidence=original_confidence
            )
    
    def _llm_minimal_context_eval(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        original_answer: str,
        original_confidence: float,
        all_contexts: List[str]
    ) -> MinimalContextResult:
        """Use LLM to re-evaluate with minimal context."""
        # Get only top 2 paragraphs
        top_paragraphs = all_contexts[:2]
        context_text = "\n\n---\n\n".join(top_paragraphs)
        option_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        prompt = f"""You are answering a medical question using ONLY the provided context. Use nothing else.

Case:
{case_description}

Question: {question}

Options:
{option_text}

CONTEXT (USE ONLY THIS - NOTHING ELSE):
{context_text[:2000]}

Based ONLY on the context above, which option is correct?

Think step-by-step:
1. What does the context say about this clinical scenario?
2. Which option is directly supported by the context?
3. If no option is clearly supported, say "Cannot determine".

Answer format: "The answer is [A/B/C/D]" or "Cannot determine from context"
"""
        
        response = self.llm_model.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=256
        )
        
        # Parse answer
        minimal_answer = original_answer  # Default to original
        answer_match = re.search(r'(?:answer is|select|choose)\s*\[?([A-D])\]?', response, re.IGNORECASE)
        if answer_match:
            minimal_answer = answer_match.group(1).upper()
        elif "cannot determine" in response.lower():
            # Keep original but lower confidence
            minimal_answer = original_answer
        
        answers_match = minimal_answer == original_answer
        
        # Determine final answer and confidence
        if answers_match:
            # Agreement: boost confidence slightly
            final_answer = original_answer
            final_confidence = min(0.95, original_confidence + 0.05)
        else:
            # Disagreement: use original but reduce confidence
            final_answer = original_answer
            final_confidence = max(0.25, original_confidence - 0.15)
            # If minimal context answer has strong support, consider switching
            if "clearly" in response.lower() or "definitely" in response.lower():
                final_answer = minimal_answer
                final_confidence = 0.65
        
        return MinimalContextResult(
            top_paragraphs=top_paragraphs,
            original_answer=original_answer,
            minimal_context_answer=minimal_answer,
            answers_match=answers_match,
            final_answer=final_answer,
            final_confidence=final_confidence
        )


class DifferentialDiagnosisReasoner:
    """
    Fix 6: Force stepwise differential diagnosis template.
    
    Template:
    Step 1: List all key symptoms from the question
    Step 2: List all possible differentials
    Step 3: Rule out each differential based on context
    Step 4: Select the final diagnosis
    Step 5: Give guideline-based management
    Step 6: Choose best option from A-D
    """
    
    def __init__(self, llm_model=None):
        """Initialize with LLM model."""
        self.llm_model = llm_model
        if llm_model is None:
            try:
                from models.ollama_model import OllamaModel
                self.llm_model = OllamaModel(
                    model_name="llama3.1:8b",
                    temperature=0.1,
                    max_tokens=768
                )
            except Exception as e:
                print(f"[WARN] Could not initialize LLM for differential diagnosis: {e}")
                self.llm_model = None
    
    def reason(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        context: List[str]
    ) -> DifferentialDiagnosisResult:
        """
        Apply stepwise differential diagnosis reasoning.
        
        Args:
            question: The clinical question
            case_description: Patient case description
            options: Answer options
            context: Retrieved context documents
            
        Returns:
            DifferentialDiagnosisResult with structured reasoning
        """
        if not self.llm_model:
            return self._rule_based_reasoning(question, case_description, options, context)
        
        try:
            return self._llm_differential_reasoning(question, case_description, options, context)
        except Exception as e:
            print(f"[WARN] Differential diagnosis reasoning failed: {e}")
            return self._rule_based_reasoning(question, case_description, options, context)
    
    def _llm_differential_reasoning(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        context: List[str]
    ) -> DifferentialDiagnosisResult:
        """Use LLM for structured differential diagnosis."""
        context_text = "\n\n".join(context[:4])  # Top 4 contexts
        option_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        prompt = f"""You are a clinical reasoning expert. Follow the EXACT 6-step differential diagnosis process below.

Clinical Case:
{case_description}

Question: {question}

Options:
{option_text}

Medical Guidelines Context:
{context_text[:2500]}

===== MANDATORY 6-STEP DIFFERENTIAL DIAGNOSIS =====

STEP 1 - LIST KEY SYMPTOMS:
Extract all symptoms, signs, and clinical findings from the case:
- Symptom 1: 
- Symptom 2:
- Symptom 3:
- (list all relevant findings)

STEP 2 - LIST POSSIBLE DIFFERENTIALS:
Based on the symptoms, what are the possible diagnoses?
- Differential 1:
- Differential 2:
- Differential 3:
- (list 3-5 differentials)

STEP 3 - RULE OUT DIFFERENTIALS:
For each differential, explain why it IS or IS NOT the likely diagnosis:
- [Differential 1]: [RULED OUT/LIKELY] because...
- [Differential 2]: [RULED OUT/LIKELY] because...
- (evaluate each)

STEP 4 - FINAL DIAGNOSIS:
Based on Step 3, the most likely diagnosis is: [diagnosis]
Confidence: [high/medium/low]

STEP 5 - GUIDELINE-BASED MANAGEMENT:
According to the provided context, the recommended management includes:
- [management step from guidelines]

STEP 6 - SELECT ANSWER:
Based on the diagnosis and management, the correct option is: [A/B/C/D]
Reasoning: [brief explanation linking diagnosis to selected option]

===== END OF REASONING =====

CONFIDENCE_SCORE: [0.0-1.0]
"""
        
        response = self.llm_model.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=768
        )
        
        # Parse structured response
        symptoms = self._extract_list(response, 'STEP 1', 'STEP 2')
        differentials = self._extract_list(response, 'STEP 2', 'STEP 3')
        ruled_out = self._extract_ruled_out(response)
        
        # Extract final diagnosis
        diag_match = re.search(r'STEP 4.*?most likely diagnosis is:?\s*(.+?)(?:\n|Confidence)', response, re.IGNORECASE | re.DOTALL)
        final_diagnosis = diag_match.group(1).strip() if diag_match else "Unknown"
        
        # Extract management
        mgmt_match = re.search(r'STEP 5.*?includes:?(.*?)(?:STEP 6|$)', response, re.IGNORECASE | re.DOTALL)
        management = mgmt_match.group(1).strip() if mgmt_match else "See guidelines"
        
        # Extract selected option
        option_match = re.search(r'correct option is:?\s*\[?([A-D])\]?', response, re.IGNORECASE)
        selected_option = option_match.group(1).upper() if option_match else "A"
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE_SCORE:\s*([0-9.]+)', response)
        confidence = float(conf_match.group(1)) if conf_match else 0.6
        confidence = min(1.0, max(0.0, confidence))
        
        return DifferentialDiagnosisResult(
            step1_symptoms=symptoms,
            step2_differentials=differentials,
            step3_ruled_out=ruled_out,
            step4_final_diagnosis=final_diagnosis,
            step5_management=management,
            step6_selected_option=selected_option,
            confidence=confidence,
            reasoning_chain=response
        )
    
    def _extract_list(self, text: str, start_marker: str, end_marker: str) -> List[str]:
        """Extract bullet list between markers."""
        section = re.search(f'{start_marker}(.*?){end_marker}', text, re.IGNORECASE | re.DOTALL)
        if not section:
            return []
        items = re.findall(r'-\s*(.+)', section.group(1))
        return [item.strip() for item in items if item.strip()]
    
    def _extract_ruled_out(self, text: str) -> Dict[str, str]:
        """Extract ruled out differentials with reasons."""
        ruled_out = {}
        step3_match = re.search(r'STEP 3(.*?)STEP 4', text, re.IGNORECASE | re.DOTALL)
        if step3_match:
            lines = step3_match.group(1).strip().split('\n')
            for line in lines:
                if ':' in line and ('RULED OUT' in line.upper() or 'LIKELY' in line.upper()):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        diff = parts[0].strip().lstrip('-').strip()
                        reason = parts[1].strip()
                        ruled_out[diff] = reason
        return ruled_out
    
    def _rule_based_reasoning(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        context: List[str]
    ) -> DifferentialDiagnosisResult:
        """Fallback rule-based reasoning."""
        # Extract basic symptoms
        full_text = f"{case_description} {question}".lower()
        symptoms = []
        symptom_keywords = ['pain', 'fever', 'cough', 'dyspnea', 'nausea', 'vomiting', 
                          'bleeding', 'weakness', 'fatigue', 'headache']
        for keyword in symptom_keywords:
            if keyword in full_text:
                symptoms.append(keyword)
        
        # Simple option selection based on context matching
        selected_option = "A"
        best_score = 0
        for label, option_text in options.items():
            score = 0
            option_lower = option_text.lower()
            for ctx in context[:3]:
                if option_lower in ctx.lower():
                    score += 2
                else:
                    words = option_lower.split()
                    score += sum(1 for w in words if len(w) > 4 and w in ctx.lower())
            if score > best_score:
                best_score = score
                selected_option = label
        
        return DifferentialDiagnosisResult(
            step1_symptoms=symptoms,
            step2_differentials=["Based on symptoms"],
            step3_ruled_out={},
            step4_final_diagnosis="Per guidelines",
            step5_management="As per retrieved context",
            step6_selected_option=selected_option,
            confidence=0.5,
            reasoning_chain="Rule-based fallback reasoning"
        )


class GuidelinePrioritizationReranker:
    """
    Fix 3: Rule-based guideline prioritization reranker.
    
    Boost scores for documents containing:
    - "management", "diagnosis", "treatment", "first-line", "indications" (+0.2)
    - Exact disease term match (+0.3)
    """
    
    def __init__(self):
        """Initialize reranker with boost keywords."""
        self.guideline_boost_keywords = [
            'management', 'diagnosis', 'treatment', 'first-line', 
            'indications', 'recommended', 'protocol', 'guideline',
            'therapy', 'initial treatment', 'primary treatment'
        ]
        self.strong_boost_keywords = [
            'first-line treatment', 'recommended treatment', 
            'treatment of choice', 'initial management',
            'standard of care', 'preferred treatment'
        ]
    
    def rerank(
        self,
        query: str,
        case_description: str,
        results: List[Tuple[any, float]],
        disease_terms: Optional[List[str]] = None
    ) -> List[Tuple[any, float]]:
        """
        Rerank results with guideline prioritization.
        
        Args:
            query: Search query
            case_description: Patient case description
            results: List of (document, score) tuples
            disease_terms: Optional list of disease terms to boost
            
        Returns:
            Reranked list of (document, adjusted_score) tuples
        """
        # Extract disease terms if not provided
        if disease_terms is None:
            disease_terms = self._extract_disease_terms(f"{query} {case_description}")
        
        reranked = []
        for doc, score in results:
            adjusted_score = score
            
            # Get document content
            if hasattr(doc, 'content'):
                content = doc.content.lower()
            elif hasattr(doc, 'document'):
                content = doc.document.content.lower()
            else:
                content = str(doc).lower()
            
            # Boost for guideline keywords
            keyword_boost = 0.0
            for keyword in self.guideline_boost_keywords:
                if keyword in content:
                    keyword_boost += 0.05  # 5% per keyword
            keyword_boost = min(0.20, keyword_boost)  # Cap at 20%
            
            # Extra boost for strong keywords
            for keyword in self.strong_boost_keywords:
                if keyword in content:
                    keyword_boost += 0.10  # 10% extra for strong matches
                    break  # Only count once
            
            # Boost for disease term match
            disease_boost = 0.0
            for term in disease_terms:
                if term.lower() in content:
                    disease_boost += 0.15  # 15% per disease match
            disease_boost = min(0.30, disease_boost)  # Cap at 30%
            
            # Apply boosts
            adjusted_score = min(1.0, score + keyword_boost + disease_boost)
            reranked.append((doc, adjusted_score))
        
        # Sort by adjusted score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def _extract_disease_terms(self, text: str) -> List[str]:
        """Extract disease terms from text."""
        disease_patterns = [
            r'\b(pneumonia|sepsis|meningitis|diabetes|hypertension|'
            r'myocardial infarction|heart failure|stroke|pulmonary embolism|'
            r'preeclampsia|eclampsia|anemia|infection|shock|arrhythmia|'
            r'tuberculosis|malaria|cholera|typhoid|hepatitis|hiv|aids|'
            r'bronchitis|asthma|copd|hypoglycemia|hyperglycemia)\b'
        ]
        
        terms = []
        for pattern in disease_patterns:
            matches = re.findall(pattern, text.lower(), re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))


class EnhancedReasoningPipeline:
    """
    Combines all reasoning improvements into a unified pipeline.
    """
    
    def __init__(self, llm_model=None):
        """Initialize all improvement components."""
        self.llm_model = llm_model
        
        # Initialize all components
        self.uncertainty_analyzer = ForcedUncertaintyAnalyzer(llm_model)
        self.minimal_context_evaluator = MinimalContextEvaluator(llm_model)
        self.differential_reasoner = DifferentialDiagnosisReasoner(llm_model)
        self.guideline_reranker = GuidelinePrioritizationReranker()
    
    def enhanced_answer(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        initial_answer: str,
        initial_confidence: float,
        retrieved_contexts: List[str],
        use_differential: bool = True,
        use_uncertainty: bool = True,
        use_minimal_context: bool = True
    ) -> Tuple[str, float, Dict]:
        """
        Apply all reasoning improvements to get final answer.
        
        Args:
            question: The clinical question
            case_description: Patient case description
            options: Answer options
            initial_answer: Initially selected answer
            initial_confidence: Initial confidence score
            retrieved_contexts: Retrieved context documents
            use_differential: Whether to use differential diagnosis
            use_uncertainty: Whether to use uncertainty analysis
            use_minimal_context: Whether to use minimal context re-evaluation
            
        Returns:
            Tuple of (final_answer, final_confidence, metadata)
        """
        metadata = {
            'initial_answer': initial_answer,
            'initial_confidence': initial_confidence
        }
        
        current_answer = initial_answer
        current_confidence = initial_confidence
        
        # Step 1: Differential Diagnosis Reasoning (if enabled)
        if use_differential:
            try:
                diff_result = self.differential_reasoner.reason(
                    question, case_description, options, retrieved_contexts
                )
                # Use differential answer if it has good confidence
                if diff_result.confidence > 0.5:
                    current_answer = diff_result.step6_selected_option
                    current_confidence = diff_result.confidence
                    metadata['differential_diagnosis'] = {
                        'symptoms': diff_result.step1_symptoms,
                        'final_diagnosis': diff_result.step4_final_diagnosis,
                        'selected': diff_result.step6_selected_option
                    }
            except Exception as e:
                print(f"[WARN] Differential reasoning failed: {e}")
        
        # Step 2: Uncertainty Analysis (Fix 4)
        if use_uncertainty:
            try:
                uncertainty_result = self.uncertainty_analyzer.analyze_uncertainty(
                    question, case_description, options, current_answer,
                    current_confidence, retrieved_contexts
                )
                current_answer = uncertainty_result.verified_answer
                current_confidence = uncertainty_result.adjusted_confidence
                metadata['uncertainty_analysis'] = {
                    'potential_errors': uncertainty_result.potential_errors,
                    'used_context': uncertainty_result.used_context,
                    'no_hallucination': uncertainty_result.no_hallucination
                }
            except Exception as e:
                print(f"[WARN] Uncertainty analysis failed: {e}")
        
        # Step 3: Minimal Context Re-evaluation (Fix 5)
        if use_minimal_context:
            try:
                minimal_result = self.minimal_context_evaluator.evaluate(
                    question, case_description, options, current_answer,
                    current_confidence, retrieved_contexts
                )
                current_answer = minimal_result.final_answer
                current_confidence = minimal_result.final_confidence
                metadata['minimal_context'] = {
                    'answers_match': minimal_result.answers_match,
                    'minimal_context_answer': minimal_result.minimal_context_answer
                }
            except Exception as e:
                print(f"[WARN] Minimal context evaluation failed: {e}")
        
        metadata['final_answer'] = current_answer
        metadata['final_confidence'] = current_confidence
        
        return current_answer, current_confidence, metadata

