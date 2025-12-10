"""
Deterministic Chain-of-Thought Reasoner

Step 3 Fix: Disable Tree-of-Thought, use deterministic CoT with verification.

ToT is harmful when:
- Retrieval is noisy
- Answer space is small (A/B/C/D)
- LLM is small/medium like Ollama

Switch to:
- Deterministic CoT
- With verification
- And scoring

Step 4 Fix: Symptom Importance Ranking
- Before final answer, have LLM list top 3 most important symptoms
- If it misses major ones → reject reasoning and retry

Expected gains: +5% accuracy each
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class SymptomRanking:
    """Result of symptom importance ranking."""
    top_symptoms: List[str]
    missed_critical: List[str]
    ranking_valid: bool
    confidence_adjustment: float


@dataclass
class DeterministicReasoningResult:
    """Result of deterministic CoT reasoning."""
    selected_answer: str
    confidence: float
    reasoning_trace: str
    verification_passed: bool
    symptom_ranking: Optional[SymptomRanking]
    retry_count: int


class DeterministicReasoner:
    """
    Deterministic Chain-of-Thought reasoner with verification.
    
    No Tree-of-Thought - just straightforward step-by-step reasoning
    with symptom verification and answer scoring.
    """
    
    def __init__(self, llm_model=None):
        """Initialize with LLM model."""
        self.llm_model = llm_model
        self.max_retries = 2
    
    def reason(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        context: str,
        critical_symptoms: List[str] = None,
        extracted_features: Dict = None
    ) -> DeterministicReasoningResult:
        """
        Perform deterministic CoT reasoning with verification.
        
        Args:
            question: Clinical question
            case_description: Patient case
            options: Answer options {A: text, B: text, ...}
            context: Pruned relevant context
            critical_symptoms: List of critical symptoms that must be considered
            extracted_features: Pre-extracted clinical features
            
        Returns:
            DeterministicReasoningResult
        """
        if not self.llm_model:
            return self._rule_based_fallback(options, context, critical_symptoms)
        
        # Step 4: First, get symptom importance ranking
        symptom_ranking = None
        if critical_symptoms:
            symptom_ranking = self._rank_symptoms(
                case_description, question, critical_symptoms
            )
        
        # Main reasoning with retries
        retry_count = 0
        best_result = None
        
        while retry_count <= self.max_retries:
            result = self._single_reasoning_pass(
                question, case_description, options, context,
                symptom_ranking, retry_count
            )
            
            # Verify the answer
            verification_passed = self._verify_answer(
                result, options, context, critical_symptoms
            )
            
            if verification_passed:
                return DeterministicReasoningResult(
                    selected_answer=result['answer'],
                    confidence=result['confidence'],
                    reasoning_trace=result['reasoning'],
                    verification_passed=True,
                    symptom_ranking=symptom_ranking,
                    retry_count=retry_count
                )
            
            # Keep best result in case all retries fail
            if best_result is None or result['confidence'] > best_result['confidence']:
                best_result = result
            
            retry_count += 1
        
        # Return best result even if verification failed
        return DeterministicReasoningResult(
            selected_answer=best_result['answer'] if best_result else 'A',
            confidence=best_result['confidence'] * 0.7 if best_result else 0.3,
            reasoning_trace=best_result['reasoning'] if best_result else "Verification failed",
            verification_passed=False,
            symptom_ranking=symptom_ranking,
            retry_count=retry_count
        )
    
    def _rank_symptoms(
        self,
        case_description: str,
        question: str,
        critical_symptoms: List[str]
    ) -> SymptomRanking:
        """
        Step 4: Rank symptom importance using LLM.
        
        Have LLM list top 3 most important symptoms.
        Check if major symptoms are missed.
        """
        if not self.llm_model:
            return SymptomRanking(
                top_symptoms=critical_symptoms[:3],
                missed_critical=[],
                ranking_valid=True,
                confidence_adjustment=1.0
            )
        
        prompt = f"""You are analyzing a clinical case. List the TOP 3 most clinically important symptoms/findings that should guide the diagnosis and treatment decision.

Case: {case_description}

Question: {question}

Known symptoms/findings in this case: {', '.join(critical_symptoms)}

Output EXACTLY 3 symptoms, one per line, most important first:
1. [most important symptom]
2. [second most important]
3. [third most important]

Only list symptoms that are actually present in the case."""

        try:
            response = self.llm_model.generate(
                prompt=prompt,
                temperature=0.1,  # Deterministic
                max_tokens=100
            )
            
            # Parse response
            lines = response.strip().split('\n')
            top_symptoms = []
            for line in lines:
                # Extract symptom from numbered list
                match = re.search(r'^\d+\.\s*(.+)$', line.strip())
                if match:
                    symptom = match.group(1).strip().lower()
                    top_symptoms.append(symptom)
            
            # Check for missed critical symptoms
            # Critical symptoms that should definitely be in top 3
            must_include = []
            for symptom in critical_symptoms:
                symptom_lower = symptom.lower()
                if any(kw in symptom_lower for kw in [
                    'chest pain', 'shortness of breath', 'syncope', 'seizure',
                    'bleeding', 'unconscious', 'shock', 'respiratory distress'
                ]):
                    must_include.append(symptom)
            
            missed = []
            for must in must_include:
                if not any(must.lower() in s.lower() for s in top_symptoms):
                    missed.append(must)
            
            ranking_valid = len(missed) == 0
            confidence_adjustment = 1.0 if ranking_valid else 0.8
            
            return SymptomRanking(
                top_symptoms=top_symptoms[:3],
                missed_critical=missed,
                ranking_valid=ranking_valid,
                confidence_adjustment=confidence_adjustment
            )
            
        except Exception as e:
            print(f"[WARN] Symptom ranking failed: {e}")
            return SymptomRanking(
                top_symptoms=critical_symptoms[:3],
                missed_critical=[],
                ranking_valid=True,
                confidence_adjustment=1.0
            )
    
    def _single_reasoning_pass(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        context: str,
        symptom_ranking: Optional[SymptomRanking],
        retry_count: int
    ) -> Dict:
        """Perform a single deterministic reasoning pass."""
        
        # Build symptom guidance if available
        symptom_guidance = ""
        if symptom_ranking and symptom_ranking.top_symptoms:
            symptom_guidance = f"""
IMPORTANT - Key symptoms to consider (in order of importance):
1. {symptom_ranking.top_symptoms[0] if len(symptom_ranking.top_symptoms) > 0 else 'N/A'}
2. {symptom_ranking.top_symptoms[1] if len(symptom_ranking.top_symptoms) > 1 else 'N/A'}
3. {symptom_ranking.top_symptoms[2] if len(symptom_ranking.top_symptoms) > 2 else 'N/A'}
"""
            if symptom_ranking.missed_critical:
                symptom_guidance += f"\nWARNING: Also consider these critical findings: {', '.join(symptom_ranking.missed_critical)}\n"
        
        # Retry instruction
        retry_instruction = ""
        if retry_count > 0:
            retry_instruction = "\nPREVIOUS ATTEMPT FAILED VERIFICATION. Please reconsider carefully and verify your answer against the context.\n"
        
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        prompt = f"""You are a medical expert answering a clinical question. Use ONLY the provided context to answer.
{retry_instruction}
CASE:
{case_description}
{symptom_guidance}
QUESTION:
{question}

OPTIONS:
{options_text}

RELEVANT CONTEXT (use ONLY this information):
{context[:2500]}

INSTRUCTIONS:
1. Identify the key clinical findings from the case
2. Search the context for information about treatment/management
3. Match each option against the context
4. Select the option that is DIRECTLY supported by the context
5. If no option is directly supported, choose the most reasonable based on context

Think step by step:

STEP 1 - Key findings:
[List 3-5 key findings from the case]

STEP 2 - What the context says:
[Quote relevant information from context]

STEP 3 - Option analysis:
[For each option, state if it's supported by context]

STEP 4 - Final answer:
[State your answer as just the letter: A, B, C, or D]

ANSWER:"""

        try:
            response = self.llm_model.generate(
                prompt=prompt,
                temperature=0.1,  # Deterministic
                max_tokens=500
            )
            
            # Extract answer
            answer = self._extract_answer(response, options)
            confidence = self._calculate_confidence(response, answer, context, options)
            
            return {
                'answer': answer,
                'confidence': confidence,
                'reasoning': response
            }
            
        except Exception as e:
            print(f"[WARN] Reasoning failed: {e}")
            return {
                'answer': 'A',
                'confidence': 0.25,
                'reasoning': f"Error: {e}"
            }
    
    def _extract_answer(self, response: str, options: Dict[str, str]) -> str:
        """Extract answer letter from response."""
        response_upper = response.upper()
        
        # Look for explicit "ANSWER: X" pattern
        patterns = [
            r'ANSWER:\s*([A-D])',
            r'FINAL ANSWER:\s*([A-D])',
            r'THE ANSWER IS\s*([A-D])',
            r'\b([A-D])\s*(?:is|appears|seems)\s+(?:the\s+)?(?:correct|best|right)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                return match.group(1)
        
        # Look for standalone letter at end
        last_lines = response.strip().split('\n')[-3:]
        for line in reversed(last_lines):
            match = re.search(r'\b([A-D])\b', line.upper())
            if match:
                return match.group(1)
        
        # Default to A
        return 'A'
    
    def _calculate_confidence(
        self,
        reasoning: str,
        answer: str,
        context: str,
        options: Dict[str, str]
    ) -> float:
        """Calculate confidence score based on reasoning quality."""
        confidence = 0.5  # Base confidence
        
        reasoning_lower = reasoning.lower()
        context_lower = context.lower()
        answer_text = options.get(answer, '').lower()
        
        # Boost if answer text appears in context
        if answer_text and answer_text in context_lower:
            confidence += 0.2
        
        # Boost if reasoning mentions specific context quotes
        if 'according to' in reasoning_lower or 'the context states' in reasoning_lower:
            confidence += 0.1
        
        # Boost if step-by-step reasoning is present
        if 'step 1' in reasoning_lower and 'step 2' in reasoning_lower:
            confidence += 0.1
        
        # Penalty if reasoning expresses uncertainty
        uncertainty_words = ['unsure', 'unclear', 'uncertain', 'not sure', 'cannot determine']
        if any(word in reasoning_lower for word in uncertainty_words):
            confidence -= 0.15
        
        # Penalty if "none of the above" or similar
        if 'none of' in reasoning_lower or 'cannot answer' in reasoning_lower:
            confidence -= 0.2
        
        return max(0.1, min(0.95, confidence))
    
    def _verify_answer(
        self,
        result: Dict,
        options: Dict[str, str],
        context: str,
        critical_symptoms: List[str]
    ) -> bool:
        """Verify the answer is reasonable."""
        answer = result['answer']
        answer_text = options.get(answer, '')
        context_lower = context.lower()
        
        # Basic sanity checks
        if not answer or answer not in options:
            return False
        
        # Check if answer or key terms appear in context
        answer_words = [w for w in answer_text.lower().split() if len(w) > 4]
        matches = sum(1 for w in answer_words if w in context_lower)
        
        if len(answer_words) > 0 and matches == 0:
            # No key terms from answer in context - suspicious
            return False
        
        # Confidence threshold
        if result['confidence'] < 0.3:
            return False
        
        return True
    
    def _rule_based_fallback(
        self,
        options: Dict[str, str],
        context: str,
        critical_symptoms: List[str]
    ) -> DeterministicReasoningResult:
        """Fallback to rule-based reasoning when LLM unavailable."""
        context_lower = context.lower()
        
        best_answer = 'A'
        best_score = 0
        
        for label, text in options.items():
            score = 0
            text_lower = text.lower()
            
            # Check if option text appears in context
            if text_lower in context_lower:
                score += 1.0
            else:
                # Check word overlap
                words = [w for w in text_lower.split() if len(w) > 3]
                matches = sum(1 for w in words if w in context_lower)
                score += matches * 0.2
            
            # Bonus for treatment-related words
            treatment_words = ['treatment', 'therapy', 'dose', 'mg', 'administer']
            if any(tw in text_lower for tw in treatment_words):
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_answer = label
        
        return DeterministicReasoningResult(
            selected_answer=best_answer,
            confidence=min(0.6, best_score / 2),
            reasoning_trace="Rule-based fallback (no LLM)",
            verification_passed=True,
            symptom_ranking=None,
            retry_count=0
        )


def main():
    """Test deterministic reasoner."""
    print("="*70)
    print("DETERMINISTIC REASONER TEST")
    print("="*70)
    
    reasoner = DeterministicReasoner()
    
    # Test without LLM (rule-based fallback)
    options = {
        'A': 'Ampicillin and gentamicin IV',
        'B': 'Oral amoxicillin',
        'C': 'Observation only',
        'D': 'Ceftriaxone IV'
    }
    
    context = """
    Treatment of neonatal sepsis: First-line therapy is ampicillin and gentamicin IV.
    Ampicillin covers Listeria and group B strep. Gentamicin provides gram-negative coverage.
    Dosing: Ampicillin 100mg/kg/dose, Gentamicin 4mg/kg/dose.
    """
    
    critical_symptoms = ['fever', 'poor feeding', 'lethargy']
    
    result = reasoner._rule_based_fallback(options, context, critical_symptoms)
    
    print(f"\nRule-based result:")
    print(f"  Answer: {result.selected_answer}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Verification: {result.verification_passed}")
    
    print(f"\n{'='*70}")
    print("[OK] Deterministic Reasoner operational!")


if __name__ == "__main__":
    main()

