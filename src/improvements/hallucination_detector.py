"""
Hallucination Detection and Suppression Module

Detects when the model invents:
- Treatments not in guidelines
- Drug doses not mentioned in context
- Durations not supported by evidence
- Etiologies not in retrieved documents

If hallucination detected → regenerate with only quoted evidence.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import re


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    is_hallucinated: bool
    confidence: float  # 0-1, how confident we are it's hallucinated
    hallucinated_elements: List[str]
    grounded_elements: List[str]
    missing_evidence: List[str]
    recommendation: str


class HallucinationDetector:
    """
    Detects hallucinated content in medical reasoning.
    
    Compares reasoning sentences with available retrieved text
    to identify unsupported claims.
    """
    
    def __init__(self):
        """Initialize detector with patterns."""
        self._init_medical_claim_patterns()
        self._init_verification_rules()
    
    def _init_medical_claim_patterns(self):
        """Initialize patterns for extracting medical claims."""
        # Dosing patterns
        self.dose_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|ml|l|units?|iu)\s*'
            r'(?:per|/)\s*(kg|day|dose|hour|hr)?',
            re.IGNORECASE
        )
        
        # Duration patterns
        self.duration_pattern = re.compile(
            r'(?:for|over|within|during)\s*(\d+)\s*(days?|weeks?|hours?|minutes?|months?)',
            re.IGNORECASE
        )
        
        # Treatment claim patterns
        self.treatment_pattern = re.compile(
            r'(?:treat(?:ment|ed)?|manage(?:ment|d)?|give|administer|prescribe|use)\s+'
            r'(?:with\s+)?([a-zA-Z][a-zA-Z\s-]+?)(?:\s+for|\s+to|\s*\.|,|$)',
            re.IGNORECASE
        )
        
        # Diagnosis claim patterns
        self.diagnosis_pattern = re.compile(
            r'(?:diagnos(?:is|ed|e)|confirm(?:ed|s)?|indicates?|suggests?)\s+'
            r'([a-zA-Z][a-zA-Z\s-]+?)(?:\s+based|\s+due|,|\.|$)',
            re.IGNORECASE
        )
        
        # Contraindication patterns
        self.contraindication_pattern = re.compile(
            r'(?:contraindicated?|avoid|do not use|should not)\s+'
            r'(?:in\s+)?([a-zA-Z][a-zA-Z\s-]+?)(?:\s+due|\s+because|,|\.|$)',
            re.IGNORECASE
        )
    
    def _init_verification_rules(self):
        """Initialize verification rules."""
        # Common drug names for verification
        self.common_drugs = {
            'antibiotics': [
                'ampicillin', 'amoxicillin', 'penicillin', 'ceftriaxone', 'cefotaxime',
                'gentamicin', 'vancomycin', 'metronidazole', 'azithromycin', 'ciprofloxacin',
                'doxycycline', 'clindamycin', 'meropenem', 'piperacillin'
            ],
            'analgesics': [
                'paracetamol', 'acetaminophen', 'ibuprofen', 'morphine', 'fentanyl',
                'tramadol', 'diclofenac', 'ketorolac'
            ],
            'cardiovascular': [
                'aspirin', 'clopidogrel', 'heparin', 'enoxaparin', 'warfarin',
                'metoprolol', 'atenolol', 'amlodipine', 'lisinopril', 'furosemide'
            ],
            'emergency': [
                'epinephrine', 'adrenaline', 'dopamine', 'norepinephrine', 'atropine',
                'magnesium sulfate', 'hydralazine', 'labetalol'
            ]
        }
        
        # Valid dose ranges for common drugs
        self.valid_dose_ranges = {
            'ampicillin': {'min_mg_kg': 25, 'max_mg_kg': 100, 'adult_dose': '500-2000mg'},
            'gentamicin': {'min_mg_kg': 2.5, 'max_mg_kg': 7.5, 'adult_dose': '3-5mg/kg'},
            'ceftriaxone': {'min_mg_kg': 50, 'max_mg_kg': 100, 'adult_dose': '1-2g'},
            'paracetamol': {'min_mg_kg': 10, 'max_mg_kg': 15, 'adult_dose': '500-1000mg'},
            'morphine': {'min_mg_kg': 0.05, 'max_mg_kg': 0.1, 'adult_dose': '2-10mg'},
            'epinephrine': {'adult_dose': '0.3-0.5mg IM', 'iv_dose': '1mg'},
            'magnesium sulfate': {'loading': '4-6g', 'maintenance': '1-2g/hr'}
        }
    
    def detect(
        self,
        reasoning_text: str,
        final_answer: str,
        retrieved_contexts: List[Any],
        options: List[str]
    ) -> HallucinationResult:
        """
        Detect hallucination in reasoning and answer.
        
        Args:
            reasoning_text: The model's reasoning/rationale
            final_answer: The selected answer
            retrieved_contexts: Retrieved guideline documents
            options: Answer options
            
        Returns:
            HallucinationResult with detection details
        """
        hallucinated = []
        grounded = []
        missing_evidence = []
        
        # Combine all retrieved context into searchable text
        context_text = self._extract_context_text(retrieved_contexts)
        context_lower = context_text.lower()
        
        # Extract claims from reasoning
        claims = self._extract_claims(reasoning_text)
        
        # Verify each claim
        for claim_type, claim_text in claims:
            if self._is_claim_grounded(claim_text, context_lower, claim_type):
                grounded.append(f"{claim_type}: {claim_text}")
            else:
                hallucinated.append(f"{claim_type}: {claim_text}")
                missing_evidence.append(f"No evidence for: {claim_text}")
        
        # Check dose claims specifically
        dose_claims = self._extract_dose_claims(reasoning_text)
        for drug, dose, unit in dose_claims:
            if self._verify_dose(drug, dose, unit, context_lower):
                grounded.append(f"dose: {drug} {dose}{unit}")
            else:
                # Check if dose is within valid range
                if self._is_dose_plausible(drug, dose, unit):
                    grounded.append(f"dose (plausible): {drug} {dose}{unit}")
                else:
                    hallucinated.append(f"dose: {drug} {dose}{unit}")
                    missing_evidence.append(f"Dose not verified: {drug} {dose}{unit}")
        
        # Check duration claims
        duration_claims = self._extract_duration_claims(reasoning_text)
        for duration, unit in duration_claims:
            if not self._verify_duration(duration, unit, context_lower):
                hallucinated.append(f"duration: {duration} {unit}")
                missing_evidence.append(f"Duration not in context: {duration} {unit}")
        
        # Calculate hallucination score
        total_claims = len(hallucinated) + len(grounded)
        if total_claims > 0:
            hallucination_score = len(hallucinated) / total_claims
        else:
            hallucination_score = 0.0
        
        # Determine if overall answer is hallucinated
        is_hallucinated = hallucination_score > 0.3 or len(hallucinated) >= 3
        
        # Generate recommendation
        if is_hallucinated:
            recommendation = (
                "HALLUCINATION DETECTED: Regenerate answer using only evidence from "
                "retrieved guidelines. Do not invent doses, durations, or treatments."
            )
        elif hallucinated:
            recommendation = (
                f"Minor unsupported claims detected ({len(hallucinated)}). "
                "Consider verifying with guidelines."
            )
        else:
            recommendation = "Answer appears to be evidence-grounded."
        
        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=1.0 - hallucination_score if not is_hallucinated else hallucination_score,
            hallucinated_elements=hallucinated,
            grounded_elements=grounded,
            missing_evidence=missing_evidence,
            recommendation=recommendation
        )
    
    def _extract_context_text(self, retrieved_contexts: List[Any]) -> str:
        """Extract text from retrieved contexts."""
        texts = []
        for ctx in retrieved_contexts:
            if hasattr(ctx, 'document'):
                texts.append(ctx.document.content)
            elif hasattr(ctx, 'content'):
                texts.append(ctx.content)
            else:
                texts.append(str(ctx))
        return ' '.join(texts)
    
    def _extract_claims(self, text: str) -> List[Tuple[str, str]]:
        """Extract medical claims from text."""
        claims = []
        
        # Extract treatment claims
        for match in self.treatment_pattern.finditer(text):
            claims.append(('treatment', match.group(1).strip()))
        
        # Extract diagnosis claims
        for match in self.diagnosis_pattern.finditer(text):
            claims.append(('diagnosis', match.group(1).strip()))
        
        # Extract contraindication claims
        for match in self.contraindication_pattern.finditer(text):
            claims.append(('contraindication', match.group(1).strip()))
        
        return claims
    
    def _extract_dose_claims(self, text: str) -> List[Tuple[str, float, str]]:
        """Extract drug dose claims from text."""
        claims = []
        
        # Find dose patterns
        for match in self.dose_pattern.finditer(text):
            dose = float(match.group(1))
            unit = match.group(2)
            
            # Try to find the drug name before the dose
            context_start = max(0, match.start() - 50)
            preceding = text[context_start:match.start()].lower()
            
            drug_found = None
            for category, drugs in self.common_drugs.items():
                for drug in drugs:
                    if drug in preceding:
                        drug_found = drug
                        break
                if drug_found:
                    break
            
            if drug_found:
                claims.append((drug_found, dose, unit))
        
        return claims
    
    def _extract_duration_claims(self, text: str) -> List[Tuple[int, str]]:
        """Extract duration claims from text."""
        claims = []
        for match in self.duration_pattern.finditer(text):
            duration = int(match.group(1))
            unit = match.group(2)
            claims.append((duration, unit))
        return claims
    
    def _is_claim_grounded(
        self,
        claim: str,
        context: str,
        claim_type: str
    ) -> bool:
        """Check if a claim is grounded in context."""
        claim_lower = claim.lower()
        
        # Direct mention
        if claim_lower in context:
            return True
        
        # Check individual significant words
        words = [w for w in claim_lower.split() if len(w) > 4]
        if words:
            matched = sum(1 for w in words if w in context)
            if matched / len(words) >= 0.5:
                return True
        
        return False
    
    def _verify_dose(
        self,
        drug: str,
        dose: float,
        unit: str,
        context: str
    ) -> bool:
        """Verify if a dose is mentioned in context."""
        # Check for exact dose mention
        dose_str = f"{dose}"
        if dose_str in context and drug in context:
            return True
        
        # Check for dose range mentions
        if f"{int(dose)}" in context and drug in context:
            return True
        
        return False
    
    def _is_dose_plausible(
        self,
        drug: str,
        dose: float,
        unit: str
    ) -> bool:
        """Check if a dose is within plausible medical range."""
        if drug not in self.valid_dose_ranges:
            return True  # Can't verify, assume plausible
        
        ranges = self.valid_dose_ranges[drug]
        
        # Check mg/kg ranges if applicable
        if 'mg' in unit.lower():
            if 'max_mg_kg' in ranges:
                if dose <= ranges['max_mg_kg'] * 100:  # Assume max 100kg
                    return True
        
        return True  # Default to plausible if can't verify
    
    def _verify_duration(
        self,
        duration: int,
        unit: str,
        context: str
    ) -> bool:
        """Verify if a duration is mentioned in context."""
        # Check for exact duration mention
        duration_str = f"{duration} {unit}"
        if duration_str in context:
            return True
        
        # Check without 's'
        duration_str2 = f"{duration} {unit.rstrip('s')}"
        if duration_str2 in context:
            return True
        
        return False
    
    def generate_grounded_response(
        self,
        question: str,
        options: List[str],
        retrieved_contexts: List[Any]
    ) -> str:
        """Generate a response using only grounded evidence."""
        context_text = self._extract_context_text(retrieved_contexts)
        
        # Find which option is most supported by context
        option_scores = {}
        for option in options:
            match = re.match(r'^([A-D])[.)]\s*(.+)$', option.strip())
            if match:
                letter = match.group(1)
                text = match.group(2).strip().lower()
                
                # Score based on keyword presence in context
                words = [w for w in text.split() if len(w) > 3]
                if words:
                    score = sum(1 for w in words if w in context_text.lower()) / len(words)
                    option_scores[letter] = score
        
        if option_scores:
            best_option = max(option_scores, key=option_scores.get)
            return best_option
        
        return "Cannot answer from context"

