"""
Clinical Intent Classifier for Medical Queries

This module classifies medical queries by clinical intent to improve
query understanding and evidence matching.

Key Features:
- Intent classification (diagnosis, treatment, management, prevention)
- Temporal understanding (initial, definitive, long-term)
- Urgency detection (emergency, urgent, routine)
- Patient context integration
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class ClinicalIntent:
    """Clinical intent classification result."""
    primary_intent: str  # 'diagnosis', 'treatment', 'management', 'prevention', 'test'
    urgency: str  # 'emergency', 'urgent', 'routine'
    temporal_context: str  # 'initial', 'definitive', 'long-term', 'maintenance', 'none'
    patient_context: Dict[str, Optional[str]]  # age, gender, pregnancy, etc.


class ClinicalIntentClassifier:
    """
    Classify medical queries by clinical intent.
    
    Day 7 Phase 2: Enhanced query understanding with clinical context.
    """
    
    def __init__(self):
        """Initialize clinical intent classifier."""
        self._init_intent_patterns()
        self._init_temporal_patterns()
        self._init_urgency_patterns()
    
    def _init_intent_patterns(self):
        """Initialize patterns for intent classification."""
        self.intent_patterns = {
            'diagnosis': [
                r'\b(diagnos|identif|confirm|rule out|differential|what is|what are)\b',
                r'\b(cause|etiology|pathology)\b',
            ],
            'treatment': [
                r'\b(treat|therapy|medication|drug|prescribe|administer|dose|dosage)\b',
                r'\b(antibiotic|antimicrobial|intervention)\b',
            ],
            'management': [
                r'\b(manag|care|approach|protocol|guideline|recommend)\b',
                r'\b(should|consider|appropriate|plan)\b',
            ],
            'prevention': [
                r'\b(prevent|prophylaxis|prophylactic|avoid|reduce risk)\b',
            ],
            'test': [
                r'\b(test|laboratory|lab|diagnostic|imaging|procedure|examination)\b',
                r'\b(what.*test|which.*test|order|indicated)\b',
            ],
        }
    
    def _init_temporal_patterns(self):
        """Initialize patterns for temporal understanding."""
        self.temporal_patterns = {
            'initial': [
                r'\b(initial|first|first-line|primary|immediate|stat|acute|emergency)\b',
                r'\b(start|begin|commence)\b',
            ],
            'definitive': [
                r'\b(definitive|final|complete|full|standard)\b',
                r'\b(long-term|chronic|maintenance)\b',
            ],
            'long-term': [
                r'\b(long-term|chronic|maintenance|ongoing|continued)\b',
            ],
            'maintenance': [
                r'\b(maintenance|ongoing|continued|follow-up)\b',
            ],
        }
    
    def _init_urgency_patterns(self):
        """Initialize patterns for urgency detection."""
        self.urgency_patterns = {
            'emergency': [
                r'\b(emergency|emergent|immediate|stat|urgent|acute|critical|life-threatening)\b',
                r'\b(severe|shock|arrest|unstable)\b',
            ],
            'urgent': [
                r'\b(urgent|soon|prompt|quick|rapid)\b',
            ],
            'routine': [
                r'\b(routine|elective|scheduled|planned)\b',
            ],
        }
    
    def classify(self, question: str, case_description: str = "") -> ClinicalIntent:
        """
        Classify clinical intent from question and case description.
        
        Args:
            question: The medical question
            case_description: Patient case description (optional)
            
        Returns:
            ClinicalIntent object with classification results
        """
        full_text = f"{case_description} {question}".lower()
        
        # Classify primary intent
        primary_intent = self._classify_intent(full_text)
        
        # Classify urgency
        urgency = self._classify_urgency(full_text)
        
        # Classify temporal context
        temporal_context = self._classify_temporal(full_text)
        
        # Extract patient context
        patient_context = self._extract_patient_context(full_text)
        
        return ClinicalIntent(
            primary_intent=primary_intent,
            urgency=urgency,
            temporal_context=temporal_context,
            patient_context=patient_context
        )
    
    def _classify_intent(self, text: str) -> str:
        """Classify primary clinical intent."""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            intent_scores[intent] = score
        
        # Return intent with highest score, default to 'treatment'
        if max(intent_scores.values()) > 0:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        return 'treatment'  # Default
    
    def _classify_urgency(self, text: str) -> str:
        """Classify urgency level."""
        urgency_scores = {}
        
        for urgency, patterns in self.urgency_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            urgency_scores[urgency] = score
        
        # Return urgency with highest score, default to 'routine'
        if max(urgency_scores.values()) > 0:
            return max(urgency_scores.items(), key=lambda x: x[1])[0]
        return 'routine'  # Default
    
    def _classify_temporal(self, text: str) -> str:
        """Classify temporal context."""
        temporal_scores = {}
        
        for temporal, patterns in self.temporal_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            temporal_scores[temporal] = score
        
        # Return temporal with highest score, default to 'none'
        if max(temporal_scores.values()) > 0:
            return max(temporal_scores.items(), key=lambda x: x[1])[0]
        return 'none'  # Default
    
    def _extract_patient_context(self, text: str) -> Dict[str, Optional[str]]:
        """Extract patient context (age, gender, pregnancy status)."""
        context = {
            'age': None,
            'age_group': None,
            'gender': None,
            'pregnancy': None,
        }
        
        # Extract age
        age_patterns = [
            r'(\d+)\s*(?:year|yr|month|mo|day|week|wk)\s*old',
            r'age\s*(?:of|:)?\s*(\d+)',
            r'(\d+)\s*(?:year|yr)\s*(?:old|of age)',
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                context['age'] = str(age)
                # Classify age group
                if age < 18:
                    context['age_group'] = 'pediatric'
                elif age >= 65:
                    context['age_group'] = 'geriatric'
                else:
                    context['age_group'] = 'adult'
                break
        
        # Extract gender
        if re.search(r'\b(female|woman|girl|pregnant|pregnancy)\b', text, re.IGNORECASE):
            context['gender'] = 'female'
        elif re.search(r'\b(male|man|boy)\b', text, re.IGNORECASE):
            context['gender'] = 'male'
        
        # Extract pregnancy status
        if re.search(r'\b(pregnant|pregnancy|gestation|gestational)\b', text, re.IGNORECASE):
            context['pregnancy'] = 'yes'
            # Extract gestational age if mentioned
            gest_match = re.search(r'(\d+)\s*(?:week|wk|month)\s*(?:gestation|pregnant)', text, re.IGNORECASE)
            if gest_match:
                context['gestational_age'] = gest_match.group(1)
        
        return context
    
    def get_evidence_preference(self, intent: ClinicalIntent) -> Dict[str, float]:
        """
        Get evidence preference weights based on clinical intent.
        
        Returns:
            Dict with preference weights for different evidence types
        """
        preferences = {
            'treatment_section': 1.0,
            'dose_specificity': 1.0,
            'temporal_match': 1.0,
        }
        
        # Adjust based on intent
        if intent.primary_intent == 'treatment':
            preferences['treatment_section'] = 1.5  # 50% boost for treatment section
            preferences['dose_specificity'] = 1.3  # 30% boost for specific doses
        
        # Adjust based on temporal context
        if intent.temporal_context == 'initial':
            preferences['temporal_match'] = 1.4  # 40% boost for initial treatment matches
        elif intent.temporal_context == 'definitive':
            preferences['temporal_match'] = 1.2  # 20% boost for definitive treatment
        
        # Adjust based on urgency
        if intent.urgency == 'emergency':
            preferences['treatment_section'] *= 1.2  # Additional boost for emergency
        
        return preferences

