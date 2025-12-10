"""
Enhanced Symptom Extraction Module

This module provides comprehensive symptom extraction from medical case descriptions.
Addresses Day 6 issue: 79 cases missing critical symptoms.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ExtractedSymptom:
    """Extracted symptom with metadata."""
    symptom: str
    severity: str  # 'critical', 'moderate', 'mild'
    negated: bool  # True if symptom is negated (e.g., "no fever")
    context: str  # Surrounding context
    position: int  # Position in text


@dataclass
class SymptomExtractionResult:
    """Result from symptom extraction."""
    symptoms: List[ExtractedSymptom]
    critical_symptoms: List[str]
    all_symptom_terms: List[str]
    missing_critical: List[str]  # Symptoms that should be present but aren't


class EnhancedSymptomExtractor:
    """
    Extract and prioritize medical symptoms from case descriptions.
    
    Features:
    - Comprehensive symptom dictionary
    - Severity classification
    - Negation handling
    - Life-threatening symptom prioritization
    - Context-aware extraction
    - Fix 2: Symptom keyword injection with UMLS synonyms
    """
    
    def __init__(self):
        """Initialize symptom extractor."""
        self._init_symptom_dictionary()
        self._init_critical_symptoms()
        self._init_negation_patterns()
        self._load_umls_synonyms()
    
    def _load_umls_synonyms(self):
        """Load UMLS synonyms for symptom expansion (Fix 2)."""
        import json
        from pathlib import Path
        
        self.umls_synonyms = {}
        umls_path = Path("data/umls_synonyms.json")
        if umls_path.exists():
            try:
                with umls_path.open("r", encoding="utf-8") as f:
                    self.umls_synonyms = json.load(f) or {}
                # Normalize keys to lowercase
                self.umls_synonyms = {k.lower(): [s.lower() for s in v] for k, v in self.umls_synonyms.items()}
            except Exception as e:
                print(f"[WARN] Failed to load UMLS synonyms: {e}")
                self.umls_synonyms = {}
    
    def _init_symptom_dictionary(self):
        """Initialize comprehensive symptom dictionary."""
        self.symptom_dict = {
            # Pain symptoms
            'pain': {
                'variants': ['pain', 'ache', 'aching', 'soreness', 'discomfort'],
                'severity': 'moderate',
                'critical': False
            },
            'chest_pain': {
                'variants': ['chest pain', 'chest discomfort', 'angina', 'precordial pain'],
                'severity': 'critical',
                'critical': True
            },
            'abdominal_pain': {
                'variants': ['abdominal pain', 'stomach pain', 'belly pain', 'abdominal discomfort'],
                'severity': 'moderate',
                'critical': False
            },
            'headache': {
                'variants': ['headache', 'head pain', 'cephalgia'],
                'severity': 'moderate',
                'critical': False
            },
            'back_pain': {
                'variants': ['back pain', 'backache', 'lumbar pain'],
                'severity': 'moderate',
                'critical': False
            },
            
            # Respiratory symptoms
            'dyspnea': {
                'variants': ['shortness of breath', 'dyspnea', 'breathlessness', 'difficulty breathing',
                           'respiratory distress', 'labored breathing', 'trouble breathing'],
                'severity': 'critical',
                'critical': True
            },
            'cough': {
                'variants': ['cough', 'coughing', 'productive cough', 'dry cough'],
                'severity': 'moderate',
                'critical': False
            },
            'wheezing': {
                'variants': ['wheezing', 'wheeze', 'stridor'],
                'severity': 'moderate',
                'critical': False
            },
            
            # Cardiovascular symptoms
            'palpitations': {
                'variants': ['palpitations', 'irregular heartbeat', 'heart racing', 'tachycardia'],
                'severity': 'moderate',
                'critical': False
            },
            'syncope': {
                'variants': ['syncope', 'fainting', 'loss of consciousness', 'passing out', 'unconscious'],
                'severity': 'critical',
                'critical': True
            },
            'dizziness': {
                'variants': ['dizziness', 'dizzy', 'lightheadedness', 'vertigo'],
                'severity': 'moderate',
                'critical': False
            },
            
            # Gastrointestinal symptoms
            'nausea': {
                'variants': ['nausea', 'nauseous', 'feeling sick'],
                'severity': 'moderate',
                'critical': False
            },
            'vomiting': {
                'variants': ['vomiting', 'vomit', 'emesis', 'throwing up'],
                'severity': 'moderate',
                'critical': False
            },
            'diarrhea': {
                'variants': ['diarrhea', 'diarrhoea', 'loose stools'],
                'severity': 'moderate',
                'critical': False
            },
            'jaundice': {
                'variants': ['jaundice', 'yellowing', 'icterus', 'yellow skin'],
                'severity': 'critical',
                'critical': True
            },
            
            # Neurological symptoms
            'seizure': {
                'variants': ['seizure', 'convulsion', 'fit', 'epileptic'],
                'severity': 'critical',
                'critical': True
            },
            'confusion': {
                'variants': ['confusion', 'altered mental status', 'disorientation', 'mental status change'],
                'severity': 'critical',
                'critical': True
            },
            'weakness': {
                'variants': ['weakness', 'weak', 'fatigue', 'tiredness', 'lethargy'],
                'severity': 'moderate',
                'critical': False
            },
            'numbness': {
                'variants': ['numbness', 'numb', 'tingling', 'paresthesia'],
                'severity': 'moderate',
                'critical': False
            },
            
            # Dermatological symptoms
            'rash': {
                'variants': ['rash', 'skin rash', 'eruption', 'dermatitis'],
                'severity': 'moderate',
                'critical': False
            },
            'bleeding': {
                'variants': ['bleeding', 'hemorrhage', 'blood loss', 'hemorrhaging'],
                'severity': 'critical',
                'critical': True
            },
            
            # Systemic symptoms
            'fever': {
                'variants': ['fever', 'pyrexia', 'elevated temperature', 'high temperature', 'hyperthermia'],
                'severity': 'moderate',
                'critical': False
            },
            'chills': {
                'variants': ['chills', 'shivering', 'rigors'],
                'severity': 'moderate',
                'critical': False
            },
            'sweating': {
                'variants': ['sweating', 'diaphoresis', 'perspiration'],
                'severity': 'moderate',
                'critical': False
            },
            
            # OB/GYN specific symptoms
            'vaginal_bleeding': {
                'variants': ['vaginal bleeding', 'vaginal blood', 'bleeding per vagina', 'pvb'],
                'severity': 'critical',
                'critical': True
            },
            'abdominal_pain_pregnancy': {
                'variants': ['abdominal pain', 'pelvic pain', 'cramping'],
                'severity': 'critical',
                'critical': True,
                'context': 'pregnancy'
            },
            'contractions': {
                'variants': ['contractions', 'labor', 'labor pains', 'uterine contractions'],
                'severity': 'critical',
                'critical': True
            },
        }
    
    def _init_critical_symptoms(self):
        """Initialize list of life-threatening symptoms."""
        self.critical_symptoms = [
            'chest_pain', 'dyspnea', 'syncope', 'seizure', 'confusion',
            'jaundice', 'bleeding', 'vaginal_bleeding', 'contractions'
        ]
    
    def _init_negation_patterns(self):
        """Initialize patterns for detecting negated symptoms."""
        self.negation_patterns = [
            r'\bno\s+',
            r'\bdenies\s+',
            r'\bdenied\s+',
            r'\bwithout\s+',
            r'\babsence\s+of\s+',
            r'\bnot\s+',
            r'\bnegative\s+for\s+',
            r'\bno\s+history\s+of\s+',
        ]
    
    def extract_symptoms(
        self,
        case_description: str,
        question: Optional[str] = None
    ) -> SymptomExtractionResult:
        """
        Extract symptoms from case description.
        
        Args:
            case_description: Patient case description
            question: Optional question text (may contain symptom hints)
        
        Returns:
            SymptomExtractionResult with extracted symptoms
        """
        full_text = f"{case_description} {question or ''}".lower()
        extracted = []
        found_symptoms = set()
        
        # Extract each symptom type
        for symptom_key, symptom_info in self.symptom_dict.items():
            for variant in symptom_info['variants']:
                # Check for negation
                pattern = r'\b' + re.escape(variant) + r'\b'
                matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
                
                for match in matches:
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Check for negation in context (50 chars before)
                    context_start = max(0, start_pos - 50)
                    context_text = full_text[context_start:start_pos]
                    
                    is_negated = any(
                        re.search(neg_pattern, context_text, re.IGNORECASE)
                        for neg_pattern in self.negation_patterns
                    )
                    
                    # Only add if not negated (or if we want to track negations separately)
                    if not is_negated:
                        symptom_obj = ExtractedSymptom(
                            symptom=variant,
                            severity=symptom_info['severity'],
                            negated=False,
                            context=full_text[max(0, start_pos-20):min(len(full_text), end_pos+20)],
                            position=start_pos
                        )
                        extracted.append(symptom_obj)
                        found_symptoms.add(symptom_key)
        
        # Sort by severity and position
        extracted.sort(key=lambda x: (
            {'critical': 0, 'moderate': 1, 'mild': 2}.get(x.severity, 2),
            x.position
        ))
        
        # Get critical symptoms
        critical = [
            s.symptom for s in extracted
            if s.severity == 'critical' and not s.negated
        ]
        
        # Get all symptom terms
        all_terms = [s.symptom for s in extracted if not s.negated]
        
        # Identify missing critical symptoms (based on context)
        missing_critical = self._identify_missing_critical(full_text, found_symptoms)
        
        return SymptomExtractionResult(
            symptoms=extracted,
            critical_symptoms=critical,
            all_symptom_terms=all_terms,
            missing_critical=missing_critical
        )
    
    def _identify_missing_critical(
        self,
        text: str,
        found_symptoms: Set[str]
    ) -> List[str]:
        """
        Identify critical symptoms that should be present based on context.
        
        Args:
            text: Full case description
            found_symptoms: Set of symptom keys already found
        
        Returns:
            List of missing critical symptom names
        """
        missing = []
        
        # Context-based rules
        if 'pregnant' in text or 'pregnancy' in text or 'gestation' in text:
            # OB/GYN cases should have specific symptoms
            if 'vaginal_bleeding' not in found_symptoms and 'abdominal_pain_pregnancy' not in found_symptoms:
                if 'bleeding' not in text.lower() and 'pain' not in text.lower():
                    missing.append('vaginal_bleeding or abdominal_pain')
        
        if 'chest' in text and 'pain' not in text.lower():
            missing.append('chest_pain')
        
        if ('difficulty' in text or 'breathing' in text) and 'dyspnea' not in found_symptoms:
            missing.append('dyspnea')
        
        if 'fever' in text and 'infection' in text:
            # High fever with infection context might indicate sepsis
            if 'confusion' not in found_symptoms:
                missing.append('altered_mental_status')
        
        return missing
    
    def prioritize_symptoms(
        self,
        extraction_result: SymptomExtractionResult
    ) -> List[str]:
        """
        Prioritize symptoms by severity and relevance.
        
        Args:
            extraction_result: Result from extract_symptoms
        
        Returns:
            Prioritized list of symptom terms
        """
        # Critical symptoms first
        critical = extraction_result.critical_symptoms
        
        # Then moderate symptoms
        moderate = [
            s.symptom for s in extraction_result.symptoms
            if s.severity == 'moderate' and not s.negated
        ]
        
        # Combine with critical first
        return critical + moderate
    
    def expand_symptom_with_synonyms(self, symptom: str) -> List[str]:
        """
        Expand a symptom with UMLS synonyms (Fix 2).
        
        Args:
            symptom: Original symptom term
            
        Returns:
            List of symptom + synonyms
        """
        symptom_lower = symptom.lower().strip()
        expanded = [symptom_lower]
        
        # Direct match in UMLS synonyms
        if symptom_lower in self.umls_synonyms:
            expanded.extend(self.umls_synonyms[symptom_lower])
        
        # Also check if symptom is a synonym of another term
        for term, synonyms in self.umls_synonyms.items():
            if symptom_lower in synonyms and term not in expanded:
                expanded.append(term)
                expanded.extend([s for s in synonyms if s not in expanded])
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for s in expanded:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        
        return unique[:5]  # Limit to 5 expansions
    
    def get_symptom_query_terms(
        self,
        case_description: str,
        question: Optional[str] = None
    ) -> str:
        """
        Get symptom terms formatted for query enhancement.
        
        Fix 2: Now includes symptom expansion with UMLS synonyms.
        
        Args:
            case_description: Case description
            question: Optional question
        
        Returns:
            String of symptom terms for query expansion (with synonyms)
        """
        result = self.extract_symptoms(case_description, question)
        prioritized = self.prioritize_symptoms(result)
        
        # Add missing critical symptoms as hints
        if result.missing_critical:
            prioritized.extend(result.missing_critical)
        
        # Fix 2: Expand symptoms with synonyms
        expanded_terms = []
        for symptom in prioritized[:5]:  # Limit to top 5 symptoms
            synonyms = self.expand_symptom_with_synonyms(symptom)
            expanded_terms.extend(synonyms[:3])  # Add up to 3 synonyms per symptom
        
        # Remove duplicates
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return ' '.join(unique_terms[:15])  # Limit to 15 terms total

