"""
Symptom Keyword Injection Module

Step 5 Fix: Multi-query retrieval with symptom injection + clinical synonyms.

Adds synonyms for key clinical terms:
- hypotension → low BP, shock
- hematemesis → vomiting blood
- dyspnea → shortness of breath

Example:
"hematemesis" → ["hematemesis", "vomiting blood", "upper GI bleed"]

Expected improvement:
- Missing symptom rate drops from 16 → 4
- Accuracy improves by 5-10%
"""

import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class SymptomInjectionResult:
    """Result of symptom extraction and synonym injection."""
    original_symptoms: List[str]
    expanded_symptoms: List[str]
    symptom_query_string: str
    critical_symptoms: List[str]


class SymptomSynonymInjector:
    """
    Extract symptoms and inject synonyms for better retrieval.
    
    This is a rule-based approach using a comprehensive dictionary.
    No model needed - just pattern matching and synonym expansion.
    """
    
    def __init__(self):
        """Initialize with comprehensive symptom synonym dictionary."""
        self._init_symptom_synonyms()
        self._init_symptom_patterns()
    
    def _init_symptom_synonyms(self):
        """
        Initialize comprehensive symptom synonym dictionary.
        
        Format: primary_symptom -> [synonyms and related terms]
        """
        self.symptom_synonyms = {
            # GI Bleeding
            'hematemesis': [
                'hematemesis', 'vomiting blood', 'blood in vomit', 'coffee ground vomitus',
                'upper GI bleed', 'upper gastrointestinal bleeding', 'bloody vomit'
            ],
            'melena': [
                'melena', 'black stool', 'tarry stool', 'dark stool', 'blood in stool',
                'GI bleed', 'gastrointestinal bleeding'
            ],
            'hematochezia': [
                'hematochezia', 'bloody stool', 'rectal bleeding', 'fresh blood per rectum',
                'lower GI bleed', 'bright red blood per rectum'
            ],
            
            # Respiratory
            'dyspnea': [
                'dyspnea', 'shortness of breath', 'difficulty breathing', 'breathlessness',
                'respiratory distress', 'labored breathing', 'air hunger', 'SOB'
            ],
            'cough': [
                'cough', 'coughing', 'productive cough', 'dry cough', 'nonproductive cough',
                'persistent cough', 'chronic cough', 'hacking cough'
            ],
            'hemoptysis': [
                'hemoptysis', 'coughing blood', 'blood in sputum', 'bloody sputum',
                'blood tinged sputum', 'pulmonary hemorrhage'
            ],
            'wheezing': [
                'wheezing', 'wheeze', 'respiratory wheeze', 'bronchospasm',
                'stridor', 'airway obstruction'
            ],
            'stridor': [
                'stridor', 'noisy breathing', 'inspiratory stridor', 'expiratory stridor',
                'upper airway obstruction', 'laryngeal stridor'
            ],
            
            # Cardiovascular
            'chest_pain': [
                'chest pain', 'chest discomfort', 'angina', 'precordial pain',
                'retrosternal pain', 'chest tightness', 'cardiac pain', 'anginal pain'
            ],
            'palpitations': [
                'palpitations', 'heart racing', 'rapid heartbeat', 'irregular heartbeat',
                'skipped beats', 'tachycardia', 'arrhythmia', 'fluttering'
            ],
            'syncope': [
                'syncope', 'fainting', 'loss of consciousness', 'passing out',
                'blackout', 'collapse', 'LOC', 'near syncope', 'presyncope'
            ],
            'edema': [
                'edema', 'swelling', 'fluid retention', 'peripheral edema',
                'pitting edema', 'leg swelling', 'ankle swelling', 'oedema'
            ],
            
            # Neurological
            'headache': [
                'headache', 'head pain', 'cephalgia', 'migraine', 'tension headache',
                'severe headache', 'throbbing headache', 'holocranial headache'
            ],
            'seizure': [
                'seizure', 'convulsion', 'fit', 'epileptic seizure', 'tonic clonic',
                'grand mal', 'petit mal', 'convulsions', 'seizure activity'
            ],
            'confusion': [
                'confusion', 'altered mental status', 'disorientation', 'AMS',
                'mental status change', 'encephalopathy', 'delirium', 'obtunded'
            ],
            'weakness': [
                'weakness', 'fatigue', 'lethargy', 'asthenia', 'muscle weakness',
                'generalized weakness', 'malaise', 'tiredness'
            ],
            'numbness': [
                'numbness', 'tingling', 'paresthesia', 'loss of sensation',
                'pins and needles', 'sensory loss', 'hypoesthesia'
            ],
            'dizziness': [
                'dizziness', 'vertigo', 'lightheadedness', 'giddiness',
                'unsteadiness', 'imbalance', 'disequilibrium'
            ],
            
            # GI Symptoms
            'nausea': [
                'nausea', 'feeling sick', 'queasiness', 'upset stomach',
                'stomach upset', 'nauseated'
            ],
            'vomiting': [
                'vomiting', 'emesis', 'throwing up', 'retching', 'projectile vomiting',
                'persistent vomiting', 'intractable vomiting'
            ],
            'diarrhea': [
                'diarrhea', 'loose stools', 'watery stool', 'frequent bowel movements',
                'bloody diarrhea', 'dysentery', 'diarrhoea'
            ],
            'constipation': [
                'constipation', 'difficulty passing stool', 'hard stool',
                'infrequent bowel movements', 'straining'
            ],
            'abdominal_pain': [
                'abdominal pain', 'stomach pain', 'belly pain', 'abdominal discomfort',
                'epigastric pain', 'periumbilical pain', 'RLQ pain', 'RUQ pain',
                'LLQ pain', 'LUQ pain', 'cramping', 'colic', 'abdominal cramps'
            ],
            'jaundice': [
                'jaundice', 'yellowing', 'icterus', 'yellow skin', 'yellow eyes',
                'scleral icterus', 'hepatic jaundice', 'obstructive jaundice'
            ],
            
            # Fever/Infection
            'fever': [
                'fever', 'pyrexia', 'elevated temperature', 'high temperature',
                'hyperthermia', 'febrile', 'temperature elevation', 'high fever'
            ],
            'chills': [
                'chills', 'shivering', 'rigors', 'cold sensation', 'shaking chills'
            ],
            'night_sweats': [
                'night sweats', 'nocturnal sweating', 'diaphoresis', 'sweating',
                'profuse sweating'
            ],
            
            # Skin
            'rash': [
                'rash', 'skin rash', 'eruption', 'dermatitis', 'skin lesion',
                'maculopapular rash', 'petechiae', 'purpura', 'urticaria'
            ],
            'itching': [
                'itching', 'pruritus', 'scratching', 'itch', 'itchy skin'
            ],
            
            # Genitourinary
            'dysuria': [
                'dysuria', 'painful urination', 'burning urination', 'urinary pain',
                'difficulty urinating', 'painful micturition'
            ],
            'hematuria': [
                'hematuria', 'blood in urine', 'bloody urine', 'gross hematuria',
                'microscopic hematuria', 'red urine'
            ],
            'urinary_frequency': [
                'urinary frequency', 'frequent urination', 'polyuria',
                'increased urination', 'nocturia'
            ],
            
            # OB/GYN
            'vaginal_bleeding': [
                'vaginal bleeding', 'per vaginal bleeding', 'PV bleeding',
                'menorrhagia', 'metrorrhagia', 'spotting', 'abnormal uterine bleeding',
                'postpartum hemorrhage', 'PPH', 'antepartum hemorrhage'
            ],
            'contractions': [
                'contractions', 'labor pains', 'uterine contractions', 'labor',
                'preterm labor', 'regular contractions'
            ],
            'labor': [
                'labor', 'delivery', 'parturition', 'childbirth', 'active labor',
                'first stage labor', 'second stage labor'
            ],
            
            # Pediatric
            'poor_feeding': [
                'poor feeding', 'feeding difficulty', 'decreased feeding',
                'refusal to feed', 'feeding intolerance', 'not feeding well'
            ],
            'irritability': [
                'irritability', 'irritable', 'fussy', 'crying', 'inconsolable',
                'excessive crying'
            ],
            'lethargy_infant': [
                'lethargy', 'lethargic', 'decreased activity', 'less active',
                'weak cry', 'floppy', 'hypotonia'
            ],
            
            # Pain types
            'back_pain': [
                'back pain', 'backache', 'lumbar pain', 'lumbago', 'lower back pain',
                'spinal pain', 'dorsalgia'
            ],
            'joint_pain': [
                'joint pain', 'arthralgia', 'joint swelling', 'arthritis',
                'polyarthralgia', 'joint stiffness'
            ],
            'muscle_pain': [
                'muscle pain', 'myalgia', 'muscle ache', 'body aches',
                'muscle soreness', 'muscular pain'
            ],
            
            # Eye symptoms
            'vision_changes': [
                'vision changes', 'blurred vision', 'visual disturbance',
                'double vision', 'diplopia', 'vision loss', 'scotoma'
            ],
            'eye_pain': [
                'eye pain', 'ocular pain', 'photophobia', 'light sensitivity',
                'painful eye', 'red eye'
            ],
            
            # Weight/appetite
            'weight_loss': [
                'weight loss', 'unintentional weight loss', 'cachexia',
                'loss of appetite', 'anorexia', 'decreased appetite'
            ],
            
            # STEP 5 FIX: Additional clinical synonyms
            # Hemodynamic
            'hypotension': [
                'hypotension', 'low blood pressure', 'low BP', 'shock',
                'hypotensive', 'BP low', 'circulatory collapse', 'hemodynamic instability'
            ],
            'hypertension': [
                'hypertension', 'high blood pressure', 'high BP', 'elevated BP',
                'hypertensive', 'BP elevated', 'blood pressure elevated'
            ],
            'tachycardia': [
                'tachycardia', 'rapid heart rate', 'fast pulse', 'elevated HR',
                'heart rate elevated', 'pulse rapid', 'racing heart'
            ],
            'bradycardia': [
                'bradycardia', 'slow heart rate', 'low pulse', 'decreased HR',
                'heart rate low', 'pulse slow'
            ],
            
            # Shock states
            'shock': [
                'shock', 'hypotensive shock', 'circulatory shock', 'hemodynamic collapse',
                'septic shock', 'hypovolemic shock', 'cardiogenic shock', 'distributive shock'
            ],
            'sepsis': [
                'sepsis', 'septic', 'systemic infection', 'bacteremia',
                'septicemia', 'blood infection', 'severe sepsis', 'sepsis syndrome'
            ],
            
            # Respiratory failure
            'respiratory_failure': [
                'respiratory failure', 'respiratory distress', 'hypoxia', 'hypoxemia',
                'oxygen desaturation', 'low oxygen', 'SpO2 low', 'respiratory compromise'
            ],
            'hypoxia': [
                'hypoxia', 'hypoxemia', 'low oxygen', 'oxygen desaturation',
                'poor oxygenation', 'hypoxic', 'low SpO2', 'cyanosis'
            ],
            
            # Metabolic
            'hypoglycemia': [
                'hypoglycemia', 'low blood sugar', 'low glucose', 'hypoglycemic',
                'blood sugar low', 'glucose low', 'symptomatic hypoglycemia'
            ],
            'hyperglycemia': [
                'hyperglycemia', 'high blood sugar', 'high glucose', 'hyperglycemic',
                'blood sugar elevated', 'glucose elevated', 'diabetic ketoacidosis', 'DKA'
            ],
            'dehydration': [
                'dehydration', 'dehydrated', 'volume depletion', 'hypovolemia',
                'dry mucous membranes', 'poor skin turgor', 'fluid deficit'
            ],
            
            # Acute conditions
            'acute_abdomen': [
                'acute abdomen', 'surgical abdomen', 'peritonitis', 'guarding',
                'rebound tenderness', 'rigid abdomen', 'abdominal rigidity'
            ],
            'meningismus': [
                'meningismus', 'meningeal signs', 'neck stiffness', 'nuchal rigidity',
                'Kernig sign', 'Brudzinski sign', 'meningitis'
            ],
            
            # Bleeding/Coagulation
            'bleeding': [
                'bleeding', 'hemorrhage', 'blood loss', 'hemorrhaging',
                'active bleeding', 'uncontrolled bleeding'
            ],
            'bruising': [
                'bruising', 'ecchymosis', 'contusion', 'easy bruising',
                'unexplained bruising'
            ],
        }
        
        # Create reverse mapping for fast lookup
        self.symptom_to_primary = {}
        for primary, synonyms in self.symptom_synonyms.items():
            for synonym in synonyms:
                self.symptom_to_primary[synonym.lower()] = primary
    
    def _init_symptom_patterns(self):
        """Initialize regex patterns for symptom extraction."""
        self.symptom_patterns = [
            # Common presentations
            r'\b(presents? with|complaining of|c/o|has|had|developed?|experiencing?)\s+([^,\.]+)',
            # Symptoms listed
            r'\b(symptoms?|signs?|features?)\s+(?:include|of|are)?\s*:?\s*([^,\.]+)',
            # Direct symptom mentions
            r'\b(pain|fever|cough|bleeding|vomiting|diarrhea|headache|weakness|fatigue)\b',
            # Duration patterns
            r'(\d+)\s*(day|week|month|hour)s?\s+(?:of|history of)\s+([^,\.]+)',
        ]
        
        # Critical symptoms that should always be captured
        self.critical_symptom_keywords = [
            'chest pain', 'shortness of breath', 'syncope', 'seizure', 'fever',
            'bleeding', 'confusion', 'weakness', 'headache', 'abdominal pain',
            'vomiting', 'diarrhea', 'rash', 'jaundice', 'cough', 'dyspnea'
        ]
    
    def extract_and_expand(
        self,
        text: str,
        include_critical: bool = True
    ) -> SymptomInjectionResult:
        """
        Extract symptoms from text and expand with synonyms.
        
        Args:
            text: Input text (case description + question)
            include_critical: Whether to prioritize critical symptoms
            
        Returns:
            SymptomInjectionResult with original and expanded symptoms
        """
        text_lower = text.lower()
        
        # Step 1: Extract original symptoms
        original_symptoms = self._extract_symptoms(text_lower)
        
        # Step 2: Expand with synonyms
        expanded_symptoms = set()
        for symptom in original_symptoms:
            expanded = self._expand_symptom(symptom)
            expanded_symptoms.update(expanded)
        
        # Step 3: Identify critical symptoms
        critical_symptoms = []
        for keyword in self.critical_symptom_keywords:
            if keyword in text_lower:
                critical_symptoms.append(keyword)
                # Also expand critical symptoms
                expanded = self._expand_symptom(keyword)
                expanded_symptoms.update(expanded)
        
        # Step 4: Create query string for injection
        expanded_list = list(expanded_symptoms)
        symptom_query_string = ' '.join(expanded_list)
        
        return SymptomInjectionResult(
            original_symptoms=original_symptoms,
            expanded_symptoms=expanded_list,
            symptom_query_string=symptom_query_string,
            critical_symptoms=critical_symptoms
        )
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text using pattern matching."""
        symptoms = []
        
        # Check for known symptoms
        for primary, synonyms in self.symptom_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in text:
                    symptoms.append(synonym)
                    break  # Only add once per symptom group
        
        # Also extract any directly mentioned critical symptoms
        for keyword in self.critical_symptom_keywords:
            if keyword in text and keyword not in symptoms:
                symptoms.append(keyword)
        
        return symptoms
    
    def _expand_symptom(self, symptom: str) -> List[str]:
        """Expand a single symptom with all synonyms."""
        symptom_lower = symptom.lower().strip()
        
        # Check if it's a known symptom
        if symptom_lower in self.symptom_to_primary:
            primary = self.symptom_to_primary[symptom_lower]
            return self.symptom_synonyms.get(primary, [symptom])
        
        # Check if it matches any synonym
        for primary, synonyms in self.symptom_synonyms.items():
            for syn in synonyms:
                if symptom_lower in syn.lower() or syn.lower() in symptom_lower:
                    return synonyms
        
        # No expansion found, return original
        return [symptom]
    
    def get_injection_string(
        self,
        case_description: str,
        question: str
    ) -> str:
        """
        Get the symptom injection string to append to retrieval query.
        
        Args:
            case_description: Patient case description
            question: The clinical question
            
        Returns:
            String of expanded symptoms to append to query
        """
        full_text = f"{case_description} {question}"
        result = self.extract_and_expand(full_text)
        return result.symptom_query_string
    
    def get_critical_symptoms(
        self,
        case_description: str,
        question: str
    ) -> List[str]:
        """Get list of critical symptoms found in the text."""
        full_text = f"{case_description} {question}"
        result = self.extract_and_expand(full_text)
        return result.critical_symptoms


def main():
    """Demo: Test symptom synonym injection."""
    print("="*70)
    print("SYMPTOM SYNONYM INJECTOR DEMO")
    print("="*70)
    
    injector = SymptomSynonymInjector()
    
    test_cases = [
        "Patient presents with hematemesis and melena for 2 days.",
        "A 65-year-old male with chest pain and shortness of breath.",
        "Newborn with fever, poor feeding, and lethargy.",
        "Woman at 32 weeks with vaginal bleeding and abdominal pain.",
        "Child with seizure and high fever.",
    ]
    
    for case in test_cases:
        print(f"\n{'-'*70}")
        print(f"Input: {case}")
        print(f"{'-'*70}")
        
        result = injector.extract_and_expand(case)
        
        print(f"\nOriginal symptoms: {result.original_symptoms}")
        print(f"Critical symptoms: {result.critical_symptoms}")
        print(f"Expanded symptoms ({len(result.expanded_symptoms)} total):")
        
        # Show first 10 expanded terms
        for sym in result.expanded_symptoms[:10]:
            print(f"  - {sym}")
        
        if len(result.expanded_symptoms) > 10:
            print(f"  ... and {len(result.expanded_symptoms) - 10} more")
        
        print(f"\nInjection query string (first 200 chars):")
        print(f"  {result.symptom_query_string[:200]}...")
    
    print(f"\n{'='*70}")
    print("[OK] Symptom Synonym Injector operational!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

