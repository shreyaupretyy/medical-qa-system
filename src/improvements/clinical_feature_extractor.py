"""
Clinical Feature Extractor

Step 1 Fix: Rule-based extraction before retrieval:
- Symptoms
- Vitals
- Risk factors  
- Chronic diseases
- Lab values

Uses regex + dictionary - simple but effective.
Expected gain: +10-15% accuracy
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClinicalFeatures:
    """Extracted clinical features from case description."""
    symptoms: List[str]
    vitals: Dict[str, str]  # e.g., {"BP": "90/60", "HR": "120"}
    risk_factors: List[str]
    chronic_diseases: List[str]
    lab_values: Dict[str, str]  # e.g., {"Hb": "7.2", "glucose": "450"}
    medications: List[str]
    procedures: List[str]
    demographics: Dict[str, str]  # age, gender, pregnancy status
    
    def to_query_string(self) -> str:
        """Convert features to a query-friendly string."""
        parts = []
        
        # Add symptoms (most important)
        if self.symptoms:
            parts.append(" ".join(self.symptoms[:5]))
        
        # Add chronic diseases
        if self.chronic_diseases:
            parts.append(" ".join(self.chronic_diseases[:3]))
        
        # Add abnormal vitals as keywords
        if self.vitals:
            for vital, value in self.vitals.items():
                if self._is_abnormal_vital(vital, value):
                    parts.append(f"abnormal {vital}")
        
        # Add risk factors
        if self.risk_factors:
            parts.append(" ".join(self.risk_factors[:3]))
        
        return " ".join(parts)
    
    def _is_abnormal_vital(self, vital: str, value: str) -> bool:
        """Check if a vital sign value is abnormal."""
        try:
            if vital.lower() in ['hr', 'heart rate', 'pulse']:
                val = float(re.search(r'\d+', value).group())
                return val < 60 or val > 100
            elif vital.lower() in ['bp', 'blood pressure']:
                systolic = float(re.search(r'(\d+)/', value).group(1))
                return systolic < 90 or systolic > 140
            elif vital.lower() in ['temp', 'temperature']:
                val = float(re.search(r'[\d.]+', value).group())
                return val > 38.0 or val < 36.0
            elif vital.lower() in ['rr', 'respiratory rate']:
                val = float(re.search(r'\d+', value).group())
                return val < 12 or val > 20
            elif vital.lower() in ['spo2', 'oxygen saturation']:
                val = float(re.search(r'\d+', value).group())
                return val < 95
        except:
            pass
        return False
    
    def get_critical_findings(self) -> List[str]:
        """Get list of critical/urgent findings."""
        critical = []
        
        # Critical symptoms
        critical_symptoms = [
            'chest pain', 'shortness of breath', 'syncope', 'seizure',
            'altered mental status', 'confusion', 'unconscious', 'bleeding',
            'severe pain', 'respiratory distress', 'shock'
        ]
        for symptom in self.symptoms:
            if any(cs in symptom.lower() for cs in critical_symptoms):
                critical.append(symptom)
        
        # Abnormal vitals
        for vital, value in self.vitals.items():
            if self._is_abnormal_vital(vital, value):
                critical.append(f"{vital}: {value}")
        
        return critical


class ClinicalFeatureExtractor:
    """
    Extract clinical features from case descriptions using rules and patterns.
    """
    
    def __init__(self):
        self._init_symptom_patterns()
        self._init_vital_patterns()
        self._init_disease_patterns()
        self._init_lab_patterns()
        self._init_risk_factors()
        self._init_medication_patterns()
    
    def _init_symptom_patterns(self):
        """Initialize symptom extraction patterns."""
        self.symptom_keywords = [
            # Pain
            'pain', 'ache', 'tenderness', 'discomfort', 'cramping',
            'chest pain', 'abdominal pain', 'headache', 'back pain',
            'joint pain', 'muscle pain', 'pelvic pain',
            
            # Respiratory
            'cough', 'dyspnea', 'shortness of breath', 'wheezing',
            'stridor', 'hemoptysis', 'sputum', 'breathlessness',
            
            # GI
            'nausea', 'vomiting', 'diarrhea', 'constipation',
            'hematemesis', 'melena', 'hematochezia', 'dysphagia',
            'abdominal distension', 'bloating',
            
            # Neurological
            'headache', 'dizziness', 'vertigo', 'syncope', 'seizure',
            'confusion', 'altered mental status', 'weakness', 'numbness',
            'tingling', 'paralysis', 'tremor',
            
            # Cardiovascular
            'palpitations', 'chest tightness', 'edema', 'swelling',
            
            # General
            'fever', 'chills', 'night sweats', 'fatigue', 'malaise',
            'weight loss', 'weight gain', 'anorexia', 'lethargy',
            
            # Skin
            'rash', 'itching', 'pruritus', 'jaundice', 'pallor',
            'cyanosis', 'petechiae', 'bruising',
            
            # Urinary
            'dysuria', 'frequency', 'urgency', 'hematuria',
            'oliguria', 'polyuria', 'incontinence',
            
            # OB/GYN
            'vaginal bleeding', 'discharge', 'contractions',
            'decreased fetal movement', 'amenorrhea',
            
            # Pediatric
            'poor feeding', 'irritability', 'excessive crying',
            'failure to thrive', 'developmental delay',
        ]
        
        self.symptom_patterns = [
            r'presents? with ([^,\.]+)',
            r'complains? of ([^,\.]+)',
            r'c/o ([^,\.]+)',
            r'reports? ([^,\.]+)',
            r'experiencing ([^,\.]+)',
            r'developed ([^,\.]+)',
            r'has ([^,\.]+) for',
            r'(\w+ pain)',
            r'(progressive \w+)',
            r'(severe \w+)',
            r'(acute \w+)',
        ]
    
    def _init_vital_patterns(self):
        """Initialize vital sign extraction patterns."""
        self.vital_patterns = {
            'BP': [
                r'(?:BP|blood pressure)[:\s]*(\d+/\d+)',
                r'(\d+/\d+)\s*(?:mmHg|mm Hg)',
            ],
            'HR': [
                r'(?:HR|heart rate|pulse)[:\s]*(\d+)',
                r'(\d+)\s*(?:bpm|beats per minute)',
            ],
            'RR': [
                r'(?:RR|respiratory rate)[:\s]*(\d+)',
                r'(\d+)\s*(?:breaths?/min)',
            ],
            'Temp': [
                r'(?:temp|temperature)[:\s]*([\d.]+)',
                r'([\d.]+)\s*°?[CF]',
                r'fever of ([\d.]+)',
            ],
            'SpO2': [
                r'(?:SpO2|O2 sat|oxygen saturation)[:\s]*(\d+)',
                r'(\d+)%\s*(?:on|room air|RA)',
            ],
            'GCS': [
                r'(?:GCS|Glasgow)[:\s]*(\d+)',
            ],
        }
    
    def _init_disease_patterns(self):
        """Initialize chronic disease patterns."""
        self.chronic_diseases = [
            # Cardiovascular
            'hypertension', 'htn', 'diabetes', 'dm', 'type 2 diabetes',
            'type 1 diabetes', 'heart failure', 'chf', 'coronary artery disease',
            'cad', 'atrial fibrillation', 'afib',
            
            # Respiratory
            'asthma', 'copd', 'chronic obstructive pulmonary disease',
            
            # Renal
            'chronic kidney disease', 'ckd', 'esrd', 'end stage renal disease',
            
            # Hepatic
            'cirrhosis', 'hepatitis', 'liver disease',
            
            # Endocrine
            'hypothyroidism', 'hyperthyroidism', 'thyroid disease',
            
            # Neurological
            'epilepsy', 'parkinson', 'alzheimer', 'dementia',
            
            # Autoimmune
            'rheumatoid arthritis', 'lupus', 'sle', 'multiple sclerosis',
            
            # Infectious
            'hiv', 'aids', 'tuberculosis', 'tb',
            
            # Cancer
            'cancer', 'malignancy', 'tumor', 'carcinoma', 'lymphoma', 'leukemia',
        ]
        
        self.disease_patterns = [
            r'history of ([^,\.]+)',
            r'known ([^,\.]+)',
            r'diagnosed with ([^,\.]+)',
            r'on treatment for ([^,\.]+)',
        ]
    
    def _init_lab_patterns(self):
        """Initialize lab value extraction patterns."""
        self.lab_patterns = {
            'Hb': [
                r'(?:Hb|hemoglobin)[:\s]*([\d.]+)',
                r'([\d.]+)\s*g/dL',
            ],
            'WBC': [
                r'(?:WBC|white blood cell)[:\s]*([\d.]+)',
                r'([\d.]+)\s*(?:x10\^9|cells/uL)',
            ],
            'Platelets': [
                r'(?:platelets?|plt)[:\s]*([\d.]+)',
            ],
            'Glucose': [
                r'(?:glucose|blood sugar|BS)[:\s]*([\d.]+)',
                r'([\d.]+)\s*mg/dL',
            ],
            'Creatinine': [
                r'(?:creatinine|Cr)[:\s]*([\d.]+)',
            ],
            'BUN': [
                r'(?:BUN|blood urea nitrogen)[:\s]*([\d.]+)',
            ],
            'Sodium': [
                r'(?:sodium|Na)[:\s]*([\d.]+)',
            ],
            'Potassium': [
                r'(?:potassium|K)[:\s]*([\d.]+)',
            ],
            'Troponin': [
                r'(?:troponin)[:\s]*([\d.]+)',
            ],
            'Lactate': [
                r'(?:lactate)[:\s]*([\d.]+)',
            ],
        }
    
    def _init_risk_factors(self):
        """Initialize risk factor patterns."""
        self.risk_factors = [
            'smoking', 'smoker', 'tobacco',
            'alcohol', 'alcoholic', 'drinking',
            'obesity', 'obese', 'overweight',
            'sedentary', 'inactive',
            'family history', 
            'immunocompromised', 'immunosuppressed',
            'pregnant', 'pregnancy',
            'elderly', 'advanced age',
            'diabetes', 'hypertension',
            'hyperlipidemia', 'high cholesterol',
        ]
    
    def _init_medication_patterns(self):
        """Initialize medication patterns."""
        self.medication_keywords = [
            'aspirin', 'metformin', 'insulin', 'lisinopril', 'atorvastatin',
            'metoprolol', 'amlodipine', 'omeprazole', 'losartan', 'gabapentin',
            'levothyroxine', 'prednisone', 'warfarin', 'heparin', 'enoxaparin',
            'furosemide', 'hydrochlorothiazide', 'carvedilol', 'pantoprazole',
            'clopidogrel', 'albuterol', 'montelukast', 'fluticasone',
            'acetaminophen', 'ibuprofen', 'naproxen', 'morphine', 'oxycodone',
            'amoxicillin', 'azithromycin', 'ciprofloxacin', 'ceftriaxone',
            'vancomycin', 'metronidazole', 'doxycycline', 'gentamicin',
        ]
        
        self.medication_patterns = [
            r'on ([^,\.]+(?:mg|mcg|units?))',
            r'taking ([^,\.]+)',
            r'prescribed ([^,\.]+)',
            r'started on ([^,\.]+)',
        ]
    
    def extract(self, case_description: str, question: str = "") -> ClinicalFeatures:
        """
        Extract all clinical features from case description.
        
        Args:
            case_description: Patient case text
            question: Optional question text
            
        Returns:
            ClinicalFeatures dataclass with all extracted features
        """
        full_text = f"{case_description} {question}".lower()
        
        symptoms = self._extract_symptoms(full_text)
        vitals = self._extract_vitals(case_description)
        chronic_diseases = self._extract_chronic_diseases(full_text)
        labs = self._extract_labs(case_description)
        risk_factors = self._extract_risk_factors(full_text)
        medications = self._extract_medications(full_text)
        demographics = self._extract_demographics(case_description)
        
        return ClinicalFeatures(
            symptoms=symptoms,
            vitals=vitals,
            risk_factors=risk_factors,
            chronic_diseases=chronic_diseases,
            lab_values=labs,
            medications=medications,
            procedures=[],  # Could be expanded
            demographics=demographics
        )
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text."""
        symptoms = set()
        
        # Direct keyword matching
        for keyword in self.symptom_keywords:
            if keyword in text:
                symptoms.add(keyword)
        
        # Pattern matching
        for pattern in self.symptom_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean and add
                cleaned = match.strip().lower()
                if len(cleaned) > 3 and len(cleaned) < 50:
                    symptoms.add(cleaned)
        
        return list(symptoms)
    
    def _extract_vitals(self, text: str) -> Dict[str, str]:
        """Extract vital signs from text."""
        vitals = {}
        
        for vital_name, patterns in self.vital_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    vitals[vital_name] = match.group(1)
                    break
        
        return vitals
    
    def _extract_chronic_diseases(self, text: str) -> List[str]:
        """Extract chronic diseases from text."""
        diseases = set()
        
        # Direct matching
        for disease in self.chronic_diseases:
            if disease in text:
                diseases.add(disease)
        
        # Pattern matching
        for pattern in self.disease_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip().lower()
                if len(cleaned) > 3:
                    diseases.add(cleaned)
        
        return list(diseases)
    
    def _extract_labs(self, text: str) -> Dict[str, str]:
        """Extract lab values from text."""
        labs = {}
        
        for lab_name, patterns in self.lab_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    labs[lab_name] = match.group(1)
                    break
        
        return labs
    
    def _extract_risk_factors(self, text: str) -> List[str]:
        """Extract risk factors from text."""
        factors = []
        for factor in self.risk_factors:
            if factor in text:
                factors.append(factor)
        return factors
    
    def _extract_medications(self, text: str) -> List[str]:
        """Extract medications from text."""
        meds = set()
        
        # Direct matching
        for med in self.medication_keywords:
            if med in text:
                meds.add(med)
        
        # Pattern matching
        for pattern in self.medication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip().lower()
                if len(cleaned) > 3:
                    meds.add(cleaned)
        
        return list(meds)
    
    def _extract_demographics(self, text: str) -> Dict[str, str]:
        """Extract demographic information."""
        demographics = {}
        text_lower = text.lower()
        
        # Age
        age_patterns = [
            r'(\d+)[- ]?(year|yr)[- ]?old',
            r'(\d+)[- ]?(month|mo)[- ]?old',
            r'(\d+)[- ]?(day|week)[- ]?old',
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                demographics['age'] = match.group(0)
                break
        
        # Check for newborn/infant/child
        if 'newborn' in text_lower or 'neonate' in text_lower:
            demographics['age_group'] = 'newborn'
        elif 'infant' in text_lower:
            demographics['age_group'] = 'infant'
        elif 'child' in text_lower or 'pediatric' in text_lower:
            demographics['age_group'] = 'child'
        elif 'elderly' in text_lower or 'geriatric' in text_lower:
            demographics['age_group'] = 'elderly'
        
        # Gender
        if 'female' in text_lower or 'woman' in text_lower or 'girl' in text_lower:
            demographics['gender'] = 'female'
        elif 'male' in text_lower or 'man' in text_lower or 'boy' in text_lower:
            demographics['gender'] = 'male'
        
        # Pregnancy
        if 'pregnant' in text_lower or 'pregnancy' in text_lower or 'gestation' in text_lower:
            demographics['pregnant'] = 'yes'
            # Extract gestational age
            ga_match = re.search(r'(\d+)\s*weeks?\s*(?:gestation|pregnant)', text_lower)
            if ga_match:
                demographics['gestational_age'] = ga_match.group(1) + ' weeks'
        
        return demographics


def main():
    """Test clinical feature extraction."""
    print("="*70)
    print("CLINICAL FEATURE EXTRACTOR TEST")
    print("="*70)
    
    extractor = ClinicalFeatureExtractor()
    
    test_cases = [
        """A 65-year-old male with history of hypertension and diabetes presents with 
        chest pain and shortness of breath for 2 hours. BP 90/60, HR 110, SpO2 92%. 
        Troponin elevated at 2.5. On aspirin and metformin.""",
        
        """A 3-day-old newborn with fever, poor feeding, and lethargy. 
        Temperature 38.5°C, HR 180. WBC 25,000. Mother had prolonged rupture of membranes.""",
        
        """A 28-year-old pregnant woman at 34 weeks gestation with severe headache, 
        visual changes, and BP 160/110. History of preeclampsia in previous pregnancy.""",
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Case {i}:")
        print(f"{'='*70}")
        print(case[:100] + "...")
        
        features = extractor.extract(case)
        
        print(f"\nExtracted Features:")
        print(f"  Symptoms: {features.symptoms}")
        print(f"  Vitals: {features.vitals}")
        print(f"  Chronic diseases: {features.chronic_diseases}")
        print(f"  Labs: {features.lab_values}")
        print(f"  Risk factors: {features.risk_factors}")
        print(f"  Medications: {features.medications}")
        print(f"  Demographics: {features.demographics}")
        print(f"  Critical findings: {features.get_critical_findings()}")
        print(f"\n  Query string: {features.to_query_string()}")
    
    print(f"\n{'='*70}")
    print("[OK] Clinical Feature Extractor operational!")


if __name__ == "__main__":
    main()

