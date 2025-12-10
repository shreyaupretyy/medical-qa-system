"""
Medical Query Understanding Module

This module processes medical queries to extract clinical features,
identify medical terminology, and understand clinical context.

Key Capabilities:
- Medical terminology expansion (abbreviations, synonyms)
- Clinical feature extraction (symptoms, demographics, context)
- Medical specialty identification
- Acuity level determination
- Negation handling
"""

import re
import json
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path


@dataclass
class ClinicalFeatures:
    """Extracted clinical features from a query."""
    symptoms: List[str]
    demographics: Dict[str, Optional[str]]  # age, gender, etc.
    medical_terms: List[str]
    medications: List[str]
    tests: List[str]
    conditions: List[str]
    urgency_keywords: List[str]
    specialty_hints: List[str]
    negations: List[str]  # Negated findings


@dataclass
class QueryUnderstanding:
    """Complete understanding of a medical query."""
    original_query: str
    expanded_query: str
    clinical_features: ClinicalFeatures
    likely_specialty: Optional[str]
    acuity_level: str  # emergency, urgent, routine
    query_type: str  # diagnosis, treatment, management, test_ordering
    clinical_intent: Optional[Any] = None  # Day 7 Phase 2: Clinical intent classification


class MedicalQueryUnderstanding:
    """
    Understands and processes medical queries.
    
    Handles:
    - Medical abbreviation expansion
    - Synonym recognition
    - Clinical feature extraction
    - Specialty identification
    - Negation detection
    """
    
    def __init__(self):
        """Initialize with medical knowledge bases."""
        self._init_medical_abbreviations()
        self._init_medical_synonyms()
        self._init_specialty_keywords()
        self._init_urgency_keywords()
        self._init_negation_patterns()
        self._load_external_umls()  # Merge external expansions if available
    
    def _init_medical_abbreviations(self):
        """Initialize medical abbreviation dictionary."""
        self.abbreviations = {
            'mi': 'myocardial infarction',
            'ami': 'acute myocardial infarction',
            'stemi': 'st-elevation myocardial infarction',
            'nstemi': 'non-st-elevation myocardial infarction',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'dm': 'diabetes mellitus',
            'dm2': 'diabetes mellitus type 2',
            'htn': 'hypertension',
            'cad': 'coronary artery disease',
            'pe': 'pulmonary embolism',
            'dvt': 'deep vein thrombosis',
            'uti': 'urinary tract infection',
            'pneumonia': 'pneumonia',
            'sepsis': 'sepsis',
            'aki': 'acute kidney injury',
            'ckd': 'chronic kidney disease',
            'ecg': 'electrocardiogram',
            'ekg': 'electrocardiogram',
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'ace': 'angiotensin converting enzyme',
            'arbs': 'angiotensin receptor blockers',
            'tpa': 'tissue plasminogen activator',
            'asa': 'aspirin',
            'nsaids': 'nonsteroidal anti-inflammatory drugs',
            # Day 7: Additional abbreviations
            'iv': 'intravenous',
            'im': 'intramuscular',
            'po': 'oral',
            'stat': 'immediately',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'qd': 'once daily',
            'qod': 'every other day',
            'prn': 'as needed',
            'mg': 'milligram',
            'g': 'gram',
            'ml': 'milliliter',
            'kg': 'kilogram'
        }
    
    def _init_medical_synonyms(self):
        """Initialize medical synonym groups."""
        self.synonym_groups = {
            'heart_attack': ['myocardial infarction', 'mi', 'ami', 'heart attack', 'cardiac arrest'],
            'chest_pain': ['chest pain', 'angina', 'chest discomfort', 'precordial pain', 'retrosternal pain', 'pressure-like pain', 'crushing pain'],
            'chest_pain_radiation': ['left arm pain', 'jaw pain', 'radiation to arm', 'radiates to back', 'radiates to shoulder'],
            'shortness_of_breath': ['dyspnea', 'shortness of breath', 'sob', 'breathlessness', 'difficulty breathing'],
            'high_blood_pressure': ['hypertension', 'htn', 'high blood pressure', 'elevated blood pressure'],
            'diabetes': ['diabetes mellitus', 'dm', 'diabetes', 'diabetes type 2', 'dm2'],
            'stroke': ['cerebrovascular accident', 'cva', 'stroke', 'brain attack'],
            'stroke_deficit': ['hemiparesis', 'unilateral weakness', 'facial droop', 'aphasia', 'dysarthria'],
            'ecg_features': ['st elevation', 'st depression', 't wave inversion', 'new lbbb', 'q waves'],
            'kidney_failure': ['renal failure', 'kidney failure', 'aki', 'acute kidney injury'],
            'infection': ['infection', 'sepsis', 'bacteremia', 'systemic infection'],
            'acute_limb_ischemia': ['pale cold limb', 'pulseless limb', 'limb ischemia'],
            'neutropenic_fever': ['fever with neutropenia', 'low neutrophils fever', 'oncology fever'],
            # Day 7: Additional synonym groups
            'treatment': ['treatment', 'therapy', 'management', 'intervention', 'care'],
            'medication': ['medication', 'drug', 'medicine', 'pharmaceutical'],
            'dose': ['dose', 'dosage', 'amount', 'quantity'],
            'duration': ['duration', 'length', 'period', 'course', 'days'],
            'antibiotic': ['antibiotic', 'antimicrobial', 'antibacterial', 'anti-infective'],
            'pain': ['pain', 'ache', 'discomfort', 'soreness'],
            'fever': ['fever', 'pyrexia', 'elevated temperature', 'hyperthermia'],
            'bleeding': ['bleeding', 'hemorrhage', 'blood loss', 'hemorrhaging'],
            # Nursing / bedside terms
            'oxygen': ['nasal cannula', 'face mask oxygen', 'supplemental oxygen'],
            'iv_access': ['iv line', 'intravenous access', 'drip', 'cannula']
        }

        # Domain-specific expansion map used for query rewriting
        self.expansion_map = {
            'retrosternal pain': ['central chest pain', 'pressure-like chest pain', 'crushing chest pain'],
            'crushing pain': ['pressure-like pain', 'tight chest pain'],
            'left arm radiation': ['radiation to upper limb', 'pain radiating to arm', 'jaw radiation'],
            'hemiparesis': ['unilateral weakness', 'one-sided weakness'],
            'ecg': ['electrocardiogram', 'ekg'],
            'stemi': ['st elevation mi', 'st-segment elevation myocardial infarction'],
            'nstemi': ['non st elevation mi', 'non-st-segment elevation myocardial infarction'],
            'troponin': ['cardiac enzyme', 'cardiac biomarker'],
            'tpa': ['thrombolysis', 'alteplase'],
            'thrombolysis': ['fibrinolysis', 'alteplase'],
            'weakness': ['paresis', 'loss of power'],
            'slurred speech': ['dysarthria'],
            'facial droop': ['facial asymmetry'],
            'nitroglycerin': ['glyceryl trinitrate', 'gtn'],
            'asa': ['aspirin'],
            'acute limb ischemia': ['pulseless limb', 'cold limb'],
            'neutropenic fever': ['oncology fever', 'fever with low neutrophils']
        }
    
    def _init_specialty_keywords(self):
        """Initialize keywords that suggest medical specialties."""
        self.specialty_keywords = {
            'cardiology': ['heart', 'cardiac', 'myocardial', 'coronary', 'troponin', 'ecg', 'ekg', 'chest pain', 'mi', 'chf', 'angina'],
            'neurology': ['stroke', 'cva', 'headache', 'seizure', 'neurological', 'brain', 'cns', 'meningitis', 'epilepsy', 'stiff neck'],
            'endocrinology': ['diabetes', 'dm', 'glucose', 'insulin', 'thyroid', 'hormone', 'a1c', 'hemoglobin a1c', 'hypoglycemia', 'hypoglycaemia'],
            'gastroenterology': ['liver', 'hepatic', 'abdomen', 'abdominal', 'gi', 'gastrointestinal', 'hepatitis', 'abscess'],
            'nephrology': ['kidney', 'renal', 'creatinine', 'bun', 'dialysis', 'aki', 'ckd', 'glomerulonephritis'],
            'pulmonology': ['lung', 'pulmonary', 'copd', 'asthma', 'pneumonia', 'respiratory', 'dyspnea'],
            'infectious_disease': ['infection', 'sepsis', 'bacteremia', 'antibiotic', 'fever', 'culture', 'sti'],
            'pediatrics': ['child', 'pediatric', 'neonate', 'newborn', 'infant', 'adolescent', 'sick newborn', 'neonatal', 'baby', 'hypoglycaemia', 'hypoglycemia', 'lethargy', 'vomiting', 'seizures', 'hypotonia', 'poor feeding'],
            'obstetrics': ['pregnant', 'pregnancy', 'obstetric', 'labor', 'delivery', 'gestation', 'gynecology'],
            # Day 7: Additional specialties with expanded keywords
            'dermatology': ['skin', 'rash', 'dermatitis', 'cellulitis', 'abscess', 'boil', 'furuncle', 'carbuncle', 'fluctuant', 'shaving'],
            'toxicology': ['poison', 'toxic', 'overdose', 'ingestion'],
            'surgery': ['surgical', 'operation', 'procedure', 'incision']
        }
    
    def _init_urgency_keywords(self):
        """Initialize keywords indicating urgency level."""
        self.urgency_keywords = {
            'emergency': ['emergency', 'emergent', 'acute', 'critical', 'urgent', 'immediate', 'stat', 'code'],
            'urgent': ['urgent', 'soon', 'prompt', 'asap', 'expedited'],
            'routine': ['routine', 'elective', 'scheduled', 'follow-up', 'maintenance']
        }
    
    def _init_negation_patterns(self):
        """Initialize patterns for detecting negations."""
        self.negation_patterns = [
            r'\bno\s+(\w+)',
            r'\bdenies\s+(\w+)',
            r'\bwithout\s+(\w+)',
            r'\babsence\s+of\s+(\w+)',
            r'\bnegative\s+for\s+(\w+)',
            r'\bnot\s+(\w+)',
            r'\bunremarkable\s+(\w+)'
        ]
    
    def understand(self, query: str) -> QueryUnderstanding:
        """
        Process a medical query and extract understanding.
        
        Args:
            query: Medical query/question
            
        Returns:
            QueryUnderstanding object with extracted information
        """
        # Expand abbreviations
        expanded_query = self._expand_abbreviations(query)
        
        # Extract clinical features
        clinical_features = self._extract_clinical_features(query, expanded_query)
        
        # Identify likely specialty
        likely_specialty = self._identify_specialty(query, expanded_query, clinical_features)
        
        # Determine acuity level
        acuity_level = self._determine_acuity(query, expanded_query)
        
        # Identify query type
        query_type = self._identify_query_type(query, expanded_query)
        
        # Synonym-based expansion for retrieval robustness
        expanded_query = self._expand_with_synonyms(expanded_query, clinical_features)
        
        return QueryUnderstanding(
            original_query=query,
            expanded_query=expanded_query,
            clinical_features=clinical_features,
            likely_specialty=likely_specialty,
            acuity_level=acuity_level,
            query_type=query_type
        )

    def _expand_with_synonyms(self, expanded_query: str, features: ClinicalFeatures) -> str:
        """
        Add domain-specific synonyms/normalizations to the query to improve retrieval.
        """
        terms_to_expand: Set[str] = set()
        for symptom in features.symptoms:
            terms_to_expand.add(symptom.lower())
        for cond in features.conditions:
            terms_to_expand.add(cond.lower())
        for term in features.medical_terms:
            terms_to_expand.add(term.lower())
        for test in features.tests:
            terms_to_expand.add(test.lower())
        
        expansions: List[str] = []
        for term in terms_to_expand:
            if term in self.expansion_map:
                expansions.extend(self.expansion_map[term])
            # Also expand by synonym_groups if present
            for group_syns in self.synonym_groups.values():
                if term in group_syns:
                    expansions.extend(group_syns)
        
        # Deduplicate and keep concise
        expansions = list({e.lower() for e in expansions if e})
        if expansions:
            expanded_query = expanded_query + " " + " ".join(expansions)
        return expanded_query

    def _load_external_umls(self):
        """
        Load additional UMLS-style expansions from data/umls_expansion.json if present.
        Schema:
        {
          "terms": [
            {"canonical": "stemi", "aliases": ["st elevation mi", ...], "categories": ["cardio"]},
            ...
          ]
        }
        """
        path = Path("data/umls_expansion.json")
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("terms", []):
                canonical = entry.get("canonical")
                aliases = entry.get("aliases", [])
                if not canonical:
                    continue
                group_key = canonical.replace(" ", "_")
                # Merge into synonym groups
                merged = list({canonical.lower(), *[a.lower() for a in aliases]})
                self.synonym_groups[group_key] = list(set(self.synonym_groups.get(group_key, []) + merged))
                # Merge into expansion map
                self.expansion_map[canonical.lower()] = list(set(self.expansion_map.get(canonical.lower(), []) + merged))
        except Exception:
            pass
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand medical abbreviations in query."""
        expanded = query.lower()
        
        # Replace abbreviations with full terms
        for abbrev, full_term in self.abbreviations.items():
            # Word boundary matching
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded = re.sub(pattern, full_term, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def _extract_clinical_features(
        self,
        original_query: str,
        expanded_query: str
    ) -> ClinicalFeatures:
        """Extract clinical features from query."""
        query_lower = expanded_query.lower()
        
        # Extract symptoms (common medical symptoms)
        symptoms = self._extract_symptoms(query_lower)
        
        # Extract demographics
        demographics = self._extract_demographics(original_query)
        
        # Extract medical terms
        medical_terms = self._extract_medical_terms(query_lower)
        
        # Extract medications
        medications = self._extract_medications(query_lower)
        
        # Extract tests/procedures
        tests = self._extract_tests(query_lower)
        
        # Extract conditions/diagnoses
        conditions = self._extract_conditions(query_lower)
        
        # Extract urgency keywords
        urgency_keywords = self._extract_urgency_keywords(query_lower)
        
        # Extract specialty hints
        specialty_hints = self._extract_specialty_hints(query_lower)
        
        # Extract negations
        negations = self._extract_negations(original_query)
        
        return ClinicalFeatures(
            symptoms=symptoms,
            demographics=demographics,
            medical_terms=medical_terms,
            medications=medications,
            tests=tests,
            conditions=conditions,
            urgency_keywords=urgency_keywords,
            specialty_hints=specialty_hints,
            negations=negations
        )
    
    def _extract_symptoms(self, query: str) -> List[str]:
        """Extract symptoms from query."""
        common_symptoms = [
            'pain', 'fever', 'cough', 'shortness of breath', 'dyspnea',
            'chest pain', 'headache', 'nausea', 'vomiting', 'diarrhea',
            'fatigue', 'weakness', 'dizziness', 'syncope', 'seizure',
            'rash', 'jaundice', 'edema', 'swelling', 'bleeding'
        ]
        
        found_symptoms = []
        for symptom in common_symptoms:
            if symptom in query:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def _extract_demographics(self, query: str) -> Dict[str, Optional[str]]:
        """Extract patient demographics."""
        demographics = {
            'age': None,
            'gender': None,
            'age_group': None
        }
        
        # Extract age
        age_patterns = [
            r'(\d+)[-\s]year[-\s]old',
            r'age\s+(\d+)',
            r'(\d+)[-\s]yo',
            r'(\d+)[-\s]y\.o\.'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                demographics['age'] = age
                if age < 18:
                    demographics['age_group'] = 'pediatric'
                elif age < 65:
                    demographics['age_group'] = 'adult'
                else:
                    demographics['age_group'] = 'geriatric'
                break
        
        # Extract gender
        if re.search(r'\bmale\b', query, re.IGNORECASE):
            demographics['gender'] = 'male'
        elif re.search(r'\bfemale\b|\bwoman\b|\bpregnant\b', query, re.IGNORECASE):
            demographics['gender'] = 'female'
        
        return demographics
    
    def _extract_medical_terms(self, query: str) -> List[str]:
        """Extract medical terminology."""
        # Common medical terms
        medical_terms = []
        
        # Check for medical conditions
        for condition_group in self.synonym_groups.values():
            for term in condition_group:
                if term in query:
                    medical_terms.append(term)
                    break
        
        # Check abbreviations
        for abbrev in self.abbreviations.keys():
            if r'\b' + abbrev + r'\b' in query:
                medical_terms.append(abbrev)
        
        return list(set(medical_terms))
    
    def _extract_medications(self, query: str) -> List[str]:
        """Extract medication names."""
        # Day 7: Expanded medication list
        common_medications = [
            'aspirin', 'metformin', 'insulin', 'morphine', 'nitroglycerin',
            'metoprolol', 'lisinopril', 'atorvastatin', 'warfarin', 'heparin',
            'antibiotic', 'anticoagulant', 'ace inhibitor', 'beta blocker',
            'metronidazole', 'ceftriaxone', 'azithromycin', 'doxycycline',
            'ciprofloxacin', 'amoxicillin', 'penicillin', 'furosemide',
            'gentamicin', 'ampicillin', 'cloxacillin', 'cefotaxime',
            'nitrofurantoin', 'trimethoprim', 'sulfamethoxazole', 'cephalexin',
            'clindamycin', 'vancomycin', 'labetalol', 'nifedipine',
            'magnesium sulfate', 'oxytocin', 'methylergonovine',
            'acetaminophen', 'ibuprofen', 'paracetamol', 'diclofenac'
        ]
        
        found_meds = []
        # Sort by length (longer first) to match "magnesium sulfate" before "magnesium"
        sorted_meds = sorted(common_medications, key=len, reverse=True)
        for med in sorted_meds:
            if med in query.lower():
                found_meds.append(med)
                # Don't add shorter matches if longer one found
                break
        
        return found_meds
    
    def _extract_tests(self, query: str) -> List[str]:
        """Extract test/procedure names."""
        common_tests = [
            'ecg', 'ekg', 'electrocardiogram', 'ct', 'mri', 'x-ray', 'ultrasound',
            'troponin', 'creatinine', 'glucose', 'hemoglobin', 'a1c',
            'blood test', 'lab test', 'culture', 'biopsy'
        ]
        
        found_tests = []
        for test in common_tests:
            if test in query:
                found_tests.append(test)
        
        return found_tests
    
    def _extract_conditions(self, query: str) -> List[str]:
        """Extract medical conditions/diagnoses."""
        conditions = []
        
        # Use synonym groups to identify conditions
        for group_name, synonyms in self.synonym_groups.items():
            for synonym in synonyms:
                if synonym in query:
                    conditions.append(group_name)
                    break
        
        return conditions
    
    def _extract_urgency_keywords(self, query: str) -> List[str]:
        """Extract urgency-related keywords."""
        found_keywords = []
        
        for urgency_level, keywords in self.urgency_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_specialty_hints(self, query: str) -> List[str]:
        """Extract specialty-related keywords."""
        found_hints = []
        
        for specialty, keywords in self.specialty_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    found_hints.append(specialty)
                    break
        
        return list(set(found_hints))
    
    def _extract_negations(self, query: str) -> List[str]:
        """Extract negated findings."""
        negations = []
        
        for pattern in self.negation_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                negated_term = match.group(1) if match.lastindex else match.group(0)
                negations.append(negated_term)
        
        return negations
    
    def _identify_specialty(
        self,
        query: str,
        expanded_query: str,
        features: ClinicalFeatures
    ) -> Optional[str]:
        """Identify likely medical specialty."""
        specialty_scores = defaultdict(int)
        
        # Score based on specialty hints
        for hint in features.specialty_hints:
            specialty_scores[hint] += 2
        
        # Score based on conditions
        condition_specialty_map = {
            'heart_attack': 'cardiology',
            'chest_pain': 'cardiology',
            'diabetes': 'endocrinology',
            'stroke': 'neurology',
            'kidney_failure': 'nephrology',
            'infection': 'infectious_disease'
        }
        
        for condition in features.conditions:
            if condition in condition_specialty_map:
                specialty_scores[condition_specialty_map[condition]] += 1
        
        # Score based on tests
        test_specialty_map = {
            'troponin': 'cardiology',
            'ecg': 'cardiology',
            'ekg': 'cardiology',
            'glucose': 'endocrinology',
            'a1c': 'endocrinology',
            'creatinine': 'nephrology'
        }
        
        for test in features.tests:
            if test in test_specialty_map:
                specialty_scores[test_specialty_map[test]] += 1
        
        # Return top specialty
        if specialty_scores:
            return max(specialty_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _determine_acuity(self, query: str, expanded_query: str) -> str:
        """Determine acuity level."""
        query_lower = expanded_query.lower()
        
        # Check for emergency keywords
        for keyword in self.urgency_keywords['emergency']:
            if keyword in query_lower:
                return 'emergency'
        
        # Check for urgent keywords
        for keyword in self.urgency_keywords['urgent']:
            if keyword in query_lower:
                return 'urgent'
        
        # Default to routine
        return 'routine'
    
    def _identify_query_type(self, query: str, expanded_query: str) -> str:
        """Identify type of query."""
        query_lower = expanded_query.lower()
        
        if any(word in query_lower for word in ['treatment', 'treat', 'therapy', 'medication', 'drug']):
            return 'treatment'
        elif any(word in query_lower for word in ['diagnosis', 'diagnose', 'diagnostic', 'test', 'order']):
            return 'test_ordering'
        elif any(word in query_lower for word in ['manage', 'management', 'care', 'protocol']):
            return 'management'
        else:
            return 'diagnosis'


def main():
    """Demo: Test query understanding."""
    print("="*70)
    print("MEDICAL QUERY UNDERSTANDING DEMO")
    print("="*70)
    
    understanding_module = MedicalQueryUnderstanding()
    
    test_queries = [
        "What is the treatment for acute MI in a 65-year-old male?",
        "Patient with elevated troponin and chest pain needs emergency care",
        "How to manage diabetes type 2 with metformin?",
        "No fever, no cough, but shortness of breath"
    ]
    
    for query in test_queries:
        print(f"\n{'-'*70}")
        print(f"Query: {query}")
        print(f"{'-'*70}")
        
        understanding = understanding_module.understand(query)
        
        print(f"\nExpanded Query: {understanding.expanded_query}")
        print(f"\nClinical Features:")
        print(f"  Symptoms: {understanding.clinical_features.symptoms}")
        print(f"  Demographics: {understanding.clinical_features.demographics}")
        print(f"  Medical Terms: {understanding.clinical_features.medical_terms}")
        print(f"  Medications: {understanding.clinical_features.medications}")
        print(f"  Tests: {understanding.clinical_features.tests}")
        print(f"  Conditions: {understanding.clinical_features.conditions}")
        print(f"  Negations: {understanding.clinical_features.negations}")
        print(f"\nSpecialty: {understanding.likely_specialty}")
        print(f"Acuity Level: {understanding.acuity_level}")
        print(f"Query Type: {understanding.query_type}")


if __name__ == "__main__":
    main()


