"""
Medical Terminology Normalization Module

Normalizes medical terms before querying retrieval:
- Abbreviations (UTI, COPD, DKA, HHS)
- Pediatric terms
- Pregnancy terms
- Measurement units
- Medical synonyms

Uses medical synonym map + UMLS synonyms.
"""

from typing import Dict, List, Optional, Set, Tuple
import re
import json
from pathlib import Path


class TerminologyNormalizer:
    """
    Normalizes medical terminology for improved retrieval.
    
    Handles:
    - Abbreviation expansion
    - Synonym replacement
    - Unit standardization
    - Age-specific term normalization
    """
    
    def __init__(self, umls_path: Optional[str] = None):
        """Initialize with UMLS synonyms."""
        self.umls_path = umls_path or "data/umls_synonyms.json"
        self._load_umls_synonyms()
        self._init_abbreviations()
        self._init_unit_conversions()
        self._init_specialty_terms()
    
    def _load_umls_synonyms(self):
        """Load UMLS synonym mappings."""
        self.umls_synonyms = {}
        try:
            path = Path(self.umls_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Skip metadata key
                    for key, value in data.items():
                        if key != '_metadata' and isinstance(value, list):
                            self.umls_synonyms[key.lower()] = [v.lower() for v in value]
        except Exception as e:
            print(f"[WARN] Could not load UMLS synonyms: {e}")
    
    def _init_abbreviations(self):
        """Initialize medical abbreviation expansions."""
        self.abbreviations = {
            # Common conditions
            'uti': 'urinary tract infection',
            'copd': 'chronic obstructive pulmonary disease',
            'dka': 'diabetic ketoacidosis',
            'hhs': 'hyperosmolar hyperglycemic state',
            'mi': 'myocardial infarction',
            'stemi': 'st elevation myocardial infarction',
            'nstemi': 'non st elevation myocardial infarction',
            'acs': 'acute coronary syndrome',
            'chf': 'congestive heart failure',
            'hf': 'heart failure',
            'cva': 'cerebrovascular accident stroke',
            'tia': 'transient ischemic attack',
            'pe': 'pulmonary embolism',
            'dvt': 'deep vein thrombosis',
            'vte': 'venous thromboembolism',
            'ards': 'acute respiratory distress syndrome',
            'rds': 'respiratory distress syndrome',
            'aki': 'acute kidney injury',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'cap': 'community acquired pneumonia',
            'hap': 'hospital acquired pneumonia',
            'vap': 'ventilator associated pneumonia',
            'sirs': 'systemic inflammatory response syndrome',
            'dm': 'diabetes mellitus',
            't2dm': 'type 2 diabetes mellitus',
            't1dm': 'type 1 diabetes mellitus',
            'htn': 'hypertension',
            'afib': 'atrial fibrillation',
            'svt': 'supraventricular tachycardia',
            'vt': 'ventricular tachycardia',
            'vfib': 'ventricular fibrillation',
            'gerd': 'gastroesophageal reflux disease',
            'ibs': 'irritable bowel syndrome',
            'ibd': 'inflammatory bowel disease',
            'tb': 'tuberculosis',
            'hiv': 'human immunodeficiency virus',
            'aids': 'acquired immunodeficiency syndrome',
            
            # OB/GYN
            'pih': 'pregnancy induced hypertension',
            'pph': 'postpartum hemorrhage',
            'gdm': 'gestational diabetes mellitus',
            'pprom': 'preterm premature rupture of membranes',
            'prom': 'premature rupture of membranes',
            'iugr': 'intrauterine growth restriction',
            'iufd': 'intrauterine fetal death',
            
            # Pediatric/Neonatal
            'eos': 'early onset sepsis',
            'los': 'late onset sepsis',
            'hie': 'hypoxic ischemic encephalopathy',
            'nec': 'necrotizing enterocolitis',
            'bpd': 'bronchopulmonary dysplasia',
            'rop': 'retinopathy of prematurity',
            'ivh': 'intraventricular hemorrhage',
            'pda': 'patent ductus arteriosus',
            
            # Symptoms
            'sob': 'shortness of breath dyspnea',
            'cp': 'chest pain',
            'ams': 'altered mental status',
            'loc': 'loss of consciousness',
            'ha': 'headache',
            'n/v': 'nausea and vomiting',
            'brbpr': 'bright red blood per rectum',
            
            # Labs
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'cmp': 'comprehensive metabolic panel',
            'lft': 'liver function test',
            'rft': 'renal function test',
            'abg': 'arterial blood gas',
            'bnp': 'brain natriuretic peptide',
            'crp': 'c reactive protein',
            'esr': 'erythrocyte sedimentation rate',
            'pt': 'prothrombin time',
            'ptt': 'partial thromboplastin time',
            'inr': 'international normalized ratio',
            
            # Imaging
            'cxr': 'chest xray',
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'usg': 'ultrasound',
            'ecg': 'electrocardiogram',
            'ekg': 'electrocardiogram',
            'echo': 'echocardiogram',
            
            # Routes/Timing
            'iv': 'intravenous',
            'im': 'intramuscular',
            'po': 'per oral by mouth',
            'pr': 'per rectum',
            'sc': 'subcutaneous',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'prn': 'as needed',
            'stat': 'immediately'
        }
    
    def _init_unit_conversions(self):
        """Initialize unit standardization."""
        self.unit_conversions = {
            'c': 'celsius',
            'f': 'fahrenheit',
            'kg': 'kilogram',
            'lb': 'pound',
            'mg': 'milligram',
            'g': 'gram',
            'mcg': 'microgram',
            'μg': 'microgram',
            'ml': 'milliliter',
            'l': 'liter',
            'mmhg': 'millimeters of mercury',
            'bpm': 'beats per minute',
            'mmol/l': 'millimoles per liter',
            'mg/dl': 'milligrams per deciliter',
            'meq/l': 'milliequivalents per liter'
        }
    
    def _init_specialty_terms(self):
        """Initialize specialty-specific term mappings."""
        self.pediatric_terms = {
            'newborn': ['neonate', 'neonatal', 'infant 0-28 days'],
            'infant': ['baby', 'young child', 'infant 1-12 months'],
            'toddler': ['young child', 'child 1-3 years'],
            'child': ['pediatric patient', 'juvenile'],
            'adolescent': ['teenager', 'teen', 'young adult']
        }
        
        self.pregnancy_terms = {
            'pregnant': ['gravid', 'gestational', 'prenatal', 'antenatal'],
            'postpartum': ['postnatal', 'after delivery', 'puerperal'],
            'labor': ['labour', 'contractions', 'active labor'],
            'delivery': ['birth', 'childbirth', 'parturition'],
            'trimester': ['first trimester', 'second trimester', 'third trimester'],
            'fetus': ['foetus', 'unborn baby', 'intrauterine']
        }
        
        self.emergency_terms = {
            'emergency': ['urgent', 'critical', 'life-threatening', 'acute'],
            'resuscitation': ['resus', 'cpr', 'life support'],
            'shock': ['hemodynamic instability', 'circulatory failure'],
            'arrest': ['cardiac arrest', 'respiratory arrest', 'code']
        }
    
    def normalize(self, text: str) -> str:
        """
        Normalize medical terminology in text.
        
        Args:
            text: Input text with medical terminology
            
        Returns:
            Normalized text with expanded abbreviations and synonyms
        """
        normalized = text.lower()
        
        # Expand abbreviations
        normalized = self._expand_abbreviations(normalized)
        
        # Add synonyms for key terms
        normalized = self._add_synonyms(normalized)
        
        # Standardize units
        normalized = self._standardize_units(normalized)
        
        # Expand specialty terms
        normalized = self._expand_specialty_terms(normalized)
        
        return normalized
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations."""
        result = text
        for abbrev, expansion in self.abbreviations.items():
            # Match whole words only
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(
                    pattern,
                    f"{abbrev} {expansion}",
                    result,
                    flags=re.IGNORECASE
                )
        return result
    
    def _add_synonyms(self, text: str) -> str:
        """Add synonyms for medical terms."""
        result = text
        for term, synonyms in self.umls_synonyms.items():
            if term in result:
                # Add first 2 synonyms
                syn_text = ' '.join(synonyms[:2])
                result = result.replace(term, f"{term} {syn_text}")
        return result
    
    def _standardize_units(self, text: str) -> str:
        """Standardize measurement units."""
        result = text
        for unit, standard in self.unit_conversions.items():
            pattern = r'(\d+)\s*' + re.escape(unit) + r'\b'
            result = re.sub(
                pattern,
                r'\1 ' + standard,
                result,
                flags=re.IGNORECASE
            )
        return result
    
    def _expand_specialty_terms(self, text: str) -> str:
        """Expand specialty-specific terms."""
        result = text
        
        # Pediatric terms
        for term, expansions in self.pediatric_terms.items():
            if term in result:
                expansion_text = ' '.join(expansions[:2])
                result = result.replace(term, f"{term} {expansion_text}")
        
        # Pregnancy terms
        for term, expansions in self.pregnancy_terms.items():
            if term in result:
                expansion_text = ' '.join(expansions[:2])
                result = result.replace(term, f"{term} {expansion_text}")
        
        # Emergency terms
        for term, expansions in self.emergency_terms.items():
            if term in result:
                expansion_text = ' '.join(expansions[:2])
                result = result.replace(term, f"{term} {expansion_text}")
        
        return result
    
    def get_all_synonyms(self, term: str) -> List[str]:
        """Get all synonyms for a medical term."""
        term_lower = term.lower()
        synonyms = set()
        
        # Check UMLS synonyms
        if term_lower in self.umls_synonyms:
            synonyms.update(self.umls_synonyms[term_lower])
        
        # Check abbreviation expansion
        if term_lower in self.abbreviations:
            synonyms.add(self.abbreviations[term_lower])
        
        # Check specialty terms
        for specialty_dict in [self.pediatric_terms, self.pregnancy_terms, self.emergency_terms]:
            if term_lower in specialty_dict:
                synonyms.update(specialty_dict[term_lower])
        
        return list(synonyms)
    
    def extract_key_concepts(self, text: str) -> Dict[str, List[str]]:
        """Extract and categorize key medical concepts from text."""
        concepts = {
            'conditions': [],
            'symptoms': [],
            'treatments': [],
            'demographics': [],
            'labs': [],
            'procedures': []
        }
        
        text_lower = text.lower()
        
        # Find conditions (from abbreviations that map to conditions)
        condition_abbrevs = ['uti', 'copd', 'dka', 'mi', 'stemi', 'chf', 'pe', 'dvt', 'ards', 'aki', 'ckd']
        for abbrev in condition_abbrevs:
            if abbrev in text_lower:
                concepts['conditions'].append(self.abbreviations.get(abbrev, abbrev))
        
        # Find symptoms
        symptom_abbrevs = ['sob', 'cp', 'ams', 'ha', 'n/v']
        for abbrev in symptom_abbrevs:
            if abbrev in text_lower:
                concepts['symptoms'].append(self.abbreviations.get(abbrev, abbrev))
        
        # Find labs
        lab_abbrevs = ['cbc', 'bmp', 'lft', 'rft', 'abg', 'bnp', 'crp']
        for abbrev in lab_abbrevs:
            if abbrev in text_lower:
                concepts['labs'].append(self.abbreviations.get(abbrev, abbrev))
        
        # Find demographic terms
        for term in ['newborn', 'infant', 'child', 'adolescent', 'pregnant', 'elderly']:
            if term in text_lower:
                concepts['demographics'].append(term)
        
        return concepts

