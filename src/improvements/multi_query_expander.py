"""
Multi-Query Expansion for Medical QA

Fix 1: Generate 4-5 rephrased queries focusing on different aspects:
- Keywords only
- Major symptoms
- Likely diagnosis
- Guideline terminology
- Patient features

Then retrieve for all queries → merge → rerank.

Expected improvement:
- MAP improves to 0.20–0.30
- P@5 improves to 0.20–0.25
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ExpandedQueries:
    """Result of multi-query expansion."""
    original_query: str
    keyword_query: str
    symptom_query: str
    diagnosis_query: str
    guideline_query: str
    patient_feature_query: str
    all_queries: List[str]


class MultiQueryExpander:
    """
    Generate multiple rephrased queries for better retrieval coverage.
    
    Uses Ollama LLM to generate focused alternative queries.
    """
    
    def __init__(self, llm_model=None):
        """Initialize with optional LLM model."""
        self.llm_model = llm_model
        if llm_model is None:
            try:
                from models.ollama_model import OllamaModel
                self.llm_model = OllamaModel(
                    model_name="llama3.1:8b",
                    temperature=0.3,  # Slightly higher for variety
                    max_tokens=256
                )
            except Exception as e:
                print(f"[WARN] Could not initialize LLM for multi-query expansion: {e}")
                self.llm_model = None
        
        # Fallback keyword extractors
        self._init_medical_keywords()
    
    def _init_medical_keywords(self):
        """Initialize medical keyword patterns for rule-based fallback."""
        self.symptom_patterns = [
            r'\b(pain|ache|fever|cough|dyspnea|breathlessness|nausea|vomiting|diarrhea|'
            r'bleeding|swelling|rash|fatigue|weakness|dizziness|confusion|seizure|syncope|'
            r'headache|chest pain|abdominal pain|shortness of breath|palpitations|'
            r'hematemesis|melena|hematuria|hemoptysis|dysphagia|dysuria|oliguria|anuria|'
            r'tachycardia|bradycardia|hypotension|hypertension|hypoxia|cyanosis|jaundice|'
            r'edema|ascites|lethargy|altered mental status|unconscious|coma)\b'
        ]
        
        self.diagnosis_patterns = [
            r'\b(pneumonia|sepsis|meningitis|diabetes|hypertension|hypotension|'
            r'myocardial infarction|heart failure|stroke|pulmonary embolism|'
            r'preeclampsia|eclampsia|anemia|infection|shock|arrhythmia|'
            r'tuberculosis|malaria|typhoid|cholera|dengue|hepatitis|hiv|aids|'
            r'asthma|copd|ckd|aki|dvt|pe|acs|stemi|nstemi|uti|cellulitis|'
            r'gastroenteritis|dehydration|hypoglycemia|hyperglycemia|dka|'
            r'neonatal sepsis|respiratory distress|jaundice|prematurity)\b'
        ]
        
        self.treatment_keywords = [
            'treatment', 'therapy', 'management', 'medication', 'drug',
            'antibiotic', 'antihypertensive', 'analgesic', 'first-line',
            'recommended', 'initial', 'emergency', 'intervention',
            'prophylaxis', 'empiric', 'definitive', 'supportive'
        ]
        
        self.guideline_keywords = [
            'guideline', 'protocol', 'recommendation', 'standard of care',
            'best practice', 'evidence-based', 'clinical pathway'
        ]
        
        # FIX 3: Add comprehensive acronym mappings
        self.acronym_expansions = {
            'mi': 'myocardial infarction heart attack',
            'stemi': 'st elevation myocardial infarction',
            'nstemi': 'non st elevation myocardial infarction',
            'acs': 'acute coronary syndrome',
            'chf': 'congestive heart failure',
            'hf': 'heart failure',
            'cva': 'cerebrovascular accident stroke',
            'tia': 'transient ischemic attack',
            'pe': 'pulmonary embolism',
            'dvt': 'deep vein thrombosis',
            'copd': 'chronic obstructive pulmonary disease',
            'ards': 'acute respiratory distress syndrome',
            'rds': 'respiratory distress syndrome',
            'aki': 'acute kidney injury',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'uti': 'urinary tract infection',
            'cap': 'community acquired pneumonia',
            'hap': 'hospital acquired pneumonia',
            'vap': 'ventilator associated pneumonia',
            'sirs': 'systemic inflammatory response syndrome',
            'dm': 'diabetes mellitus',
            't2dm': 'type 2 diabetes mellitus',
            'dka': 'diabetic ketoacidosis',
            'htn': 'hypertension high blood pressure',
            'afib': 'atrial fibrillation',
            'svt': 'supraventricular tachycardia',
            'vt': 'ventricular tachycardia',
            'vfib': 'ventricular fibrillation',
            'pih': 'pregnancy induced hypertension',
            'pph': 'postpartum hemorrhage',
            'gdm': 'gestational diabetes mellitus',
            'eos': 'early onset sepsis',
            'los': 'late onset sepsis',
            'hie': 'hypoxic ischemic encephalopathy',
            'nec': 'necrotizing enterocolitis',
            'sob': 'shortness of breath dyspnea',
            'cp': 'chest pain',
            'ams': 'altered mental status',
            'loc': 'loss of consciousness',
            'gcs': 'glasgow coma scale',
            'cbc': 'complete blood count',
            'lft': 'liver function test',
            'rft': 'renal function test',
            'abg': 'arterial blood gas',
            'ecg': 'electrocardiogram',
            'ekg': 'electrocardiogram',
            'cxr': 'chest xray',
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'usg': 'ultrasound',
            'iv': 'intravenous',
            'po': 'per oral by mouth',
            'im': 'intramuscular',
            'prn': 'as needed',
            'stat': 'immediately',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'hs': 'at bedtime',
            'ac': 'before meals',
            'pc': 'after meals'
        }
        
        # Common spelling variations
        self.spelling_variations = {
            'diarrhea': ['diarrhoea', 'loose stools', 'watery stool'],
            'anemia': ['anaemia', 'low hemoglobin', 'low hb'],
            'edema': ['oedema', 'swelling', 'fluid retention'],
            'hemorrhage': ['haemorrhage', 'bleeding', 'blood loss'],
            'fetus': ['foetus', 'baby', 'unborn child'],
            'pediatric': ['paediatric', 'child', 'children'],
            'cesarean': ['caesarean', 'c-section', 'surgical delivery'],
            'esophagus': ['oesophagus', 'food pipe', 'gullet'],
            'hemoglobin': ['haemoglobin', 'hb', 'hgb'],
            'leukocyte': ['leucocyte', 'white blood cell', 'wbc'],
            'septicemia': ['septicaemia', 'blood poisoning', 'sepsis'],
            'hypoglycemia': ['hypoglycaemia', 'low blood sugar', 'low glucose'],
            'hyperglycemia': ['hyperglycaemia', 'high blood sugar', 'high glucose']
        }
    
    def expand(
        self,
        question: str,
        case_description: str,
        use_llm: bool = True
    ) -> ExpandedQueries:
        """
        Generate multiple query variants for better retrieval.
        
        Args:
            question: The clinical question
            case_description: Patient case description
            use_llm: Whether to use LLM for expansion (fallback to rules if False)
            
        Returns:
            ExpandedQueries with all generated variants
        """
        full_query = f"{case_description} {question}"
        
        if use_llm and self.llm_model:
            try:
                return self._llm_expand(question, case_description)
            except Exception as e:
                print(f"[WARN] LLM expansion failed, using rule-based: {e}")
        
        # Rule-based fallback
        return self._rule_based_expand(question, case_description)
    
    def _llm_expand(self, question: str, case_description: str) -> ExpandedQueries:
        """Use LLM to generate query variants."""
        prompt = f"""You are a medical query expansion assistant. Given a clinical case and question, generate 4 alternative search queries to find relevant medical guidelines.

Clinical Case:
{case_description}

Question: {question}

Generate exactly 4 alternative queries, each on a new line, focusing on:
1. KEYWORDS: Extract only the key medical terms (medications, conditions, procedures)
2. SYMPTOMS: Focus on the patient's symptoms and clinical presentation
3. DIAGNOSIS: Focus on the likely diagnosis or differential diagnoses
4. GUIDELINE: Rephrase as a guideline-seeking query (e.g., "treatment protocol for X")

Output format (exactly 4 lines, no numbering, no labels):
[keyword query]
[symptom query]
[diagnosis query]
[guideline query]

Important: Each query should be concise (10-20 words) and search-optimized."""

        response = self.llm_model.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=256
        )
        
        # Parse response
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        
        # Filter out any lines that look like labels
        queries = []
        for line in lines:
            # Skip lines that are just labels
            if line.lower().startswith(('keywords:', 'symptoms:', 'diagnosis:', 'guideline:', '1.', '2.', '3.', '4.')):
                # Try to extract the actual query after the label
                parts = line.split(':', 1)
                if len(parts) > 1:
                    queries.append(parts[1].strip())
            else:
                queries.append(line)
        
        # Ensure we have at least 4 queries
        while len(queries) < 4:
            queries.append(f"{case_description} {question}")
        
        # Also extract patient features
        patient_features = self._extract_patient_features(case_description)
        patient_feature_query = f"{patient_features} {question}" if patient_features else queries[0]
        
        original = f"{case_description} {question}"
        
        return ExpandedQueries(
            original_query=original,
            keyword_query=queries[0][:500],  # Limit length
            symptom_query=queries[1][:500],
            diagnosis_query=queries[2][:500],
            guideline_query=queries[3][:500],
            patient_feature_query=patient_feature_query[:500],
            all_queries=[original, queries[0], queries[1], queries[2], queries[3], patient_feature_query]
        )
    
    def _rule_based_expand(self, question: str, case_description: str) -> ExpandedQueries:
        """Rule-based query expansion fallback with acronym and variation expansion."""
        full_text = f"{case_description} {question}".lower()
        
        # FIX 3: Expand acronyms in the text first
        expanded_text = self._expand_acronyms(full_text)
        
        # Extract keywords (now from expanded text)
        keywords = self._extract_keywords(expanded_text)
        keyword_query = " ".join(keywords) if keywords else expanded_text
        
        # Extract symptoms with variations
        symptoms = self._extract_symptoms(expanded_text)
        symptom_query = f"patient with {' '.join(symptoms)} treatment management" if symptoms else expanded_text
        
        # Extract diagnoses with variations
        diagnoses = self._extract_diagnoses(expanded_text)
        diagnosis_query = f"{' '.join(diagnoses)} management guidelines treatment protocol" if diagnoses else expanded_text
        
        # Create guideline query
        guideline_query = self._create_guideline_query(question, case_description)
        
        # Patient features
        patient_features = self._extract_patient_features(case_description)
        patient_feature_query = f"{patient_features} {question}" if patient_features else expanded_text
        
        # FIX 3: Add acronym-expanded query as additional variant
        acronym_query = self._create_acronym_query(full_text)
        
        original = f"{case_description} {question}"
        
        # Include more query variants for better coverage
        all_queries = [
            original,
            keyword_query,
            symptom_query,
            diagnosis_query,
            guideline_query,
            patient_feature_query
        ]
        
        # Add acronym query if different
        if acronym_query and acronym_query != original:
            all_queries.append(acronym_query)
        
        return ExpandedQueries(
            original_query=original,
            keyword_query=keyword_query,
            symptom_query=symptom_query,
            diagnosis_query=diagnosis_query,
            guideline_query=guideline_query,
            patient_feature_query=patient_feature_query,
            all_queries=all_queries
        )
    
    def _expand_acronyms(self, text: str) -> str:
        """Expand medical acronyms in text."""
        expanded = text
        for acronym, expansion in self.acronym_expansions.items():
            # Match word boundaries
            pattern = r'\b' + re.escape(acronym) + r'\b'
            if re.search(pattern, expanded, re.IGNORECASE):
                expanded = re.sub(pattern, f"{acronym} {expansion}", expanded, flags=re.IGNORECASE)
        return expanded
    
    def _create_acronym_query(self, text: str) -> str:
        """Create a query with expanded acronyms and spelling variations."""
        expanded = self._expand_acronyms(text)
        
        # Also add spelling variations
        for word, variations in self.spelling_variations.items():
            if word in expanded:
                expanded += " " + " ".join(variations[:2])  # Add first 2 variations
        
        return expanded[:1000]  # Limit length
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract medical keywords from text."""
        keywords = []
        
        # Medical terms (longer words are more specific)
        words = re.findall(r'\b[a-z]{5,}\b', text.lower())
        
        # Filter to medical-sounding terms
        medical_suffixes = ['itis', 'osis', 'emia', 'pathy', 'ectomy', 'plasty', 'gram', 
                          'scope', 'mycin', 'cillin', 'azole', 'pril', 'sartan', 'olol']
        
        for word in words:
            if any(word.endswith(suffix) for suffix in medical_suffixes):
                keywords.append(word)
            elif len(word) > 6 and word not in ['patient', 'present', 'history', 'physical']:
                keywords.append(word)
        
        # Limit to top 10
        return list(set(keywords))[:10]
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text."""
        symptoms = []
        for pattern in self.symptom_patterns:
            matches = re.findall(pattern, text.lower(), re.IGNORECASE)
            symptoms.extend(matches)
        return list(set(symptoms))[:5]
    
    def _extract_diagnoses(self, text: str) -> List[str]:
        """Extract diagnoses from text."""
        diagnoses = []
        for pattern in self.diagnosis_patterns:
            matches = re.findall(pattern, text.lower(), re.IGNORECASE)
            diagnoses.extend(matches)
        return list(set(diagnoses))[:3]
    
    def _extract_patient_features(self, case_description: str) -> str:
        """Extract patient demographics and key features."""
        features = []
        
        # Age
        age_match = re.search(r'(\d+)[- ]?(year|yr|month|mo|week|wk|day)[- ]?old', case_description.lower())
        if age_match:
            features.append(age_match.group(0))
        
        # Gender
        if 'female' in case_description.lower() or 'woman' in case_description.lower():
            features.append('female')
        elif 'male' in case_description.lower() or 'man' in case_description.lower():
            features.append('male')
        
        # Pregnancy
        if 'pregnant' in case_description.lower() or 'pregnancy' in case_description.lower():
            features.append('pregnant')
        
        # Key conditions
        conditions = ['diabetic', 'hypertensive', 'immunocompromised', 'hiv', 'cancer']
        for cond in conditions:
            if cond in case_description.lower():
                features.append(cond)
        
        return " ".join(features)
    
    def _create_guideline_query(self, question: str, case_description: str) -> str:
        """Create a guideline-focused query."""
        # Extract the main topic
        question_lower = question.lower()
        
        # Check for treatment questions
        if any(word in question_lower for word in ['treatment', 'treat', 'manage', 'therapy']):
            # Extract condition
            conditions = self._extract_diagnoses(f"{case_description} {question}")
            if conditions:
                return f"{conditions[0]} treatment guidelines protocol recommended"
        
        # Check for diagnosis questions
        if any(word in question_lower for word in ['diagnosis', 'diagnose', 'identify', 'cause']):
            symptoms = self._extract_symptoms(case_description)
            if symptoms:
                return f"diagnosis differential {' '.join(symptoms[:3])} clinical guidelines"
        
        # Default: add guideline keywords
        return f"{question} clinical guidelines recommendations"


def merge_and_deduplicate_results(
    all_results: List[Tuple[any, float]],
    top_k: int = 25
) -> List[Tuple[any, float]]:
    """
    Merge results from multiple queries and deduplicate.
    
    Uses reciprocal rank fusion for better ranking.
    
    Args:
        all_results: List of (document, score) tuples from multiple queries
        top_k: Number of results to return
        
    Returns:
        Merged and deduplicated results
    """
    # Track unique documents and their aggregated scores
    doc_scores = {}
    doc_objects = {}
    
    k = 60  # RRF constant
    
    for rank, (doc, score) in enumerate(all_results):
        # Create unique key for document
        if hasattr(doc, 'document'):
            # RetrievalResult object
            actual_doc = doc.document
            doc_key = (
                actual_doc.metadata.get('guideline_id', ''),
                actual_doc.metadata.get('chunk_index', 0)
            )
        elif hasattr(doc, 'metadata'):
            # Document object
            doc_key = (
                doc.metadata.get('guideline_id', ''),
                doc.metadata.get('chunk_index', 0)
            )
        else:
            doc_key = str(doc)[:100]  # Fallback
        
        # Reciprocal Rank Fusion score
        rrf_score = 1 / (k + rank + 1)
        
        if doc_key in doc_scores:
            # Aggregate: RRF + original score weighted
            doc_scores[doc_key] += rrf_score + score * 0.3
        else:
            doc_scores[doc_key] = rrf_score + score * 0.3
            doc_objects[doc_key] = doc
    
    # Sort by aggregated score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top-k
    results = []
    for doc_key, score in sorted_docs[:top_k]:
        doc = doc_objects[doc_key]
        # Normalize score to 0-1 range
        normalized_score = min(1.0, score / 2.0)
        results.append((doc, normalized_score))
    
    return results
