"""
Specialty Adapter Module

Adapts retrieval and reasoning for specific medical specialties:
- Domain detection (identify specialty from query)
- Specialty-specific query expansion
- Domain-adapted reasoning templates
- Specialty-focused retrieval enhancement

Addresses Day 4 Issue: OB/GYN 0%, Dermatology 20%, Gastroenterology 20%
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.document_processor import Document


@dataclass
class SpecialtyAdaptation:
    """Specialty-specific adaptation result."""
    detected_specialty: str
    confidence: float
    adapted_query: str
    specialty_keywords: List[str]
    reasoning_template: str
    retrieval_focus: List[str]


class SpecialtyAdapter:
    """
    Adapts system for medical specialties.
    
    This addresses specialty-specific failures by:
    - Detecting specialty from query
    - Applying specialty-specific query expansion
    - Using domain-adapted reasoning templates
    - Focusing retrieval on specialty-relevant documents
    """
    
    def __init__(self):
        """Initialize specialty adapter."""
        self._init_specialty_keywords()
        self._init_specialty_reasoning_templates()
        self._init_specialty_retrieval_focus()
    
    def _init_specialty_keywords(self):
        """Initialize specialty-specific keywords."""
        self.specialty_keywords = {
            'obstetrics_gynecology': [
                'pregnant', 'pregnancy', 'obstetric', 'gynecologic', 'gynecology',
                'labor', 'delivery', 'gestation', 'fetal', 'maternal', 'prenatal',
                'postpartum', 'menstrual', 'ovarian', 'uterine', 'cervical',
                'vaginal', 'pelvic', 'contraception', 'fertility', 'ob/gyn',
                'obstetrics', 'gynecology', 'preeclampsia', 'eclampsia', 'cesarean',
                'vaginal delivery', 'amniotic', 'placenta', 'umbilical'
            ],
            'dermatology': [
                'skin', 'rash', 'dermatitis', 'eczema', 'psoriasis', 'lesion',
                'dermatologic', 'cutaneous', 'melanoma', 'basal cell', 'squamous',
                'acne', 'urticaria', 'hives', 'pruritus', 'erythema'
            ],
            'gastroenterology': [
                'liver', 'hepatic', 'abdomen', 'abdominal', 'gi', 'gastrointestinal',
                'hepatitis', 'cirrhosis', 'jaundice', 'ascites', 'gastritis',
                'peptic ulcer', 'ibd', 'crohn', 'colitis', 'pancreatitis',
                'gallbladder', 'biliary', 'esophageal', 'gastric'
            ],
            'cardiology': [
                'heart', 'cardiac', 'myocardial', 'coronary', 'cardiovascular',
                'troponin', 'ecg', 'ekg', 'chest pain', 'mi', 'chf', 'angina',
                'arrhythmia', 'atrial fibrillation', 'ventricular'
            ],
            'neurology': [
                'brain', 'neurological', 'cns', 'stroke', 'cva', 'seizure',
                'headache', 'migraine', 'meningitis', 'epilepsy', 'parkinson',
                'alzheimer', 'dementia', 'neuropathy'
            ],
            'endocrinology': [
                'diabetes', 'glucose', 'insulin', 'thyroid', 'hormone', 'a1c',
                'hemoglobin a1c', 'metabolic', 'hypoglycemia', 'hyperglycemia',
                'diabetic', 'endocrine'
            ],
            'nephrology': [
                'kidney', 'renal', 'creatinine', 'bun', 'dialysis', 'aki',
                'ckd', 'nephro', 'glomerular', 'nephritis', 'nephrotic'
            ],
            'pulmonology': [
                'lung', 'pulmonary', 'respiratory', 'copd', 'asthma',
                'pneumonia', 'dyspnea', 'bronchitis', 'emphysema', 'respiratory'
            ],
            'infectious_disease': [
                'infection', 'sepsis', 'bacteremia', 'antibiotic', 'fever',
                'culture', 'bacterial', 'viral', 'fungal', 'pathogen'
            ],
            'pediatrics': [
                'child', 'pediatric', 'neonate', 'newborn', 'infant',
                'adolescent', 'pediatric', 'pediatrician'
            ],
        }
    
    def _init_specialty_reasoning_templates(self):
        """Initialize specialty-specific reasoning templates."""
        self.reasoning_templates = {
            'obstetrics_gynecology': (
                "1. Assess pregnancy status and gestational age\n"
                "2. Consider maternal and fetal safety\n"
                "3. Evaluate obstetric/gynecologic history\n"
                "4. Check for contraindications in pregnancy\n"
                "5. Apply OB/GYN-specific treatment guidelines"
            ),
            'dermatology': (
                "1. Describe lesion characteristics (morphology, distribution)\n"
                "2. Consider differential diagnoses based on appearance\n"
                "3. Evaluate patient history and exposures\n"
                "4. Consider systemic vs localized disease\n"
                "5. Apply dermatologic treatment guidelines"
            ),
            'gastroenterology': (
                "1. Localize symptoms (upper vs lower GI)\n"
                "2. Consider liver function and hepatic involvement\n"
                "3. Evaluate GI-specific risk factors\n"
                "4. Consider endoscopic findings if available\n"
                "5. Apply gastroenterology treatment guidelines"
            ),
            'cardiology': (
                "1. Assess cardiac risk factors and presentation\n"
                "2. Evaluate cardiac biomarkers and ECG findings\n"
                "3. Consider acute vs chronic cardiac conditions\n"
                "4. Check for cardiac contraindications\n"
                "5. Apply cardiology treatment guidelines"
            ),
            'default': (
                "1. Extract clinical features\n"
                "2. Generate differential diagnoses\n"
                "3. Gather and score evidence\n"
                "4. Match treatment guidelines\n"
                "5. Select answer with confidence"
            ),
        }
    
    def _init_specialty_retrieval_focus(self):
        """Initialize specialty-specific retrieval focus terms."""
        self.retrieval_focus = {
            'obstetrics_gynecology': [
                'pregnancy', 'maternal', 'fetal', 'obstetric', 'gynecologic',
                'gestational', 'prenatal', 'postpartum'
            ],
            'dermatology': [
                'skin', 'cutaneous', 'dermatologic', 'rash', 'lesion',
                'dermatitis', 'eczema'
            ],
            'gastroenterology': [
                'gastrointestinal', 'hepatic', 'liver', 'abdominal',
                'gi', 'gastroenterology'
            ],
        }
    
    def detect_specialty(self, query: str, case_description: str = "") -> SpecialtyAdaptation:
        """
        Detect medical specialty from query.
        
        Args:
            query: Medical query
            case_description: Optional case description
            
        Returns:
            SpecialtyAdaptation with detected specialty
        """
        full_text = (case_description + " " + query).lower()
        
        # Score each specialty
        specialty_scores = defaultdict(float)
        
        for specialty, keywords in self.specialty_keywords.items():
            for keyword in keywords:
                if keyword in full_text:
                    specialty_scores[specialty] += 1.0
        
        # Normalize scores
        total_score = sum(specialty_scores.values())
        if total_score > 0:
            specialty_scores = {
                spec: score / total_score
                for spec, score in specialty_scores.items()
            }
        
        # Get top specialty
        if specialty_scores:
            detected_specialty = max(specialty_scores.items(), key=lambda x: x[1])[0]
            confidence = specialty_scores[detected_specialty]
        else:
            detected_specialty = 'general_medicine'
            confidence = 0.0
        
        # Adapt query for specialty
        adapted_query = self._adapt_query_for_specialty(query, detected_specialty)
        
        # Get specialty keywords
        specialty_keywords = self.specialty_keywords.get(detected_specialty, [])
        
        # Get reasoning template
        reasoning_template = self.reasoning_templates.get(
            detected_specialty,
            self.reasoning_templates['default']
        )
        
        # Get retrieval focus
        retrieval_focus = self.retrieval_focus.get(detected_specialty, [])
        
        return SpecialtyAdaptation(
            detected_specialty=detected_specialty,
            confidence=confidence,
            adapted_query=adapted_query,
            specialty_keywords=specialty_keywords,
            reasoning_template=reasoning_template,
            retrieval_focus=retrieval_focus
        )
    
    def _adapt_query_for_specialty(self, query: str, specialty: str) -> str:
        """Adapt query with specialty-specific terms."""
        adapted = query
        
        # Add specialty-specific expansion terms
        if specialty in self.retrieval_focus:
            focus_terms = self.retrieval_focus[specialty]
            # Add terms that aren't already in query
            query_lower = query.lower()
            for term in focus_terms:
                if term not in query_lower:
                    adapted += f" {term}"
        
        return adapted
    
    def adapt_retrieval(
        self,
        query: str,
        specialty: str,
        retrieved_docs: List[Document]
    ) -> List[Document]:
        """
        Adapt retrieval results for specialty.
        
        Re-ranks documents to prioritize specialty-relevant ones.
        """
        if specialty == 'general_medicine' or not retrieved_docs:
            return retrieved_docs
        
        # Score documents by specialty relevance
        scored_docs = []
        specialty_keywords = self.specialty_keywords.get(specialty, [])
        
        for doc in retrieved_docs:
            score = 0.0
            doc_text_lower = doc.content.lower()
            doc_category = doc.metadata.get('category', '').lower()
            
            # Check category match
            if specialty.replace('_', ' ') in doc_category:
                score += 2.0
            
            # Check keyword matches
            for keyword in specialty_keywords:
                if keyword in doc_text_lower:
                    score += 1.0
            
            scored_docs.append((doc, score))
        
        # Sort by specialty relevance
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return re-ranked documents
        return [doc for doc, _ in scored_docs]
    
    def adapt_reasoning(
        self,
        specialty: str,
        reasoning_steps: List[Dict]
    ) -> List[Dict]:
        """
        Adapt reasoning steps for specialty.
        
        Adds specialty-specific considerations to reasoning.
        """
        if specialty == 'general_medicine':
            return reasoning_steps
        
        # Get specialty template
        template = self.reasoning_templates.get(specialty)
        if not template:
            return reasoning_steps
        
        # Add specialty-specific reasoning step
        specialty_step = {
            'step': len(reasoning_steps) + 1,
            'name': f'{specialty.replace("_", " ").title()} Considerations',
            'description': template,
            'specialty_specific': True
        }
        
        reasoning_steps.append(specialty_step)
        return reasoning_steps
    
    def get_specialty_specific_guidelines(
        self,
        specialty: str,
        all_guidelines: List[Document]
    ) -> List[Document]:
        """Filter guidelines to specialty-specific ones."""
        if specialty == 'general_medicine':
            return all_guidelines
        
        specialty_keywords = self.specialty_keywords.get(specialty, [])
        specialty_guidelines = []
        
        for guideline in all_guidelines:
            category = guideline.metadata.get('category', '').lower()
            title = guideline.metadata.get('title', '').lower()
            
            # Check if guideline matches specialty
            if specialty.replace('_', ' ') in category:
                specialty_guidelines.append(guideline)
            elif any(kw in title for kw in specialty_keywords[:3]):  # Check top 3 keywords
                specialty_guidelines.append(guideline)
        
        return specialty_guidelines


def main():
    """Demo: Test specialty adapter."""
    print("="*70)
    print("SPECIALTY ADAPTER DEMO")
    print("="*70)
    
    adapter = SpecialtyAdapter()
    
    test_cases = [
        ("What is the treatment for a pregnant patient with preeclampsia?", ""),
        ("Patient with severe rash and itching", ""),
        ("How to manage liver disease in a patient with jaundice?", ""),
        ("What is the treatment for acute MI?", ""),
    ]
    
    for query, case_desc in test_cases:
        print(f"\n{'-'*70}")
        print(f"Query: {query}")
        print(f"{'-'*70}")
        
        adaptation = adapter.detect_specialty(query, case_desc)
        
        print(f"\nDetected Specialty: {adaptation.detected_specialty}")
        print(f"Confidence: {adaptation.confidence:.2%}")
        print(f"\nAdapted Query: {adaptation.adapted_query}")
        print(f"\nSpecialty Keywords: {adaptation.specialty_keywords[:5]}")
        print(f"\nReasoning Template:")
        print(adaptation.reasoning_template)
        print(f"\nRetrieval Focus: {adaptation.retrieval_focus[:5]}")
    
    print("\n" + "="*70)
    print("[OK] Specialty Adapter operational!")
    print("="*70)


if __name__ == "__main__":
    main()

