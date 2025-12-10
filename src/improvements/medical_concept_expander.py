"""
Medical Concept Expansion

This module expands queries with related medical concepts to improve retrieval.

Day 7 Phase 3: Retrieval Optimization
"""

from typing import List, Set, Dict
import re
import json
from pathlib import Path


class MedicalConceptExpander:
    """
    Expand queries with related medical concepts.
    
    Uses simple medical concept relationships (free resources).
    """
    
    def __init__(self):
        """Initialize concept expander."""
        self._init_concept_relationships()
        self.umls_synonyms = self._load_umls_synonyms()
    
    def _init_concept_relationships(self):
        """Initialize medical concept relationships."""
        self.concept_relationships = {
            # Condition relationships
            'myocardial infarction': ['chest pain', 'troponin', 'ecg', 'ekg', 'stemi', 'nstemi', 'acs', 'coronary'],
            'mi': ['myocardial infarction', 'chest pain', 'troponin', 'ecg', 'heart attack'],
            'heart attack': ['myocardial infarction', 'mi', 'chest pain', 'troponin'],
            'pneumonia': ['fever', 'cough', 'chest pain', 'dyspnea', 'respiratory', 'lung'],
            'sepsis': ['fever', 'infection', 'bacteremia', 'shock', 'hypotension'],
            'meningitis': ['headache', 'fever', 'stiff neck', 'cns', 'neurological'],
            'hypoglycemia': ['glucose', 'diabetes', 'insulin', 'sugar', 'low glucose'],
            'hypoglycaemia': ['glucose', 'diabetes', 'insulin', 'sugar', 'low glucose'],
            
            # Treatment relationships
            'treatment': ['therapy', 'management', 'medication', 'intervention'],
            'antibiotic': ['antimicrobial', 'antibacterial', 'anti-infective'],
            'aspirin': ['asa', 'acetylsalicylic acid', 'antiplatelet'],
            
            # Medication relationships
            'gentamicin': ['aminoglycoside', 'antibiotic', 'gram-negative'],
            'ceftriaxone': ['cephalosporin', 'antibiotic', 'rocephin'],
            'metronidazole': ['flagyl', 'antiprotozoal', 'anaerobic'],
            
            # Symptom relationships
            'chest pain': ['angina', 'precordial pain', 'retrosternal pain'],
            'shortness of breath': ['dyspnea', 'sob', 'breathlessness'],
            'fever': ['pyrexia', 'elevated temperature', 'hyperthermia'],
            
            # Specialty relationships
            'pediatric': ['child', 'neonate', 'newborn', 'infant', 'baby'],
            'obstetrics': ['pregnancy', 'pregnant', 'gestation', 'gynecology'],
            'cardiology': ['heart', 'cardiac', 'coronary', 'cardiovascular'],
        }

    def _load_umls_synonyms(self) -> Dict[str, List[str]]:
        """
        Load lightweight UMLS-derived synonym map if available.
        Expects a JSON dict at data/umls_synonyms.json: {"term": ["syn1", "syn2", ...], ...}
        Gracefully falls back to empty if file missing.
        """
        umls_path = Path("data/umls_synonyms.json")
        if umls_path.exists():
            try:
                with umls_path.open("r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                # normalize keys/lists to lower
                norm = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        norm[k.lower()] = [s.lower() for s in v]
                print(f"[INFO] Loaded UMLS synonym map with {len(norm)} entries from {umls_path}")
                return norm
            except Exception as e:
                print(f"[WARN] Failed to load UMLS synonyms: {e}")
        return {}
    
    def expand(self, query: str, max_expansions: int = 3) -> str:
        """
        Expand query with related medical concepts.
        
        Args:
            query: Original query
            max_expansions: Maximum number of concepts to add
            
        Returns:
            Expanded query
        """
        query_lower = query.lower()
        expansions = []
        
        # Find related concepts
        for concept, related in self.concept_relationships.items():
            if concept in query_lower:
                # Add related concepts that aren't already in query
                for related_concept in related:
                    if related_concept not in query_lower and related_concept not in expansions:
                        expansions.append(related_concept)
                        if len(expansions) >= max_expansions:
                            break
                if len(expansions) >= max_expansions:
                    break

        # UMLS synonym expansion (if available)
        if self.umls_synonyms and len(expansions) < max_expansions:
            # Simple phrase match against keys; if key present, add its synonyms
            for term, syns in self.umls_synonyms.items():
                if term in query_lower:
                    for syn in syns:
                        if syn not in query_lower and syn not in expansions:
                            expansions.append(syn)
                            if len(expansions) >= max_expansions:
                                break
                if len(expansions) >= max_expansions:
                    break
        
        # Add expansions to query
        if expansions:
            expanded_query = f"{query} {' '.join(expansions)}"
            return expanded_query
        
        return query
    
    def get_critical_concepts(self, query: str, case_description: str = "") -> List[str]:
        """
        Extract critical medical concepts from query and case.
        
        Returns:
            List of critical concepts that should be in retrieved documents
        """
        full_text = f"{case_description} {query}".lower()
        critical_concepts = []
        
        # Extract conditions
        condition_keywords = [
            'myocardial infarction', 'mi', 'pneumonia', 'sepsis', 'meningitis',
            'hypoglycemia', 'hypoglycaemia', 'diabetes', 'hypertension', 'hypotension'
        ]
        for keyword in condition_keywords:
            if keyword in full_text:
                critical_concepts.append(keyword)
        
        # Extract medications
        medication_keywords = [
            'gentamicin', 'ceftriaxone', 'aspirin', 'metronidazole', 'insulin',
            'ampicillin', 'cefotaxime', 'azithromycin'
        ]
        for keyword in medication_keywords:
            if keyword in full_text:
                critical_concepts.append(keyword)
        
        return critical_concepts[:5]  # Limit to 5

