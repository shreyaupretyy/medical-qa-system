"""
Medical Query Enhancement Module

This module enhances medical queries with:
1. Medical Named Entity Recognition (NER)
2. Synonym expansion using medical lexicons
3. Abbreviation resolution
4. Query decomposition for complex cases

Addresses Day 4 Issue: Precision@k = 0.0 due to poor query understanding
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EnhancedQuery:
    """Enhanced query with medical entities and expansions."""
    original_query: str
    expanded_query: str
    medical_entities: Dict[str, List[str]]  # entity_type -> [entities]
    synonyms: List[str]
    abbreviations_resolved: Dict[str, str]  # abbrev -> full_term
    query_variants: List[str]  # Alternative query formulations
    medical_concepts: Set[str]


class MedicalQueryEnhancer:
    """
    Enhances medical queries with NER, synonym expansion, and normalization.
    
    This addresses the critical retrieval precision issue by:
    - Expanding medical abbreviations (MI → myocardial infarction)
    - Adding medical synonyms (heart attack → myocardial infarction)
    - Extracting medical entities (symptoms, diseases, treatments)
    - Creating query variants for better retrieval
    """
    
    def __init__(self):
        """Initialize with medical knowledge bases."""
        self._init_medical_abbreviations()
        self._init_medical_synonyms()
        self._init_medical_entities()
        self._init_medical_concepts()
    
    def _init_medical_abbreviations(self):
        """Initialize comprehensive medical abbreviation dictionary."""
        self.abbreviations = {
            # Cardiology
            'mi': 'myocardial infarction',
            'ami': 'acute myocardial infarction',
            'stemi': 'st-elevation myocardial infarction',
            'nstemi': 'non-st-elevation myocardial infarction',
            'chf': 'congestive heart failure',
            'cad': 'coronary artery disease',
            'acs': 'acute coronary syndrome',
            'afib': 'atrial fibrillation',
            'vt': 'ventricular tachycardia',
            'vf': 'ventricular fibrillation',
            'ecg': 'electrocardiogram',
            'ekg': 'electrocardiogram',
            'echo': 'echocardiogram',
            'cath': 'catheterization',
            
            # Pulmonary
            'copd': 'chronic obstructive pulmonary disease',
            'pe': 'pulmonary embolism',
            'ards': 'acute respiratory distress syndrome',
            'copd': 'chronic obstructive pulmonary disease',
            
            # Endocrinology
            'dm': 'diabetes mellitus',
            'dm2': 'diabetes mellitus type 2',
            't1dm': 'type 1 diabetes mellitus',
            't2dm': 'type 2 diabetes mellitus',
            'dka': 'diabetic ketoacidosis',
            'htn': 'hypertension',
            
            # Gastroenterology
            'gi': 'gastrointestinal',
            'gerd': 'gastroesophageal reflux disease',
            'ibd': 'inflammatory bowel disease',
            'ibs': 'irritable bowel syndrome',
            'peptic': 'peptic ulcer disease',
            
            # Neurology
            'cva': 'cerebrovascular accident',
            'tia': 'transient ischemic attack',
            'sa': 'subarachnoid hemorrhage',
            'ich': 'intracerebral hemorrhage',
            'seizure': 'seizure disorder',
            
            # Nephrology
            'aki': 'acute kidney injury',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'bun': 'blood urea nitrogen',
            'cr': 'creatinine',
            
            # Infectious Disease
            'uti': 'urinary tract infection',
            'pneumonia': 'pneumonia',
            'sepsis': 'sepsis',
            'mrsa': 'methicillin-resistant staphylococcus aureus',
            
            # Medications
            'ace': 'angiotensin converting enzyme',
            'arb': 'angiotensin receptor blocker',
            'asa': 'aspirin',
            'tpa': 'tissue plasminogen activator',
            'nsaids': 'nonsteroidal anti-inflammatory drugs',
            'ppi': 'proton pump inhibitor',
            'h2': 'h2 receptor antagonist',
            
            # Tests/Procedures
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'us': 'ultrasound',
            'cxr': 'chest x-ray',
            'ekg': 'electrocardiogram',
            'cbc': 'complete blood count',
            'cmp': 'comprehensive metabolic panel',
            'pt': 'prothrombin time',
            'inr': 'international normalized ratio',
            'a1c': 'hemoglobin a1c',
            'trop': 'troponin',
            'bnp': 'b-type natriuretic peptide',
        }
    
    def _init_medical_synonyms(self):
        """Initialize medical synonym groups for query expansion."""
        self.synonym_groups = {
            'myocardial_infarction': [
                'myocardial infarction', 'mi', 'ami', 'heart attack',
                'cardiac infarction', 'coronary infarction'
            ],
            'chest_pain': [
                'chest pain', 'angina', 'chest discomfort', 'precordial pain',
                'retrosternal pain', 'chest tightness'
            ],
            'shortness_of_breath': [
                'shortness of breath', 'dyspnea', 'sob', 'breathlessness',
                'difficulty breathing', 'respiratory distress'
            ],
            'hypertension': [
                'hypertension', 'htn', 'high blood pressure',
                'elevated blood pressure', 'bp elevation'
            ],
            'diabetes': [
                'diabetes mellitus', 'dm', 'diabetes', 'diabetes type 2',
                'dm2', 'type 2 diabetes', 't2dm'
            ],
            'stroke': [
                'cerebrovascular accident', 'cva', 'stroke', 'brain attack',
                'cerebral infarction', 'brain infarction'
            ],
            'kidney_failure': [
                'renal failure', 'kidney failure', 'aki', 'acute kidney injury',
                'renal insufficiency', 'kidney dysfunction'
            ],
            'infection': [
                'infection', 'sepsis', 'bacteremia', 'systemic infection',
                'septicemia', 'bloodstream infection'
            ],
            'pneumonia': [
                'pneumonia', 'pneumonitis', 'lung infection',
                'respiratory infection', 'pulmonary infection'
            ],
            'liver_disease': [
                'liver disease', 'hepatic disease', 'hepatitis',
                'liver dysfunction', 'hepatic dysfunction'
            ],
            # Day 7: Additional synonym groups for better retrieval
            'newborn': [
                'newborn', 'neonate', 'neonatal', 'infant', 'baby',
                'newborn infant', 'neonatal period'
            ],
            'hypoglycemia': [
                'hypoglycemia', 'hypoglycaemia', 'low blood sugar', 'low glucose',
                'hypoglycemic', 'blood glucose low'
            ],
            'meningitis': [
                'meningitis', 'meningeal', 'meningeal infection', 'cns infection',
                'bacterial meningitis', 'meningeal inflammation'
            ],
            'skin_infection': [
                'skin infection', 'bacterial skin infection', 'cellulitis', 'boil', 'abscess',
                'furuncle', 'carbuncle', 'skin abscess', 'cutaneous infection'
            ],
            'liver_abscess': [
                'liver abscess', 'hepatic abscess', 'amoebic liver abscess', 'hepatic infection',
                'hepatic collection', 'liver collection'
            ],
            'treatment': [
                'treatment', 'therapy', 'management', 'intervention', 'care', 'medication',
                'therapeutic', 'therapeutic intervention'
            ],
            'initial_treatment': [
                'initial treatment', 'first-line treatment', 'primary treatment', 'first treatment',
                'initial therapy', 'first-line therapy', 'primary therapy', 'initial management'
            ],
            'antibiotic': [
                'antibiotic', 'antimicrobial', 'antibacterial', 'anti-infective', 'antimicrobial agent',
                'antibacterial agent', 'anti-infective agent'
            ],
            'neonatal': [
                'neonatal', 'newborn', 'neonate', 'infant', 'baby', 'neonatal period',
                'newborn infant', 'neonatal care'
            ],
            'sick_newborn': [
                'sick newborn', 'ill newborn', 'newborn sepsis', 'neonatal sepsis',
                'newborn infection', 'neonatal infection'
            ],
        }
    
    def _init_medical_entities(self):
        """Initialize medical entity patterns for NER."""
        self.entity_patterns = {
            'symptom': [
                r'\b(pain|fever|cough|shortness of breath|dyspnea|chest pain|headache|'
                r'nausea|vomiting|diarrhea|fatigue|weakness|dizziness|syncope|seizure|'
                r'rash|jaundice|edema|swelling|bleeding)\b',
            ],
            'disease': [
                r'\b(myocardial infarction|heart attack|stroke|cva|diabetes|pneumonia|'
                r'sepsis|infection|hypertension|kidney failure|renal failure)\b',
            ],
            'medication': [
                r'\b(aspirin|metformin|insulin|morphine|nitroglycerin|metoprolol|'
                r'lisinopril|atorvastatin|warfarin|heparin|antibiotic|anticoagulant)\b',
            ],
            'test': [
                r'\b(ecg|ekg|electrocardiogram|ct|mri|troponin|creatinine|glucose|'
                r'hemoglobin|a1c|blood test|lab test|culture|biopsy)\b',
            ],
            'procedure': [
                r'\b(catheterization|surgery|biopsy|endoscopy|colonoscopy|'
                r'angiography|echocardiogram)\b',
            ],
        }
    
    def _init_medical_concepts(self):
        """Initialize medical concept mappings."""
        self.medical_concepts = {
            'cardiac': ['heart', 'cardiac', 'myocardial', 'coronary', 'cardiovascular'],
            'pulmonary': ['lung', 'pulmonary', 'respiratory', 'breathing'],
            'neurological': ['brain', 'neurological', 'cns', 'cerebral'],
            'gastrointestinal': ['liver', 'hepatic', 'abdomen', 'abdominal', 'gi'],
            'renal': ['kidney', 'renal', 'nephro'],
            'endocrine': ['diabetes', 'glucose', 'insulin', 'thyroid', 'hormone'],
        }
    
    def enhance(self, query: str) -> EnhancedQuery:
        """
        Enhance a medical query with NER, expansion, and normalization.
        
        Args:
            query: Original medical query
            
        Returns:
            EnhancedQuery with all enhancements
        """
        # Step 1: Extract medical entities
        medical_entities = self._extract_medical_entities(query)
        
        # Step 2: Resolve abbreviations
        abbreviations_resolved = self._resolve_abbreviations(query)
        expanded_query = self._expand_abbreviations_in_text(query)
        
        # Step 3: Expand synonyms
        synonyms = self._extract_synonyms(expanded_query)
        expanded_query = self._expand_synonyms(expanded_query)
        
        # Step 4: Extract medical concepts
        medical_concepts = self._extract_medical_concepts(expanded_query)
        
        # Step 5: Generate query variants
        query_variants = self._generate_query_variants(expanded_query, medical_entities, synonyms)
        
        return EnhancedQuery(
            original_query=query,
            expanded_query=expanded_query,
            medical_entities=medical_entities,
            synonyms=synonyms,
            abbreviations_resolved=abbreviations_resolved,
            query_variants=query_variants,
            medical_concepts=medical_concepts
        )
    
    def _extract_medical_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract medical entities using pattern matching."""
        entities = defaultdict(list)
        query_lower = query.lower()
        
        # Extract by entity type
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query_lower, re.IGNORECASE)
                for match in matches:
                    entity = match.group(0)
                    if entity not in entities[entity_type]:
                        entities[entity_type].append(entity)
        
        return dict(entities)
    
    def _resolve_abbreviations(self, query: str) -> Dict[str, str]:
        """Resolve abbreviations in query."""
        resolved = {}
        query_lower = query.lower()
        
        for abbrev, full_term in self.abbreviations.items():
            # Word boundary matching
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, query_lower, re.IGNORECASE):
                resolved[abbrev] = full_term
        
        return resolved
    
    def _expand_abbreviations_in_text(self, query: str) -> str:
        """Expand abbreviations in query text."""
        expanded = query
        query_lower = query.lower()
        
        # Sort by length (longer first) to avoid partial matches
        sorted_abbrevs = sorted(self.abbreviations.items(), key=lambda x: len(x[0]), reverse=True)
        
        for abbrev, full_term in sorted_abbrevs:
            # Word boundary matching
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Replace with full term (preserve case)
                expanded = re.sub(pattern, full_term, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def _extract_synonyms(self, query: str) -> List[str]:
        """Extract synonyms that match query terms."""
        synonyms = []
        query_lower = query.lower()
        
        for group_name, synonym_list in self.synonym_groups.items():
            for synonym in synonym_list:
                if synonym in query_lower:
                    # Add other synonyms from the group
                    for other_synonym in synonym_list:
                        if other_synonym != synonym and other_synonym not in synonyms:
                            synonyms.append(other_synonym)
                    break
        
        return synonyms
    
    def _expand_synonyms(self, query: str) -> str:
        """Expand query with synonyms."""
        expanded = query
        query_lower = query.lower()
        
        for group_name, synonym_list in self.synonym_groups.items():
            # Check if any synonym from group is in query
            found_synonym = None
            for synonym in synonym_list:
                if synonym in query_lower:
                    found_synonym = synonym
                    break
            
            if found_synonym:
                # Add other synonyms from the group to the query
                for other_synonym in synonym_list:
                    if other_synonym != found_synonym and other_synonym not in query_lower:
                        expanded += f" {other_synonym}"
        
        return expanded
    
    def _extract_medical_concepts(self, query: str) -> Set[str]:
        """Extract medical concepts from query."""
        concepts = set()
        query_lower = query.lower()
        
        for concept, keywords in self.medical_concepts.items():
            if any(kw in query_lower for kw in keywords):
                concepts.add(concept)
        
        return concepts
    
    def _generate_query_variants(self, query: str, entities: Dict, synonyms: List[str]) -> List[str]:
        """Generate alternative query formulations."""
        variants = [query]  # Original is first variant
        
        # Variant 1: Add resolved abbreviations
        if entities.get('disease') or entities.get('symptom'):
            variant = query
            for abbrev, full_term in self.abbreviations.items():
                if abbrev in query.lower():
                    variant = variant.replace(abbrev, full_term)
            if variant != query:
                variants.append(variant)
        
        # Variant 2: Add synonyms
        if synonyms:
            variant = query + " " + " ".join(synonyms[:3])  # Add top 3 synonyms
            variants.append(variant)
        
        # Variant 3: Focus on medical entities
        if entities:
            entity_terms = []
            for entity_list in entities.values():
                entity_terms.extend(entity_list[:2])  # Top 2 per type
            if entity_terms:
                variant = " ".join(entity_terms)
                variants.append(variant)
        
        return variants[:5]  # Limit to 5 variants
    
    def get_enhanced_query_for_retrieval(self, query: str, max_expansions: int = 5) -> str:
        """
        Get the best enhanced query for retrieval.
        
        Day 7: Enhanced for "initial treatment" queries and better retrieval.
        """
        enhanced = self.enhance(query)
        
        # Start with original query (preserve original intent)
        # PubMedBERT: For medical queries, start with enhanced version for better semantic matching
        retrieval_query = enhanced.expanded_query if hasattr(enhanced, 'expanded_query') and enhanced.expanded_query else query
        
        # Day 7 Phase 2: Enhanced "initial treatment" query expansion
        query_lower = query.lower()
        if 'initial' in query_lower and ('treatment' in query_lower or 'therapy' in query_lower or 'management' in query_lower or 'step' in query_lower):
            # Add synonyms for "initial treatment" - more comprehensive
            retrieval_query += " first-line treatment primary treatment initial therapy first-line therapy immediate treatment stat treatment first treatment initial management first-line management immediate management"
            # Add medication-related terms for treatment queries
            retrieval_query += " medication drug prescription administer dose dosage"
        
        # Day 7 Phase 2: Add temporal context terms (more comprehensive)
        if 'initial' in query_lower or 'first' in query_lower:
            retrieval_query += " immediate stat first first-line primary acute emergency urgent"
        elif 'definitive' in query_lower or 'long-term' in query_lower:
            retrieval_query += " definitive standard complete full long-term maintenance chronic"
        
        # Day 7: Enhanced Pediatrics query expansion
        if any(term in query_lower for term in ['newborn', 'neonate', 'infant', 'child', 'pediatric', 'pediatric', 'sick newborn']):
            retrieval_query += " neonatal newborn infant pediatric child pediatric care neonatal care"
            # Add common pediatric conditions
            if 'sepsis' in query_lower or 'infection' in query_lower:
                retrieval_query += " neonatal sepsis newborn infection"
            if 'hypoglycemia' in query_lower or 'glucose' in query_lower:
                retrieval_query += " neonatal hypoglycemia newborn hypoglycemia"
        
        # Day 7: Enhanced management query expansion
        if 'management' in query_lower or 'manage' in query_lower:
            retrieval_query += " treatment therapy care protocol intervention approach"
        
        # PubMedBERT: Extract and boost guideline names from query
        # Check for quoted guideline names (e.g., "Sick Newborn" guideline)
        guideline_name_patterns = [
            r"'([^']+)' guideline",
            r'"([^"]+)" guideline',
            r'guideline[,\s]+([^,\.]+)',
            r'based on the ([^,\.]+)',
            r'according to the ([^,\.]+)',
            r'per the ([^,\.]+)',
            r'Sick Newborn',  # Direct mentions
            r'Amoebic Liver Abscess',
            r'Acute Glomerulonephritis',
            r'Neonatal Hypoglycaemia',
            r'Headache',
            r'Bacterial Skin Infections',
            r'Diabetes Mellitus',
            r'Sexually Transmitted Infections',
            r'HIV Infection and AIDS',
            r'Fever',
            r'Stridor',
            r'Acute Epiglottitis',
            r'Oral and Dental Conditions',
            r'Antibiotic Prophylaxis In Surgery',
            r'Management of Acute Pain',
            r'General Management of Poisoning',
            r'Medicines Use In The Elderly',
            r'Local Anaesthetic Agents',
            r'Respiratory Distress in Children'
        ]
        import re
        guideline_name_found = None
        for pattern in guideline_name_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if match.groups():
                    guideline_name = match.group(1).strip()
                else:
                    guideline_name = match.group(0).strip()
                guideline_name_found = guideline_name
                # Add the guideline name multiple times for emphasis (boosted)
                retrieval_query = f"{guideline_name} {guideline_name} {guideline_name} {retrieval_query}"
                # Also add variations
                retrieval_query += f" {guideline_name.replace(' ', '_')} {guideline_name.replace('_', ' ')}"
                # Add common variations
                if 'Sick Newborn' in guideline_name:
                    retrieval_query += " sick newborn neonate neonatal problems newborn"
                elif 'Amoebic' in guideline_name:
                    retrieval_query += " amoebic amebic entamoeba histolytica"
                break
        
        # Also check for common guideline mentions without quotes
        if not guideline_name_found:
            common_guidelines = [
                'Sick Newborn', 'Amoebic Liver Abscess', 'Neonatal Hypoglycaemia',
                'Headache', 'Bacterial Skin Infections', 'Diabetes Mellitus',
                'Acute Glomerulonephritis', 'Fever', 'Stridor'
            ]
            for gl_name in common_guidelines:
                if gl_name.lower() in query_lower:
                    retrieval_query = f"{gl_name} {gl_name} {retrieval_query}"
                    break
        
        # PubMedBERT: Add more abbreviation expansions for better recall
        if enhanced.abbreviations_resolved:
            # Add expanded terms for key abbreviations
            for abbrev, full_term in list(enhanced.abbreviations_resolved.items())[:15]:  # Increased to top 15 for better recall
                if abbrev.lower() in query.lower():
                    retrieval_query += f" {full_term}"
                    # Also add common variations
                    if 'myocardial' in full_term.lower():
                        retrieval_query += " heart cardiac"
                    if 'infarction' in full_term.lower():
                        retrieval_query += " attack ischemia"
        
        # PubMedBERT: Add more synonyms for better recall
        if enhanced.synonyms:
            retrieval_query += " " + " ".join(enhanced.synonyms[:12])  # Increased to top 12 for better recall
        
        # PubMedBERT: Add medical entities for better recall
        if enhanced.medical_entities:
            entity_terms = []
            for entity_list in enhanced.medical_entities.values():
                entity_terms.extend(entity_list[:5])  # Increased to top 5 per type
            if entity_terms:
                retrieval_query += " " + " ".join(entity_terms[:15])  # Top 15 entities for better recall
        
        # PubMedBERT: Add medical concepts for better recall
        if enhanced.medical_concepts:
            key_concepts = list(enhanced.medical_concepts)[:15]  # Top 15 concepts
            retrieval_query += " " + " ".join(key_concepts)
        
        return retrieval_query


def main():
    """Demo: Test medical query enhancement."""
    print("="*70)
    print("MEDICAL QUERY ENHANCER DEMO")
    print("="*70)
    
    enhancer = MedicalQueryEnhancer()
    
    test_queries = [
        "What is the treatment for acute MI?",
        "Patient with elevated troponin and chest pain",
        "How to manage DM2 with metformin?",
        "Newborn with suspected sepsis needs treatment",
        "OB/GYN patient with pregnancy complications"
    ]
    
    for query in test_queries:
        print(f"\n{'-'*70}")
        print(f"Original Query: {query}")
        print(f"{'-'*70}")
        
        enhanced = enhancer.enhance(query)
        
        print(f"\nExpanded Query: {enhanced.expanded_query}")
        print(f"\nMedical Entities:")
        for entity_type, entities in enhanced.medical_entities.items():
            print(f"  {entity_type}: {entities}")
        
        print(f"\nAbbreviations Resolved: {enhanced.abbreviations_resolved}")
        print(f"\nSynonyms: {enhanced.synonyms[:5]}")  # Show first 5
        print(f"\nMedical Concepts: {enhanced.medical_concepts}")
        print(f"\nQuery Variants ({len(enhanced.query_variants)}):")
        for i, variant in enumerate(enhanced.query_variants[:3], 1):
            print(f"  {i}. {variant}")
        
        retrieval_query = enhancer.get_enhanced_query_for_retrieval(query)
        print(f"\nRetrieval Query: {retrieval_query}")
    
    print(f"\n{'='*70}")
    print("[OK] Medical Query Enhancer operational!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

