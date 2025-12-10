# Data Documentation

**Author:** Shreya Uprety  
**Last Updated:** December 11, 2025

---

## Overview

The `data/` directory contains all medical knowledge, indexes, and datasets used by the Medical Question-Answering System. This includes structured clinical guidelines (20 topics), FAISS and BM25 indexes for retrieval, generated clinical question datasets, and UMLS synonym/concept expansion mappings.

---

## Directory Structure

```
data/
├── guidelines/                         # 20 structured clinical guidelines
├── indexes/                            # FAISS and BM25 indexes  
├── processed/questions/                # Generated clinical cases
├── raw/                                # Raw PDF extraction
├── standard-treatment-guidelines.pdf   # Source medical PDF
├── umls_synonyms.json                  # Medical synonym mappings
└── umls_expansion.json                 # Concept expansion rules
```

---

## Guideline Files

Each guideline follows a structured template:

```
GUIDELINE: [Condition Name]
CATEGORY: [Medical Category]

DEFINITION:
[Detailed condition definition, epidemiology, pathophysiology]

DIAGNOSIS:
[Diagnostic criteria, clinical presentation, laboratory findings]

TREATMENT:
[First-line treatment, medication protocols, dosing]

MANAGEMENT:
[Long-term management, monitoring, patient education]
```

**Coverage:** 20 guidelines across Cardiovascular (6), Respiratory (3), Infectious (2), Gastrointestinal (3), Renal/Metabolic (2), Rheumatologic (2), Psychiatric (1), and General Medicine (1).

---

## Index Files

### FAISS Index (`indexes/faiss_index.bin`)
- Type: IndexFlatL2 (brute-force L2 distance)
- Dimensions: 384 (MiniLM-L6-v2 embeddings)
- Indexed Chunks: 205
- Size: ~300KB

### Document Metadata (`indexes/documents.pkl`)
Stores chunk metadata for retrieved documents (source file, section, guideline name, category).

### BM25 Index (`indexes/bm25_index.pkl`)
Lexical retrieval using BM25 algorithm (k1=1.5, b=0.75), vocabulary size ~5,000 terms.

### Concept Index (`indexes/concept_index.json`)
Maps medical concepts to relevant guideline chunks for concept-first retrieval.

---

## Processed Data

### Clinical Questions (`processed/questions/questions_1.json`)

**Format:**
```json
{
    "questions": [
        {
            "id": "q_001",
            "case_description": "Clinical scenario...",
            "question": "What is the most appropriate next step?",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
            "correct_answer": "B",
            "explanation": "Reasoning...",
            "difficulty": "medium",
            "question_type": "immediate_action",
            "relevance": "high",
            "guideline_source": "guideline_01_cardiovascular_emergencies_acs.txt",
            "medical_concepts": [...],
            "condition": "Acute Coronary Syndrome"
        }
    ],
    "metadata": {
        "total_questions": 100,
        "answer_distribution": {"A": 25, "B": 25, "C": 22, "D": 28},
        "difficulty_distribution": {"easy": 23, "medium": 46, "hard": 31}
    }
}
```

---

## UMLS Data

### Synonym Mappings (`umls_synonyms.json`)
Medical term synonyms from UMLS Metathesaurus (~500 concepts, ~3,000 synonyms).

```json
{
    "myocardial_infarction": ["heart attack", "MI", "acute MI", "AMI"],
    "hypertension": ["high blood pressure", "HTN", "elevated BP"]
}
```

### Concept Expansion (`umls_expansion.json`)
Maps concepts to related concepts (hierarchical and associative relationships).

```json
{
    "chest_pain": {
        "is_a": ["angina", "cardiac pain"],
        "related": ["dyspnea", "palpitations"],
        "causes": ["acute_coronary_syndrome", "pulmonary_embolism"]
    }
}
```

---

## Data Pipeline

### Guideline Creation
```
PDF → PDFExtractor → raw/extracted_text.txt → GuidelineGenerator → guidelines/*.txt → rebuild_index.py → indexes/
```

### Question Generation
```
guidelines/*.txt → generate_clinical_cases_v5.py → LLM (Llama 3.1 8B) → processed/questions/questions_1.json
```

---

## Related Documentation

- [Data Creation Documentation](data_creation_documentation.md)
- [Retrieval Documentation](retrieval_documentation.md)
- [Part 1: Dataset Creation](part_1_dataset_creation.md)

---

**Documentation Author:** Shreya Uprety
