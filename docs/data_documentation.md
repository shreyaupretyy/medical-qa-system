# Data Documentation

**Author:** Shreya Uprety  
**Repository:** https://github.com/shreyaupretyy/medical-qa-system

## Overview

The `data/` directory contains all medical knowledge, indexes, and datasets used by the Medical Question-Answering System. This includes structured clinical guidelines (20 topics), FAISS indexes for semantic retrieval, generated clinical question datasets (50 cases with 52% system accuracy), and UMLS synonym/concept expansion mappings.

### Current System Performance (Based on Data)

- **Accuracy:** 52% on 50 clinical cases
- **Medical Concept Coverage:** 75.1%
- **Guideline Coverage:** 100%
- **Retrieval Performance:** MAP 0.268, Recall@5 56%

---

## Directory Structure

```
data/
├── guidelines/                         # 20 structured clinical guidelines
│   ├── guideline_01_cardiovascular_emergencies_acs.txt
│   ├── guideline_02_stroke_management_ischemic.txt
│   ├── guideline_03_heart_failure_management.txt
│   ├── guideline_04_hypertensive_emergencies.txt
│   ├── guideline_05_asthma_copd_exacerbation.txt
│   ├── guideline_06_pneumonia_management.txt
│   ├── guideline_07_pulmonary_embolism.txt
│   ├── guideline_08_gi_bleed_management.txt
│   ├── guideline_09_acute_pancreatitis.txt
│   ├── guideline_10_liver_failure.txt
│   ├── guideline_11_diabetic_emergencies.txt
│   ├── guideline_12_thyroid_storm.txt
│   ├── guideline_13_sepsis_management.txt
│   ├── guideline_14_uti_pyelonephritis.txt
│   ├── guideline_15_aki_management.txt
│   ├── guideline_16_acute_gout_attack.txt
│   ├── guideline_17_sle_flare_management.txt
│   ├── guideline_18_dvt_management.txt
│   ├── guideline_19_depression_suicidality.txt
│   └── guideline_20_seizure_management.txt
│
├── indexes/                            # FAISS indexes for semantic retrieval
│   ├── faiss_index.bin                 # Semantic search index (384 dim)
│   └── documents.pkl                   # Document metadata (205 chunks)
│
├── processed/questions/                # Generated clinical cases (50)
│   └── questions_1.json                # Main evaluation dataset
│
├── raw/                                # Raw PDF extraction
│   └── extracted_text.txt              # Raw text from PDF
│
├── standard-treatment-guidelines.pdf   # Source medical PDF (20 guidelines)
├── umls_synonyms.json                  # Medical synonym mappings
└── umls_expansion.json                 # Concept expansion rules
```

---

## Guideline Files (20 Total)

### Format and Structure

Each guideline follows a structured template optimized for retrieval and reasoning:

```
GUIDELINE: [Condition Name]
CATEGORY: [Medical Category]

DEFINITION:
[Comprehensive definition, epidemiology, pathophysiology]

DIAGNOSIS:
[Diagnostic criteria, clinical presentation, laboratory/imaging findings]

TREATMENT:
[Evidence-based treatment protocols with specific dosages]

MANAGEMENT:
[Long-term management strategies, monitoring, patient education]

CONTRAINDICATIONS:
[Important safety warnings and contraindications]
```

### Performance Coverage

Based on 50-case evaluation achieving 100% guideline coverage:

| Guideline | Category | Cases | Accuracy | Key Features |
|-----------|----------|-------|----------|--------------|
| guideline_01 | Cardiovascular | 11 | 54.5% | ACS protocols, ECG criteria |
| guideline_02 | Cardiovascular | 3 | 66.7% | Stroke imaging protocols |
| guideline_05 | Respiratory | 8 | 62.5% | Asthma/COPD exacerbation |
| guideline_08 | Gastroenterology | 7 | 71.4% | GI bleed management |
| guideline_11 | Endocrine | 6 | 66.7% | Diabetic emergencies |
| guideline_13 | Infectious Disease | 3 | 0% | Sepsis management |
| guideline_20 | Neurology | 2 | 0% | Seizure management |

### Specialty Coverage (11 Specialties)

- **Cardiovascular:** 11 cases, 54.5% accuracy
- **Respiratory:** 8 cases, 62.5% accuracy
- **Gastroenterology:** 7 cases, 71.4% accuracy
- **Endocrine:** 6 cases, 66.7% accuracy
- **Infectious Disease:** 3 cases, 0% accuracy
- **Nephrology:** 3 cases, 66.7% accuracy
- **Rheumatology:** 3 cases, 33.3% accuracy
- **Hematology:** 3 cases, 33.3% accuracy
- **Psychiatry:** 3 cases, 33.3% accuracy
- **Critical Care:** 1 case, 100% accuracy
- **Neurology:** 2 cases, 0% accuracy

---

## Index Files

### FAISS Index (`indexes/faiss_index.bin`)

- **Type:** IndexFlatL2 (brute-force L2 distance)
- **Dimensions:** 384 (sentence-transformers/all-MiniLM-L6-v2 embeddings)
- **Indexed Chunks:** 205 guideline segments
- **Size:** ~300KB
- **Performance:** Achieves 0.213 MAP, 44.5% recall@5 (Semantic-First strategy)
- **Identified Bottleneck:** General-purpose embeddings limit accuracy to 52%

### Document Metadata (`indexes/documents.pkl`)

Stores chunk metadata for retrieved documents:

```python
{
    "chunk_id": "guideline_01_001",
    "text": "Acute coronary syndrome requires immediate aspirin...",
    "source_file": "guideline_01_cardiovascular_emergencies_acs.txt",
    "section": "TREATMENT",
    "guideline_name": "Acute Coronary Syndrome",
    "category": "Cardiovascular",
    "medical_concepts": ["ACS", "aspirin", "nitroglycerin", "ECG"]
}
```

**Note:** BM25 is implemented in-memory at runtime and not stored as a separate index file.

---

## Processed Data: Clinical Questions Dataset

### Location: `data/processed/questions/questions_1.json`

### Dataset Statistics (50 Questions)

```json
{
  "metadata": {
    "evaluation_date": "2025-12-11T04:06:17.777485",
    "total_cases": 50,
    "split": "all",
    "evaluation_time_seconds": 5432.80569434166,
    "generator_version": "v5.0",
    "seed": 42,
    "system_accuracy": 0.52,
    "medical_concept_coverage": 0.7507612568837058,
    "guideline_coverage": 1.0
  },
  
  "questions": [
    {
      "question_id": "Q_001",
      "case_description": "47-year-old male with chest pain radiating to left arm...",
      "question": "What is the best answer?",
      "options": {
        "A": "Administer aspirin 325mg and nitroglycerin 0.4mg SL",
        "B": "Order cardiac enzymes and repeat ECG in 4 hours",
        "C": "Perform immediate cardioversion at 200J",
        "D": "Discharge with follow-up in cardiology clinic"
      },
      "correct_answer": "A",
      "confidence": 0.95,
      "specialty": "Cardiovascular",
      "difficulty": "moderate",
      "question_type": "diagnosis",
      "relevance_level": "high",
      "guideline_source": "guideline_01_cardiovascular_emergencies_acs.txt",
      "medical_concepts": ["chest_pain", "ACS", "aspirin", "nitroglycerin"],
      "condition": "Acute Coronary Syndrome",
      "vital_signs": {
        "BP": "160/95",
        "HR": "110",
        "RR": "22",
        "SpO2": "96%",
        "Temp": "98.6°F"
      },
      "symptoms": ["chest pain", "left arm radiation", "diaphoresis"]
    }
    // 49 more questions with similar structure...
  ]
}
```

### Dataset Distribution (Based on 52% Accuracy Evaluation)

| Dimension | Categories | Count | Percentage | Accuracy |
|-----------|-----------|-------|------------|----------|
| **Answer Distribution** | A | 19 | 38% | 52.6% |
| | B | 10 | 20% | 40.0% |
| | C | 12 | 24% | 75.0% |
| | D | 7 | 14% | 71.4% |
| | Cannot answer | 2 | 4% | N/A |
| **Difficulty** | Simple | 12 | 24% | 58.3% |
| | Moderate | 25 | 50% | 52.0% |
| | Complex | 13 | 26% | 46.2% |
| **Question Type** | Diagnosis | 46 | 92% | 52.2% |
| | Treatment | 2 | 4% | 100% |
| | Other | 2 | 4% | 0% |
| **Relevance Level** | High | 29 | 58% | 44.8% |
| | Medium | 10 | 20% | 80.0% |
| | Low | 11 | 22% | 45.5% |
| **Specialty** | Cardiovascular | 11 | 22% | 54.5% |
| | Respiratory | 8 | 16% | 62.5% |
| | Gastroenterology | 7 | 14% | 71.4% |
| | Endocrine | 6 | 12% | 66.7% |
| | Other (6 specialties) | 18 | 36% | 33.3% |

---

## UMLS Data for Medical Concept Expansion

### Synonym Mappings (`umls_synonyms.json`)

Medical term synonyms from UMLS Metathesaurus supporting 75.1% concept coverage:

```json
{
    "acute_coronary_syndrome": {
        "synonyms": ["ACS", "heart attack", "myocardial infarction", "MI", "AMI"],
        "category": "Cardiovascular",
        "frequency": "high"
    },
    "hypertension": {
        "synonyms": ["high blood pressure", "HTN", "elevated BP", "hypertensive"],
        "category": "Cardiovascular",
        "frequency": "high"
    },
    "diabetes_mellitus": {
        "synonyms": ["DM", "diabetes", "sugar", "hyperglycemia"],
        "category": "Endocrine",
        "frequency": "high"
    },
    "pneumonia": {
        "synonyms": ["community-acquired pneumonia", "CAP", "lung infection"],
        "category": "Respiratory",
        "frequency": "medium"
    },
    "sepsis": {
        "synonyms": ["septic shock", "systemic infection", "SIRS"],
        "category": "Infectious Disease",
        "frequency": "medium"
    }
}
```

**Statistics:**
- **Total Concepts:** ~500 medical concepts
- **Total Synonyms:** ~3,000 synonym mappings
- **Coverage:** 75.1% of medical concepts in questions
- **Impact:** +7% MAP improvement when enabled

### Concept Expansion (`umls_expansion.json`)

Maps concepts to related concepts for better retrieval:

```json
{
    "chest_pain": {
        "is_a": ["angina", "cardiac pain", "thoracic pain"],
        "related": ["dyspnea", "palpitations", "diaphoresis", "nausea"],
        "causes": ["acute_coronary_syndrome", "pulmonary_embolism", "aortic_dissection", "pericarditis"],
        "diagnostic_tests": ["ECG", "troponin", "chest_xray", "CT_angiogram"],
        "treatments": ["aspirin", "nitroglycerin", "morphine", "oxygen"]
    },
    "fever": {
        "is_a": ["pyrexia", "hyperthermia"],
        "related": ["infection", "inflammation", "sepsis"],
        "causes": ["bacterial_infection", "viral_infection", "inflammatory_condition"],
        "diagnostic_tests": ["blood_cultures", "cbc", "urinalysis", "chest_xray"],
        "treatments": ["antipyretics", "antibiotics", "fluid_resuscitation"]
    }
}
```

**Expansion Levels:**
- **Level 1:** Direct synonyms (e.g., "MI" → "myocardial infarction")
- **Level 2:** Related symptoms/signs (e.g., "chest pain" → "dyspnea", "diaphoresis")
- **Level 3:** Diagnostic tests (e.g., "chest pain" → "ECG", "troponin")
- **Level 4:** Treatments (e.g., "ACS" → "aspirin", "nitroglycerin")

---

## Raw Data

### PDF Extraction (`raw/extracted_text.txt`)

**Source:** `data/standard-treatment-guidelines.pdf`

**Characteristics:**
- **Size:** ~200,000 characters
- **Format:** Raw text extracted with pdfplumber
- **Cleaning:** Removed headers, footers, page numbers
- **Structure:** Semi-structured clinical guidelines

**Extraction Process:**

```python
# From src/data_creation/pdf_extractor.py
extractor = PDFExtractor("data/standard-treatment-guidelines.pdf")
raw_text = extractor.extract_text()  # -> raw/extracted_text.txt
print(f"Extracted {len(raw_text)} characters")
print(f"Contains {raw_text.count('guideline')} guideline mentions")
```

---

## Data Pipeline

### 1. Guideline Creation Pipeline (Source → Indexes)

```
PDF (standard-treatment-guidelines.pdf)
    ↓ pdfplumber extraction
raw/extracted_text.txt (200K characters)
    ↓ LLM structuring (Llama 3.1 8B)
guidelines/ (20 structured *.txt files)
    ↓ chunking + embedding
indexes/faiss_index.bin (205 chunks, 384 dim)
indexes/documents.pkl (metadata)
```

**Script:**

```bash
# Rebuild all indexes from guidelines
python scripts/rebuild_index.py
```

### 2. Question Generation Pipeline (Guidelines → Dataset)

```
guidelines/ (20 *.txt files)
    ↓ ClinicalCaseGeneratorV5
    50 clinical case templates
    ↓ LLM enhancement + quality checks
processed/questions/questions_1.json (50 questions)
    ↓ Cryptographic balancing
Balanced dataset (A:38%, B:20%, C:24%, D:14%, Cannot answer:4%)
```

**Script:**

```bash
# Generate 50 clinical cases (current evaluation set)
python scripts/generate_clinical_cases_v5.py 50

# Output: data/processed/questions/questions_1.json
```

### 3. Evaluation Pipeline (Dataset → Performance Metrics)

```
questions_1.json (50 questions)
    ↓ Multi-stage retrieval (MAP: 0.268)
    ↓ Hybrid reasoning (Accuracy: 52%)
reports/evaluation_results.json
reports/charts/performance_summary.png
reports/charts/confusion_matrix.png
```

**Script:**

```bash
# Evaluate on 50 cases
python scripts/evaluate_new_dataset.py --num-cases 50

# Output includes:
# - Accuracy: 52%
# - MAP: 0.268
# - Medical Concept Coverage: 75.1%
# - Guideline Coverage: 100%
```

---

## Data Quality Metrics

### Current Performance (Based on 50-case Evaluation)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Accuracy | 52% | 80% | ⚠️ Needs improvement |
| Medical Concept Coverage | 75.1% | 90% | ⚠️ Moderate |
| Guideline Coverage | 100% | 100% | ✅ Excellent |
| Hallucination Rate | 0.0% | 0.0% | ✅ Perfect |
| Safety Score | 0.96 | 1.0 | ⚠️ Good |

### Identified Data Issues (From 52% Accuracy Analysis)

1. **Medical Terminology Coverage (75.1%)**
   - **Issue:** 24.9% of medical concepts not covered
   - **Impact:** 48% of cases had terminology misunderstandings
   - **Solution:** Expand UMLS mappings and concept coverage

2. **Specialty Performance Variation**
   - **Best:** Critical Care (100%), Gastroenterology (71.4%)
   - **Worst:** Infectious Disease (0%), Neurology (0%)
   - **Solution:** Enhance data for low-performing specialties

3. **Question Type Imbalance**
   - **Current:** 92% diagnosis, 4% treatment, 4% other
   - **Target:** 60% diagnosis, 30% treatment, 10% management
   - **Solution:** Generate more treatment and management questions

4. **Difficulty Distribution**
   - **Simple:** 24% (58.3% accuracy)
   - **Moderate:** 50% (52% accuracy)
   - **Complex:** 26% (46.2% accuracy)
   - **Solution:** Add more complex cases to challenge the system

---

## Data Versioning

### Current Version: v1.0 (50 Questions)

**Characteristics:**
- **Size:** 50 clinical cases
- **Accuracy Baseline:** 52% system accuracy
- **Coverage:** 11 medical specialties
- **Generation:** ClinicalCaseGeneratorV5 with cryptographic balancing

**Metadata:**

```json
{
  "version": "1.0",
  "generation_date": "2025-12-11",
  "total_questions": 50,
  "system_accuracy": 0.52,
  "retrieval_map": 0.268,
  "medical_concept_coverage": 0.751,
  "guideline_coverage": 1.0,
  "hallucination_rate": 0.0,
  "safety_score": 0.96
}
```

### Planned Versions

**v2.0 (Target: 100 questions)**
- Expand to 1000+ medical concepts (target: 90% coverage)
- Balance specialty distribution
- Add more treatment/management questions

**v3.0 (Target: 200 questions)**
- Include rare/edge cases
- Add imaging/lab interpretation questions
- Include multi-system cases

**v4.0 (Target: 500 questions)**
- Comprehensive coverage of all major specialties
- Include procedural/surgical questions
- Add pediatric/geriatric-specific cases

---

## Usage Examples

### Loading the Dataset

```python
import json

# Load clinical questions
with open("data/processed/questions/questions_1.json", "r") as f:
    dataset = json.load(f)

print(f"Total questions: {dataset['metadata']['total_cases']}")
print(f"System accuracy: {dataset['metadata'].get('system_accuracy', 'N/A')}")

# Access first question
question = dataset["questions"][0]
print(f"Question ID: {question['question_id']}")
print(f"Specialty: {question['specialty']}")
print(f"Correct answer: {question['correct_answer']}")
```

### Accessing Guidelines

```python
# Load a specific guideline
with open("data/guidelines/guideline_01_cardiovascular_emergencies_acs.txt", "r") as f:
    guideline_content = f.read()

# Parse structured sections
lines = guideline_content.split("\n")
for line in lines:
    if line.startswith("GUIDELINE:"):
        print(f"Guideline: {line.replace('GUIDELINE:', '').strip()}")
    elif line.startswith("CATEGORY:"):
        print(f"Category: {line.replace('CATEGORY:', '').strip()}")
```

### Using UMLS Mappings

```python
import json

# Load UMLS synonyms
with open("data/umls_synonyms.json", "r") as f:
    umls_synonyms = json.load(f)

# Expand medical term
term = "ACS"
if term in umls_synonyms:
    synonyms = umls_synonyms[term]["synonyms"]
    print(f"Synonyms for {term}: {synonyms}")
    # Output: ["acute coronary syndrome", "heart attack", "myocardial infarction"]
```

---

## Related Documentation

- **Data Creation Documentation** - How the dataset was created
- **Part 1: Dataset Creation** - Case generation methodology
- **Experimental Results** - 52% accuracy evaluation
- **Error Analysis** - Data-related issues identified

---

## Maintenance and Updates

### Rebuilding FAISS Index

```bash
# After updating guidelines, rebuild FAISS index
python scripts/rebuild_index.py
```

### Adding New Guidelines

1. Add new `.txt` file to `data/guidelines/` following the template
2. Run `python scripts/rebuild_index.py`
3. The new guideline will be included in semantic retrieval

### Expanding UMLS Mappings

1. Edit `data/umls_synonyms.json` and `data/umls_expansion.json`
2. No need to rebuild indexes (loaded at runtime)

### Generating More Questions

```bash
# Generate additional questions
python scripts/generate_clinical_cases_v5.py 100 --seed 123

# Will create: data/processed/questions/questions_123.json
```

---

**Documentation Author:** Shreya Uprety  
**Dataset Version:** 1.0 (50 questions, 52% accuracy baseline)  
**Medical Concept Coverage:** 75.1%  
**Guideline Coverage:** 100%  
**Last Updated:** Based on evaluation results (2025-12-11)