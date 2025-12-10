# Part 1: Dataset Creation

**Author:** Shreya Uprety  
**Last Updated:** December 11, 2025

---

## Overview

This document details the complete dataset creation pipeline for the Medical Question-Answering System, including guideline extraction, structuring, and clinical case generation.

---

## Pipeline Overview

```
Standard Treatment Guidelines PDF
        ↓
PDF Extraction (pdfplumber)
        ↓
Raw Text Processing
        ↓
LLM-Based Guideline Structuring (Ollama Llama 3.1 8B)
        ↓
Structured Clinical Guidelines (20 topics)
        ↓
Clinical Case Generation (LLM + Quality Controls)
        ↓
Multi-Choice Question Dataset (100 cases)
```

---

## Stage 1: PDF Extraction

**Tool:** `src/data_creation/pdf_extractor.py`

### Process

1. **Read PDF:** Uses pdfplumber for text extraction
2. **Clean Text:** Remove headers, footers, page numbers
3. **Section Detection:** Identify topic boundaries
4. **Output:** `data/raw/extracted_text.txt`

### Example Output

```
CARDIOVASCULAR EMERGENCIES

Acute Coronary Syndrome

Acute coronary syndrome (ACS) refers to a spectrum of conditions...
Diagnosis is based on clinical presentation, ECG findings, and troponin levels...
Treatment includes MONA protocol...
```

---

## Stage 2: Guideline Structuring

**Tool:** `src/data_creation/guideline_generator.py`

### Process

1. **Topic Identification:** Extract 20 clinical topics
2. **LLM Structuring:** Use Llama 3.1 8B to organize content into sections
3. **Template Application:** Format as GUIDELINE/CATEGORY/DEFINITION/DIAGNOSIS/TREATMENT/MANAGEMENT
4. **Quality Validation:** Ensure all sections present

### Prompt Template

```
You are a medical expert. Structure the following medical guideline content:

Topic: {topic_name}
Category: {category}

Raw Content:
{raw_text}

Output format:
GUIDELINE: {topic}
CATEGORY: {category}

DEFINITION:
[Provide comprehensive definition]

DIAGNOSIS:
[Provide diagnostic criteria]

TREATMENT:
[Provide treatment protocols]

MANAGEMENT:
[Provide long-term management]
```

### Output Example

```
GUIDELINE: Acute Coronary Syndrome (ACS)
CATEGORY: Cardiovascular Emergencies

DEFINITION:
Acute Coronary Syndrome encompasses a spectrum of conditions caused by sudden 
reduction in coronary blood flow, including unstable angina, NSTEMI, and STEMI.

DIAGNOSIS:
Clinical Presentation:
- Chest pain/pressure lasting >10 minutes
- Pain radiating to jaw, neck, shoulders, arms
ECG: ST-segment elevation (STEMI), ST depression (NSTEMI)
Troponin: Elevated in NSTEMI/STEMI

TREATMENT:
Immediate Management (MONA + Antiplatelet):
- Morphine for pain (2-4mg IV)
- Oxygen if SpO2 <90%
- Nitroglycerin (0.4mg SL)
- Aspirin 325mg chewed

MANAGEMENT:
Secondary Prevention:
- Dual antiplatelet therapy for 12 months
- High-intensity statin
- Beta-blocker
- Cardiac rehabilitation
```

---

## Stage 3: Clinical Case Generation

**Tool:** `scripts/generate_clinical_cases_v5.py`

### Features

1. **Realistic Vital Signs:** Age-appropriate, clinically plausible
2. **Balanced Answer Distribution:** Cryptographic shuffling for 25% A/B/C/D
3. **Quality Controls:** Fever checks, stroke protocol validation, clean options
4. **Difficulty Levels:** Easy (straightforward), Medium (differential needed), Hard (complex)
5. **Question Types:** Diagnosis, Treatment, Management, Immediate Action

### Generation Algorithm

```python
# For each guideline:
for guideline in guidelines:
    # 1. Select clinical scenario
    scenario = extract_scenario(guideline)
    
    # 2. Generate realistic vitals
    vitals = generate_vitals(age, condition)
    
    # 3. Create case description
    case_desc = f"{age}-year-old {gender} presents with {symptoms}. Vitals: {vitals}"
    
    # 4. Generate question and options (LLM)
    question, options = llm.generate(
        prompt=f"Create a clinical question for:\n{case_desc}\n\nGuideline: {guideline}"
    )
    
    # 5. Apply quality checks
    if has_fever(vitals) and "infection" not in options:
        regenerate()
    
    if is_stroke(condition) and not mentions_imaging(options):
        regenerate()
    
    # 6. Cryptographic answer shuffling
    answer_key = hash(question_id) % 4  # A=0, B=1, C=2, D=3
    shuffled_options = shuffle_with_key(options, answer_key)
    
    # 7. Validate answer distribution
    if distribution_imbalanced(all_answers):
        reshuffle_recent_cases()
```

### Quality Checks

**Fever > 101°F:**
- Must consider infectious differentials in options

**Stroke Cases:**
- Must mention CT/MRI before anticoagulation

**Wide Pulse Pressure:**
- Must explain (e.g., aortic regurgitation) or normalize

**Clean Options:**
- No embedded explanations: "A) Aspirin (prevents platelet aggregation)" → "A) Aspirin"

---

## Dataset Quality Metrics

### Answer Distribution

Target: 25% for each option

**Achieved:**
- A: 25% (25/100)
- B: 25% (25/100)
- C: 22% (22/100)
- D: 28% (28/100)

**Method:** Cryptographic shuffling ensures fairness

### Difficulty Distribution

- Easy: 23% (straightforward guidelines)
- Medium: 46% (differential diagnosis required)
- Hard: 31% (complex multi-system)

### Question Type Distribution

- Diagnosis: 23%
- Treatment: 25%
- Management: 27%
- Immediate Action: 25%

### Relevance Distribution

- High: 55% (directly from guidelines)
- Medium: 18% (synthesized from multiple sections)
- Low: 18% (general medical knowledge)

---

## Example Generated Case

```json
{
    "id": "q_001",
    "case_description": "A 58-year-old man presents to the emergency department with sudden onset of severe, crushing chest pain radiating to the left arm. The pain started 45 minutes ago while climbing stairs. He appears diaphoretic and anxious. Vital signs: BP 145/90 mmHg, HR 98 bpm, RR 22/min, SpO2 96% on room air. He has a history of hypertension and hyperlipidemia.",
    "question": "What is the most appropriate next step in management?",
    "options": {
        "A": "Obtain troponin levels and observe",
        "B": "Administer aspirin 325mg and obtain ECG immediately",
        "C": "Start IV fluids and schedule stress test",
        "D": "Give sublingual nitroglycerin only"
    },
    "correct_answer": "B",
    "explanation": "This patient presents with classic symptoms of acute coronary syndrome. The most appropriate immediate action is to administer aspirin (antiplatelet therapy) and obtain an ECG to assess for ST-segment changes. This aligns with the MONA protocol for ACS management.",
    "difficulty": "medium",
    "question_type": "immediate_action",
    "relevance": "high",
    "guideline_source": "guideline_01_cardiovascular_emergencies_acs.txt",
    "medical_concepts": ["acute coronary syndrome", "chest pain", "ECG", "aspirin", "STEMI"],
    "condition": "Acute Coronary Syndrome"
}
```

---

## Generation Scripts

### `generate_from_pdf.py`

Complete pipeline from PDF to questions:

```bash
# Generate guidelines and questions
python scripts/generate_from_pdf.py

# Generate only guidelines
python scripts/generate_from_pdf.py --guidelines-only

# Generate only questions (requires existing guidelines)
python scripts/generate_from_pdf.py --questions-only
```

### `generate_clinical_cases_v5.py`

Generate cases from existing guidelines:

```bash
# Generate 100 cases
python scripts/generate_clinical_cases_v5.py 100

# Generate 50 cases
python scripts/generate_clinical_cases_v5.py 50
```

---

## Data Validation

### Guideline Validation

- **Structure Check:** All sections present (GUIDELINE, CATEGORY, DEFINITION, DIAGNOSIS, TREATMENT, MANAGEMENT)
- **Content Quality:** Minimum 100 tokens per section
- **Medical Accuracy:** Verified against standard treatment protocols

### Question Validation

- **Option Quality:** All 4 options plausible
- **Answer Verification:** Correct answer supported by guideline
- **Explanation Quality:** Clear reasoning chain
- **Clinical Realism:** Vital signs within physiological ranges

---

## Known Limitations

1. **LLM Hallucination:** Rare cases where generated options don't exactly match guideline
2. **Guideline Coverage:** Limited to 20 topics (expandable)
3. **Question Diversity:** Generator sometimes creates similar patterns
4. **Vital Sign Realism:** Some edge cases (e.g., fever + normal WBC)

---

## Future Improvements

1. **Expand Guidelines:** Add 50+ more clinical topics
2. **Multi-Step Cases:** Complex cases requiring sequential reasoning
3. **Image Integration:** Include ECGs, X-rays, CT scans
4. **Differential Diagnosis:** Explicitly track competing diagnoses
5. **Real-World Variation:** Add comorbidities, medication interactions

---

## Related Documentation

- [Data Documentation](data_documentation.md)
- [Data Creation Documentation](data_creation_documentation.md)
- [Retrieval Documentation](retrieval_documentation.md)

---

**Documentation Author:** Shreya Uprety
