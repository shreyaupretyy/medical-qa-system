# Medical Question-Answering System

**Author:** Shreya Uprety  
**Repository:** https://github.com/shreyaupretyy/medical-qa-system

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Core Components](#core-components)
5. [Reasoning Methods](#reasoning-methods)
6. [Retrieval Strategies](#retrieval-strategies)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Dataset Generation](#dataset-generation)
9. [Experimental Results](#experimental-results)
10. [Installation and Setup](#installation-and-setup)
11. [Usage](#usage)
12. [Project Structure](#project-structure)
13. [Documentation](#documentation)
14. [License](#license)

---

## Introduction

The Medical Question-Answering System is a sophisticated Retrieval-Augmented Generation (RAG) pipeline designed to answer multiple-choice clinical questions with high accuracy and reliability. The system combines advanced information retrieval, multi-stage reasoning, and medical domain expertise to provide evidence-based answers grounded in clinical guidelines.

This system addresses critical challenges in medical question answering:

- Accurate retrieval of relevant clinical guidelines from large knowledge bases
- Multi-hop reasoning across complex medical concepts
- Confidence calibration and hallucination detection
- Safety verification for clinical recommendations
- Explainable decision-making with reasoning chains

The architecture supports multiple reasoning strategies, including Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Structured Medical Reasoning, with optional LangChain/LangGraph integration for flexible workflow orchestration.

---

## System Overview

The Medical Question-Answering System operates through a multi-stage pipeline:

1. **Query Understanding**: Extracts clinical features, identifies medical specialties, and expands queries with medical concepts
2. **Multi-Stage Retrieval**: Combines BM25, semantic search, and concept-based retrieval with cross-encoder reranking
3. **Context Processing**: Prunes irrelevant information and prioritizes guideline-based evidence
4. **Reasoning Engine**: Applies Chain-of-Thought, Tree-of-Thought, or Structured Medical Reasoning
5. **Safety and Quality Verification**: Detects hallucinations, verifies medical safety, and calibrates confidence
6. **Answer Selection**: Selects the most appropriate answer with supporting evidence

### Key Features

- **Multi-Stage Retrieval**: Combines lexical (BM25) and semantic (FAISS) retrieval with medical concept expansion
- **Hybrid Reasoning**: Supports CoT, ToT, and 5-step structured medical reasoning
- **Safety-First Design**: Includes hallucination detection and medical safety verification
- **Comprehensive Evaluation**: Tracks accuracy, retrieval quality, reasoning coherence, and calibration metrics
- **LangChain Integration**: Optional workflow orchestration without affecting core pipeline accuracy
- **Condition-Level Error Analysis**: Identifies confusion between similar medical conditions

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INPUT CLINICAL QUESTION                         │
│                 (Case Description + Question)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      QUERY UNDERSTANDING                            │
│  • Clinical Feature Extraction                                      │
│  • Medical Concept Expansion (UMLS)                                 │
│  • Specialty Detection (Cardiology, Neurology, etc.)                │
│  • Symptom Synonym Injection                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MULTI-STAGE RETRIEVAL                          │
│                                                                     │
│  Stage 1: Broad Retrieval (k=150)                                  │
│    ├─ BM25 (Lexical matching)                                      │
│    ├─ FAISS (Semantic search)                                      │
│    └─ Concept-First (Medical concept expansion)                    │
│                                                                     │
│  Stage 2: Focused Retrieval (k=100)                                │
│    ├─ Multi-Query Expansion                                        │
│    ├─ Symptom-Enhanced Queries                                     │
│    └─ Guideline Prioritization                                     │
│                                                                     │
│  Stage 3: Reranking (k=30)                                         │
│    ├─ Cross-Encoder Reranking                                      │
│    ├─ Context Pruning                                              │
│    └─ Evidence Quality Scoring                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        REASONING ENGINE                             │
│                                                                     │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐   │
│  │ Chain-of-Thought │ │ Tree-of-Thought  │ │   Structured     │   │
│  │   (Primary)      │ │    (Complex)     │ │Medical Reasoning │   │
│  │  34% accuracy    │ │  52% accuracy    │ │  44% accuracy    │   │
│  │  4,955ms avg     │ │  41,367ms avg    │ │  26,991ms avg    │   │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘   │
│                                                                     │
│  3-Stage Hybrid Pipeline:                                          │
│    1. CoT (fast, general cases)                                    │
│    2. ToT (if complex AND confidence < 0.75)                       │
│    3. Structured (fallback with LLM enhancement)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SAFETY AND QUALITY VERIFICATION                        │
│  • Hallucination Detection                                          │
│  • Medical Safety Verification                                      │
│  • Confidence Calibration                                           │
│  • Terminology Normalization                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ANSWER SELECTION                              │
│  • Selected Answer (A/B/C/D)                                        │
│  • Confidence Score                                                 │
│  • Reasoning Chain                                                  │
│  • Supporting Evidence                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Dataset Creation

The system includes tools for generating high-quality clinical question datasets:

- **PDF Extractor**: Extracts medical guidelines from PDF documents
- **Guideline Generator**: Structures raw text into formatted clinical guidelines
- **Question Generator**: Creates multiple-choice questions from guidelines
- **Clinical Case Generator v5**: Generates realistic clinical scenarios with controlled answer distribution

See [Part 1: Dataset Creation](docs/part_1_dataset_creation.md) for details.

### 2. Retrieval Pipeline

Multi-stage retrieval combining multiple strategies:

- **BM25 Retriever**: Lexical matching for keyword-based retrieval
- **FAISS Store**: Semantic search using sentence embeddings
- **Concept-First Retriever**: Medical concept expansion using UMLS
- **Hybrid Retriever**: Combines lexical and semantic scores
- **Multi-Stage Retriever**: Sequential refinement with reranking

See [Part 2: RAG Implementation](docs/part_2_rag_implementation.md) for details.

### 3. Reasoning Engine

Three reasoning methods with hybrid orchestration:

- **Chain-of-Thought (CoT)**: Step-by-step logical reasoning
- **Tree-of-Thought (ToT)**: Multi-branch exploration for complex questions
- **Structured Medical Reasoning**: 5-step clinical decision framework

See [Reasoning Documentation](docs/reasoning_documentation.md) for details.

### 4. Evaluation Framework

Comprehensive metrics tracking:

- **Accuracy Metrics**: Exact match, semantic similarity
- **Retrieval Metrics**: Precision@k, Recall@k, MAP, MRR
- **Reasoning Metrics**: Chain completeness, evidence utilization
- **Calibration Metrics**: Brier score, Expected Calibration Error
- **Safety Metrics**: Hallucination rate, dangerous error count

See [Part 3: Evaluation Framework](docs/part_3_evaluation_framework.md) for details.

---

## Reasoning Methods

### Chain-of-Thought (CoT)

Linear step-by-step reasoning that builds a logical chain from evidence to conclusion.

**Performance:**
- **Accuracy: 34%**
- **Average Time: 4,955ms**
- Best for: Straightforward clinical scenarios

**Process:**
1. Analyze clinical presentation
2. Extract key symptoms and findings
3. Consider differential diagnoses
4. Apply guideline recommendations
5. Select most appropriate answer

### Tree-of-Thought (ToT)

Multi-branch reasoning exploring multiple diagnostic pathways simultaneously.

**Performance:**
- **Accuracy: 52% (highest)**
- **Average Time: 41,367ms**
- Best for: Complex multi-system cases

**Process:**
1. Generate multiple reasoning branches
2. Explore each diagnostic possibility
3. Evaluate evidence for each branch
4. Prune unlikely pathways
5. Select highest-confidence conclusion

### Structured Medical Reasoning

5-step clinical decision framework with LLM enhancement.

**Performance:**
- **Accuracy: 44%**
- **Average Time: 26,991ms**
- **Best Calibration: Brier score 0.295, ECE 0.283**
- Best for: Systematic clinical evaluation

**Process:**
1. **Patient Profile**: Extract demographics, symptoms, vitals
2. **Differential Diagnosis**: Generate possible conditions (LLM-enhanced)
3. **Evidence Analysis**: Match symptoms to conditions
4. **Guideline Application**: Apply clinical guidelines
5. **Final Decision**: Select answer with confidence score
6. **LLM Verification**: Validate reasoning coherence

### 3-Stage Hybrid Pipeline

The system uses a cascading approach:

1. **Primary**: CoT reasoning (fast, general cases)
2. **Escalation**: If question is complex AND confidence < 0.75, use ToT
3. **Fallback**: If CoT/ToT unavailable, use Structured Medical Reasoning

---

## Retrieval Strategies

### BM25 (Lexical Retrieval)

Traditional term-frequency based retrieval.

**Strengths:**
- Excellent for exact medical term matching
- Fast retrieval (milliseconds)
- No embedding computation required

**Limitations:**
- Misses semantic relationships
- Sensitive to vocabulary mismatch

**Performance (Single-Stage BM25):**
- **MAP: 0.207**
- **MRR: 0.414**
- **Precision@5: 17.4%**
- **Recall@5: 43.5%**
- **Time: 1.40ms (fastest)**

### FAISS (Semantic Retrieval)

Vector similarity search using sentence embeddings.

**Strengths:**
- Captures semantic meaning
- Finds conceptually related content
- Robust to paraphrasing

**Current Model:**
- sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- GPU-accelerated with FP16 precision
- 205 indexed guideline chunks

**Performance (Single-Stage FAISS):**
- **MAP: 0.211**
- **MRR: 0.422**
- **Precision@5: 17.6%**
- **Recall@5: 44.0%**
- **Time: 8.58ms**

### Concept-First Retrieval

Expands queries with medical concepts from UMLS.

**Process:**
1. Extract medical entities from query
2. Expand with UMLS synonyms and related concepts
3. Retrieve using expanded query
4. Score by concept coverage

**Example:**
- Query: "chest pain"
- Expanded: "chest pain, angina, myocardial ischemia, coronary syndrome"

**Performance:**
- **MAP: 0.212**
- **MRR: 0.424**
- **Precision@5: 18.0%**
- **Recall@5: 45.0% (best recall)**
- **Time: 11.62ms**

### Hybrid Retrieval

Combines BM25 and semantic scores with weighted fusion.

**Formula:**
```
hybrid_score = α × bm25_score + (1-α) × semantic_score
```

**Parameters:**
- α = 0.5 (equal weighting)
- Adjustable based on query type

**Performance (Hybrid Linear):**
- **MAP: 0.211**
- **MRR: 0.421**
- **Precision@5: 17.8%**
- **Recall@5: 44.5%**
- **Time: 8.33ms**

### Multi-Stage Retrieval

Three-stage pipeline with progressive refinement:

**Stage 1: Broad Retrieval (k=150)**
- Cast wide net with BM25 + FAISS + Concept-First
- Maximize recall

**Stage 2: Focused Retrieval (k=100)**
- Multi-query expansion
- Symptom synonym injection
- Guideline prioritization

**Stage 3: Reranking (k=30)**
- Cross-encoder reranking
- Context pruning
- Final evidence selection

**Performance:**
- **MAP: 0.204**
- **MRR: 0.408**
- **Precision@5: 17.0%**
- **Recall@5: 42.5%**
- **Time: 2,878ms**

### Semantic-First Retrieval

FAISS retrieval with BM25 reranking.

**Performance:**
- **MAP: 0.213 (best MAP)**
- **MRR: 0.425 (best MRR)**
- **Precision@5: 17.8%**
- **Recall@5: 44.5%**
- **Time: 9.65ms**

---

## Evaluation Metrics

### Accuracy Metrics

**Exact Match Accuracy**
- Binary correct/incorrect evaluation
- **Current: 52%**

**Semantic Accuracy**
- Measures answer similarity even if not exact match
- **Current: 52%** (same as exact match)
- Allows partial credit for close answers

### Retrieval Metrics

**Precision@k**
- Percentage of retrieved documents that are relevant
- **Current:**
  - Precision@1: 0.0%
  - Precision@3: 10.7%
  - Precision@5: 11.2%
  - Precision@10: 7.98%

**Recall@k**
- Percentage of relevant documents retrieved
- **Current:**
  - Recall@1: 0.0%
  - Recall@3: 32.0%
  - Recall@5: 56.0%
  - Recall@10: 78.0%

**Mean Average Precision (MAP)**
- Average precision across all queries
- **Current: 0.268**

**Mean Reciprocal Rank (MRR)**
- Average of reciprocal ranks of first relevant document
- **Current: 0.268**

**Context Relevance Score**
- Measures how relevant retrieved context is to the question
- Scale: 0 (irrelevant) to 2 (highly relevant)
- **Distribution:** 0.0 (40%), 1.0 (15%), 2.0 (45%)

**Medical Concept Coverage**
- Percentage of medical concepts covered by retrieved documents
- **Current: 75.1%**

**Guideline Coverage**
- Percentage of guidelines referenced in retrieved documents
- **Current: 100%**

### Reasoning Metrics

**Chain Completeness**
- Measures if all reasoning steps are present
- **Current: 100%** (all methods produce complete chains)

**Evidence Utilization Rate**
- Percentage of retrieved evidence used in reasoning
- **Current: 100%**

**Confidence Distribution**
- Tracks prediction confidence ranges
- **Current Distribution:**
  - 90-100%: 7 cases
  - 40-50%: 11 cases
  - 0-10%: 5 cases
  - 30-40%: 13 cases
  - 70-80%: 2 cases
  - 20-30%: 4 cases
  - 10-20%: 7 cases
  - 80-90%: 1 case

### Calibration Metrics

**Brier Score**
- Measures accuracy of probabilistic predictions
- Lower is better (0 = perfect calibration)
- Formula: `BS = (1/N) Σ(p - y)²`
- **Current: 0.254**

**Expected Calibration Error (ECE)**
- Measures difference between confidence and accuracy
- **Current: 0.179**

### Safety Metrics

**Hallucination Rate**
- Percentage of answers not grounded in context
- **Current: 0.0%** (strict prompting successful)

**Dangerous Error Count**
- Answers that could lead to patient harm
- **Current: 2**

**Safety Score**
- Composite safety metric
- **Current: 0.96**

**Contraindication Check Accuracy**
- Accuracy of checking for contraindications
- **Current: 0.0%**

**Urgency Recognition Accuracy**
- Accuracy of recognizing urgent/emergent conditions
- **Current: 0.0%**

### Confusion Matrix

**Answer-Level Confusion Matrix**
- Tracks predicted vs true answers (A/B/C/D)
- Visualized with green-bordered diagonal for correct predictions

**Confusion Matrix Results:**
- **Option A:** Precision: 47.6%, Recall: 90.9%, F1: 62.5%
- **Option B:** Precision: 40.0%, Recall: 30.8%, F1: 34.8%
- **Option C:** Precision: 75.0%, Recall: 64.3%, F1: 69.2%
- **Option D:** Precision: 71.4%, Recall: 41.7%, F1: 52.6%

**Macro Averages:**
- **Macro Precision: 58.5%**
- **Macro Recall: 56.9%**
- **Macro F1: 54.8%**
- **Balanced Accuracy: 56.9%**

**Condition-Level Confusion Analysis**
- Identifies confusion between similar medical conditions
- Tracks:
  - Similar-condition errors (same medical group)
  - Unrelated-condition errors (different groups)
  - Top confusion pairs

**Medical Similarity Groups:**
- Respiratory infections, Cardiac ischemic, Thrombotic, Infections, etc.
- 16 predefined medical condition groups

### Error Analysis

**Error Categories:**
- **Reasoning Errors (16 cases, 32%):** System retrieved relevant info but made incorrect reasoning
- **Knowledge Errors (8 cases, 16%):** System has incorrect medical knowledge or interpretation
- **Retrieval failures: 0%** (all relevant documents successfully retrieved)

**Error Breakdown by Medical Domain:**
- **Cardiovascular:** 54.5% accuracy (6/11 correct)
- **Gastroenterology:** 71.4% accuracy (5/7 correct)
- **Respiratory:** 62.5% accuracy (5/8 correct)
- **Endocrine:** 66.7% accuracy (4/6 correct)
- **Infectious Disease:** 0% accuracy (0/3 correct)
- **Neurology:** 0% accuracy (0/2 correct)

**Common Pitfalls:**
- **Overconfident Wrong Answers (2 cases):** High confidence (>80%) but incorrect answers
- **Missing Critical Symptoms (20 cases):** Reasoning fails to consider important symptoms from case description
- **Medical Terminology Misunderstanding (24 cases):** System fails to properly interpret medical abbreviations or terms

---

## Dataset Generation

### Guideline Extraction

The system processes medical treatment guidelines through a structured pipeline:

1. **PDF Extraction**: Extracts text from standard-treatment-guidelines.pdf
2. **Topic Identification**: Identifies 20 clinical topics across categories
3. **LLM Structuring**: Uses Ollama (Llama 3.1 8B) to organize content
4. **Guideline Generation**: Creates structured guideline documents

**Output Format:**
```
GUIDELINE: [Topic Name]
CATEGORY: [Medical Category]

DEFINITION:
[Condition definition and epidemiology]

DIAGNOSIS:
[Diagnostic criteria and assessment]

TREATMENT:
[Evidence-based treatment protocols]

MANAGEMENT:
[Long-term management strategies]
```

### Clinical Case Generation

**Generator Features:**
- Realistic vital sign constraints
- Balanced answer distribution
- Varied clinical presentations
- Controlled difficulty levels (simple/moderate/complex)
- Multiple question types (diagnosis/treatment/other)

**Generation Process:**
1. Select clinical guideline
2. Generate realistic vital signs
3. Create case scenario with LLM
4. Generate 4 plausible options
5. Apply cryptographic shuffling for answer balance
6. Validate clinical consistency



## Experimental Results

### Retrieval Strategy Comparison

**Experiment:** Compared 6 retrieval strategies on 100 clinical cases

**Results:**

| Strategy | MAP | MRR | P@5 | R@5 | Time (ms) |
|----------|-----|-----|-----|-----|-----------|
| Single BM25 | 0.207 | 0.414 | 17.4% | 43.5% | **1.40** |
| Single FAISS | 0.211 | 0.422 | 17.6% | 44.0% | 8.58 |
| Hybrid Linear | 0.211 | 0.421 | 17.8% | 44.5% | 8.33 |
| Concept-First | 0.212 | 0.424 | 18.0% | **45.0%** | 11.62 |
| **Semantic-First** | **0.213** | **0.425** | 17.8% | 44.5% | 9.65 |
| Multi-Stage | 0.204 | 0.408 | 17.0% | 42.5% | 2,878 |

**Key Findings:**
- **Semantic-First** achieves best MAP (0.213) and MRR (0.425)
- **Concept-First** achieves best recall (45.0%)
- **Single BM25** is fastest (1.40ms)
- All strategies benefit from medical concept expansion

### Reasoning Method Comparison

**Experiment:** Evaluated 50 clinical cases across CoT, ToT, and Structured Medical Reasoning

**Results:**

| Method | Accuracy | Avg Time (ms) | Brier Score | ECE | Best For |
|--------|----------|---------------|-------------|-----|----------|
| Chain-of-Thought | 34% | **4,955** | 0.424 | 0.266 | General cases |
| **Tree-of-Thought** | **52%** | 41,367 | 0.344 | 0.310 | Complex scenarios |
| Structured Medical | 44% | 26,991 | **0.295** | **0.283** | Systematic evaluation |

**Key Findings:**
- **Tree-of-Thought** achieves highest accuracy (52%) but is 8x slower than CoT
- **Structured Medical** provides best calibration (Brier: 0.295, ECE: 0.283)
- **Chain-of-Thought** is fastest (4,955ms) but lowest accuracy (34%)
- Hybrid pipeline balances speed and accuracy

### Performance by Specialty

**Results (50 cases across 11 specialties):**

| Specialty | Accuracy | Cases | Correct/Total |
|-----------|----------|-------|---------------|
| Critical Care | 100% | 1 | 1/1 |
| Gastroenterology | 71.4% | 7 | 5/7 |
| Endocrine | 66.7% | 6 | 4/6 |
| Nephrology | 66.7% | 3 | 2/3 |
| Respiratory | 62.5% | 8 | 5/8 |
| Cardiovascular | 54.5% | 11 | 6/11 |
| Rheumatology | 33.3% | 3 | 1/3 |
| Hematology | 33.3% | 3 | 1/3 |
| Psychiatry | 33.3% | 3 | 1/3 |
| Infectious Disease | 0% | 3 | 0/3 |
| Neurology | 0% | 2 | 0/2 |

**Key Insights:**
- System excels in Critical Care, Gastroenterology, and Endocrine
- Struggles with Infectious Disease and Neurology
- Performance varies significantly by medical domain

### Error Analysis

**Top Confusion Pairs:**
1. Heart Failure → Angina
2. DVT → Pulmonary Embolism
3. Gastritis → GI Bleed

**Error Distribution:**
- **Reasoning errors:** 16 cases (32%)
- **Knowledge errors:** 8 cases (16%)
- **Retrieval errors:** 0 cases (0%)

**Root Causes:**
1. Incomplete differential diagnosis
2. Missing critical symptoms in reasoning
3. Medical terminology misunderstanding
4. Overconfident predictions

### Hallucination Prevention

**Experiment:** Tested strict prompting to prevent LLM memory usage

**Results:**
- **Hallucination Rate: 0.0%**
- Method: Explicit instructions to use only provided context
- Verification: Tested on all 50 cases with hallucination detection

### Calibration Analysis

**Findings:**
- **Brier Score: 0.254** (moderate calibration)
- **ECE: 0.179** (reduced by 42% with calibration)
- Issue: Model overconfident on incorrect predictions
- Solution: Implemented confidence calibration module

---

## Installation and Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster embeddings)
- Ollama with Llama 3.1 8B model

### Installation

```bash
# Clone repository
git clone https://github.com/shreyaupretyy/medical-qa-system.git
cd medical-qa-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support
pip install -r requirements-gpu.txt

# Install Ollama and download model
# Visit https://ollama.ai for installation instructions
ollama pull llama3.1:8b
```

### Configuration

Edit `config/pipeline_config.yaml` to customize:

```yaml
retrieval:
  k_stage1: 150
  k_stage2: 100
  k_stage3: 30
  bm25_weight: 0.5
  semantic_weight: 0.5

reasoning:
  method: hybrid  # Options: cot, tot, structured, hybrid
  temperature: 0.0
  max_tokens: 512

evaluation:
  enable_hallucination_detection: true
  enable_safety_verification: true
  enable_confidence_calibration: true
```

---

## Usage

### Evaluate the System

```bash
# Evaluate on 50 cases (full dataset)
python scripts/evaluate_new_dataset.py --num-cases 50

# Evaluate on subset of cases
python scripts/evaluate_new_dataset.py --num-cases 10 --seed 42

# Custom dataset
python scripts/evaluate_new_dataset.py --dataset path/to/questions.json
```

**Output:**
- Detailed metrics in `reports/new_dataset_eval_N_cases.json`
- Confusion matrix in `reports/charts/confusion_matrix.png`
- Performance summary in `reports/charts/performance_summary.png`
- Error analysis in `reports/charts/error_analysis.png`

### Compare Reasoning Methods

```bash
# Compare CoT, ToT, and Structured on 50 cases
python scripts/compare_reasoning_methods.py --num-cases 50
```

**Output:** `reports/reasoning_method_comparison.json`

### Compare Retrieval Strategies

```bash
# Compare all 6 retrieval strategies
python scripts/compare_retrieval_strategies.py
```

**Output:** `reports/retrieval_strategy_comparison.json`

### Generate New Clinical Cases

```bash
# Generate 50 cases
python scripts/generate_clinical_cases_v5.py 50

# Output: data/processed/questions/questions_1.json
```

### Rebuild FAISS Index

```bash
# Rebuild index from guidelines
python scripts/rebuild_index.py
```

### Generate Guidelines from PDF

```bash
# Extract and generate guidelines from PDF
python scripts/generate_from_pdf.py

# Or specify custom PDF
python scripts/generate_from_pdf.py path/to/guidelines.pdf
```

### LangChain Integration Example

```python
from examples.langchain_integration_example import MedicalQALangChain

# Initialize
qa_system = MedicalQALangChain()

# Ask question
result = qa_system.answer_question(
    case_description="58-year-old man with chest pain...",
    question="What is the most appropriate next step?"
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['reasoning']}")
```

---



## Project Structure

```
medical-qa-system/
│
├── config/
│   └── pipeline_config.yaml          # System configuration
│
├── data/
│   ├── guidelines/                    # Structured clinical guidelines
│   │   ├── guideline_01_cardiovascular_emergencies_acs.txt
│   │   ├── guideline_02_stroke_management_ischemic.txt
│   │   └── ... (20 guidelines)
│   │
│   ├── indexes/                       # FAISS and BM25 indexes
│   │   ├── faiss_index.bin
│   │   └── documents.pkl
│   │
│   ├── processed/
│   │   └── questions/
│   │       └── questions_1.json       # Generated clinical cases
│   │
│   ├── raw/
│   │   └── extracted_text.txt         # Raw PDF extraction
│   │
│   ├── standard-treatment-guidelines.pdf
│   ├── umls_synonyms.json            # Medical synonym mappings
│   └── umls_expansion.json           # Concept expansion rules
│
├── src/
│   ├── data_creation/                # Dataset generation tools
│   │   ├── pdf_extractor.py
│   │   ├── guideline_generator.py
│   │   └── question_generator.py
│   │
│   ├── retrieval/                    # Multi-stage retrieval
│   │   ├── bm25_retriever.py
│   │   ├── faiss_store.py
│   │   ├── concept_first_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── multi_stage_retriever.py
│   │
│   ├── reasoning/                    # Reasoning engines
│   │   ├── rag_pipeline.py           # Main RAG pipeline
│   │   ├── medical_reasoning.py      # CoT reasoning
│   │   ├── tree_of_thought.py        # ToT reasoning
│   │   └── query_understanding.py
│   │
│   ├── improvements/                 # Optional enhancements
│   │   ├── clinical_feature_extractor.py
│   │   ├── confidence_calibrator.py
│   │   ├── context_pruner.py
│   │   ├── deterministic_reasoner.py
│   │   ├── hallucination_detector.py
│   │   ├── medical_concept_expander.py
│   │   ├── safety_verifier.py
│   │   ├── structured_reasoner.py
│   │   └── ... (19 improvement modules)
│   │
│   ├── evaluation/                   # Evaluation framework
│   │   ├── pipeline.py
│   │   ├── metrics_calculator.py
│   │   ├── analyzer.py
│   │   ├── condition_confusion_analyzer.py
│   │   └── visualizer.py
│   │
│   ├── langchain_integration/        # LangChain/LangGraph wrappers
│   │   ├── wrappers.py
│   │   ├── graph.py
│   │   └── README.md
│   │
│   ├── models/                       # Model interfaces
│   │   ├── embeddings.py
│   │   └── ollama_model.py
│   │
│   ├── optimization/                 # Parameter tuning
│   │   ├── retrieval_tuner.py
│   │   ├── parameter_optimizer.py
│   │   └── symptom_extractor.py
│   │
│   ├── specialties/                  # Specialty handlers
│   │   └── obgyn_handler.py
│   │
│   └── utils/
│       └── config_loader.py
│
├── scripts/                          # Execution scripts
│   ├── evaluate_new_dataset.py
│   ├── compare_reasoning_methods.py
│   ├── compare_retrieval_strategies.py
│   ├── generate_clinical_cases_v5.py
│   ├── generate_from_pdf.py
│   └── rebuild_index.py
│
├── examples/
│   └── langchain_integration_example.py
│
├── reports/                          # Evaluation results
│   ├── charts/
│   │   ├── confusion_matrix.png
│   │   ├── performance_summary.png
│   │   └── error_analysis.png
│   │
│   ├── evaluation_results.json
│   ├── reasoning_method_comparison.json
│   ├── retrieval_strategy_comparison.json
│   └── condition_confusion_50_cases.json
│
├── tests/                            # Unit tests
│   ├── test_multi_stage_rag.py
│   └── test_reasoning_template.py
│
├── docs/                             # Documentation
│   ├── config_documentation.md
│   ├── data_documentation.md
│   ├── data_creation_documentation.md
│   ├── retrieval_documentation.md
│   ├── reasoning_documentation.md
│   ├── improvements_documentation.md
│   ├── evaluation_documentation.md
│   ├── models_documentation.md
│   ├── optimization_documentation.md
│   ├── langchain_integration_documentation.md
│   ├── scripts_documentation.md
│   ├── reports_documentation.md
│   ├── part_1_dataset_creation.md
│   ├── part_2_rag_implementation.md
│   ├── part_3_evaluation_framework.md
│   └── part_4_experiments.md
│
├── requirements.txt
├── requirements-gpu.txt
└── README.md
```

---

## Documentation

Detailed documentation is available in the `docs/` directory:

### Component Documentation

- [Configuration Documentation](docs/config_documentation.md) - Pipeline configuration and settings
- [Data Documentation](docs/data_documentation.md) - Dataset structure and formats
- [Data Creation Documentation](docs/data_creation_documentation.md) - Guideline and question generation
- [Retrieval Documentation](docs/retrieval_documentation.md) - Multi-stage retrieval strategies
- [Reasoning Documentation](docs/reasoning_documentation.md) - CoT, ToT, and Structured reasoning
- [Improvements Documentation](docs/improvements_documentation.md) - Optional enhancement modules
- [Evaluation Documentation](docs/evaluation_documentation.md) - Metrics and analysis
- [Models Documentation](docs/models_documentation.md) - Embedding and LLM interfaces
- [Optimization Documentation](docs/optimization_documentation.md) - Parameter tuning
- [LangChain Integration Documentation](docs/langchain_integration_documentation.md) - Workflow orchestration
- [Scripts Documentation](docs/scripts_documentation.md) - Execution scripts
- [Reports Documentation](docs/reports_documentation.md) - Result interpretation

### Part-Wise Documentation

- [Part 1: Dataset Creation](docs/part_1_dataset_creation.md) - Case generation methodology
- [Part 2: RAG Implementation](docs/part_2_rag_implementation.md) - Multi-stage retrieval design
- [Part 3: Evaluation Framework](docs/part_3_evaluation_framework.md) - Comprehensive metrics
- [Part 4: Experiments](docs/part_4_experiments.md) - Experimental results and analysis

---

## LangChain and LangGraph Integration

The system includes optional LangChain/LangGraph integration that provides:

- **Workflow Orchestration**: Define complex reasoning workflows as graphs
- **State Management**: Track intermediate states across pipeline stages
- **Modularity**: Swap components without rewriting core logic
- **Observability**: Built-in logging and debugging

**Key Design Principle:**
LangChain integration is **non-intrusive** - it wraps the existing pipeline without modifying core components. This ensures:
- No impact on evaluation accuracy
- Optional usage (system works standalone)
- Easy comparison between vanilla and LangChain modes

**Integration Points:**
- Retrieval wrapper for document fetching
- Reasoning wrapper for answer generation
- Graph orchestration for multi-step workflows
- State tracking across pipeline stages

See [LangChain Integration Documentation](docs/langchain_integration_documentation.md) for implementation details.

---



## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Add tests for new functionality
5. Update documentation
6. Submit pull request

---

## License

This project is licensed under the MIT License.

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{medical_qa_system_2025,
  author = {Uprety, Shreya},
  title = {Medical Question-Answering System: A Multi-Stage RAG Pipeline},
  year = {2025},
  url = {https://github.com/shreyaupretyy/medical-qa-system}
}
```

---

## Contact

**Author:** Shreya Uprety  
**Repository:** https://github.com/shreyaupretyy/medical-qa-system

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Documentation Author:** Shreya Uprety  