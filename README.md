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
│                         INPUT CLINICAL QUESTION                      │
│                    (Case Description + Question)                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      QUERY UNDERSTANDING                             │
│  • Clinical Feature Extraction                                       │
│  • Medical Concept Expansion (UMLS)                                  │
│  • Specialty Detection (Cardiology, Neurology, etc.)                 │
│  • Symptom Synonym Injection                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-STAGE RETRIEVAL                             │
│                                                                       │
│  Stage 1: Broad Retrieval (k=150)                                   │
│    ├─ BM25 (Lexical matching)                                       │
│    ├─ FAISS (Semantic search)                                       │
│    └─ Concept-First (Medical concept expansion)                     │
│                                                                       │
│  Stage 2: Focused Retrieval (k=100)                                 │
│    ├─ Multi-Query Expansion                                         │
│    ├─ Symptom-Enhanced Queries                                      │
│    └─ Guideline Prioritization                                      │
│                                                                       │
│  Stage 3: Reranking (k=30)                                          │
│    ├─ Cross-Encoder Reranking                                       │
│    ├─ Context Pruning                                               │
│    └─ Evidence Quality Scoring                                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      REASONING ENGINE                                │
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │ Chain-of-Thought │  │ Tree-of-Thought  │  │   Structured     │ │
│  │    (Primary)     │  │   (Complex)      │  │Medical Reasoning │ │
│  │   40% accuracy   │  │  50% accuracy    │  │  35% accuracy    │ │
│  │   4,019ms avg    │  │  43,981ms avg    │  │  26,320ms avg    │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│                                                                       │
│  3-Stage Hybrid Pipeline:                                           │
│  1. CoT (fast, general cases)                                       │
│  2. ToT (if complex AND confidence < 0.75)                          │
│  3. Structured (fallback with LLM enhancement)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               SAFETY AND QUALITY VERIFICATION                        │
│  • Hallucination Detection                      │
│  • Medical Safety Verification                                      │
│  • Confidence Calibration                                           │
│  • Terminology Normalization                                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ANSWER SELECTION                                │
│  • Selected Answer (A/B/C/D)                                        │
│  • Confidence Score                                                  │
│  • Reasoning Chain                                                   │
│  • Supporting Evidence                                               │
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
- Accuracy: 40%
- Average Time: 4,019ms
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
- Accuracy: 45-50% (highest)
- Average Time: 43,981ms
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
- Accuracy: 35% (improved from 15% with LLM)
- Average Time: 26,320ms
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

**Known Issue:**
- General-purpose model (not medical-domain optimized)
- Identified as root cause of accuracy drop from 80% to 60%
- Recommendation: Switch to PubMedBERT for medical domain

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

### Hybrid Retrieval

Combines BM25 and semantic scores with weighted fusion.

**Formula:**
```
hybrid_score = α × bm25_score + (1-α) × semantic_score
```

**Parameters:**
- α = 0.5 (equal weighting)
- Adjustable based on query type

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

**Results:**
- Current Performance: Precision@5: 4%, Recall@5: 20%, MAP: 0.118
- Issue: Low precision due to general-purpose embeddings

---

## Evaluation Metrics

### Accuracy Metrics

**Exact Match Accuracy**
- Binary correct/incorrect evaluation
- Current: 60% (target: 80% with medical embeddings)

**Semantic Accuracy**
- Measures answer similarity even if not exact match
- Allows partial credit for close answers

### Retrieval Metrics

**Precision@k**
- Percentage of retrieved documents that are relevant
- Current: Precision@5 = 0.108000

**Recall@k**
- Percentage of relevant documents retrieved
- Current: Recall@5 = 0..54

**Mean Average Precision (MAP)**
- Average precision across all queries
- Current: 0.2523

**Mean Reciprocal Rank (MRR)**
- Average of reciprocal ranks of first relevant document
- Current: 0.2523

**Context Relevance Score**
- Measures how relevant retrieved context is to the question
- Scale: 0 (irrelevant) to 2 (highly relevant)

### Reasoning Metrics

**Chain Completeness**
- Measures if all reasoning steps are present
- Current: 100% (all methods produce complete chains)

**Evidence Utilization Rate**
- Percentage of retrieved evidence used in reasoning
- Current: 100%

**Confidence Distribution**
- Tracks prediction confidence ranges

### Calibration Metrics

**Brier Score**
- Measures accuracy of probabilistic predictions
- Lower is better (0 = perfect calibration)
- Formula: `BS = (1/N) Σ(p - y)²`

**Expected Calibration Error (ECE)**
- Measures difference between confidence and accuracy
- Current: 0.179 

### Safety Metrics

**Hallucination Rate**
- Percentage of answers not grounded in context
- Current: 0.0% (strict prompting successful)

**Dangerous Error Count**
- Answers that could lead to patient harm
- Current: 0

**Safety Score**
- Composite safety metric
- Current: 0.98

### Confusion Matrix

**Answer-Level Confusion Matrix**
- Tracks predicted vs true answers (A/B/C/D)
- Visualized with green-bordered diagonal for correct predictions

![Confusion Matrix](reports/charts/confusion_matrix.png)

**Condition-Level Confusion Analysis**
- Identifies confusion between similar medical conditions
- Tracks:
  - Similar-condition errors (same medical group)
  - Unrelated-condition errors (different groups)
  - Top confusion pairs (e.g., "Pneumonia → Bronchitis")

**Medical Similarity Groups:**
- Respiratory infections, Cardiac ischemic, Thrombotic, Infections, etc.
- 16 predefined medical condition groups

### Error Analysis

**Error Categories:**
- Retrieval failures (0% current)
- Reasoning failures (100% of errors)
- High-confidence errors
- Safety-critical errors

**Error Breakdown by Concept:**
- Tracks errors by medical domain (cardiovascular, endocrine, etc.)
- Identifies systematic weaknesses

**Common Pitfalls:**
- Missing critical symptoms
- Medical terminology misunderstanding
- Incomplete differential diagnosis

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

**Generator v5.0 Features:**
- Realistic vital sign constraints
- Balanced answer distribution (25% A/B/C/D)
- Varied clinical presentations
- Controlled difficulty levels (easy/medium/hard)
- Multiple question types (diagnosis/treatment/management/immediate)

**Generation Process:**
1. Select clinical guideline
2. Generate realistic vital signs
3. Create case scenario with LLM
4. Generate 4 plausible options
5. Apply cryptographic shuffling for answer balance
6. Validate clinical consistency

**Quality Checks:**
- Fever > 101°F → Must consider infectious differentials
- Stroke → CT/MRI before anticoagulation
- Wide pulse pressure → Explained or normalized
- Clean option formatting (no embedded explanations)

**Dataset Statistics:**
- Total Questions: 100
- Distribution:
  - Answers: A (25%), B (25%), C (22%), D (21%)
  - Difficulty: Easy (23%), Medium (46%), Hard (22%)
  - Types: Diagnosis (23%), Treatment (23%), Management (23%), Immediate (22%)
  - Relevance: High (55%), Medium (18%), Low (18%)

---

## Experimental Results

### Retrieval Strategy Comparison

**Experiment:** Compared BM25, Concept-First, Hybrid, and Multi-Stage retrieval

**Results:**
- Multi-Stage: Best overall performance (highest MAP and recall)
- Concept-First: Best for medical concept matching
- BM25: Fast but lower accuracy
- Hybrid: Balanced performance

### Reasoning Method Comparison

**Experiment:** Evaluated 50 clinical cases across CoT, ToT, and Structured Medical Reasoning

**Results:**

| Method | Accuracy | Avg Time (ms) | Best For |
|--------|----------|---------------|----------|
| Chain-of-Thought | 40% | 4,019 | General cases |
| Tree-of-Thought | 50% | 43,981 | Complex scenarios |
| Structured Medical | 35% | 26,320 | Systematic evaluation |

**Key Findings:**
- ToT achieves highest accuracy but 10x slower than CoT
- Structured Medical improved from 15% to 35% with LLM enhancement
- Hybrid pipeline balances speed and accuracy

### Error Analysis

**Top Confusion Pairs:**
1. Heart Failure → Angina
2. DVT → Pulmonary Embolism
3. Gastritis → GI Bleed

**Error Distribution:**
- Similar condition errors: 33.3%
- Unrelated condition errors: 66.7%

**Root Causes:**
1. Embedding model mismatch (general vs medical domain)
2. Insufficient retrieval precision (4% at k=5)
3. Missing critical symptoms in reasoning
4. Medical terminology misunderstanding

### Hallucination Prevention

**Experiment:** Tested strict prompting to prevent LLM memory usage

**Results:**
- Hallucination Rate: 0.0%
- Method: Explicit instructions to use only provided context
- Verification: Tested on 10 cases with hallucination detection

### Calibration Analysis

**Findings:**
- Brier Score: 0.385 (moderate calibration)
- ECE: 0.46 (high calibration error)
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
# Evaluate on 10 cases
python scripts/evaluate_new_dataset.py --num-cases 10 --seed 42

# Evaluate on full dataset
python scripts/evaluate_new_dataset.py --num-cases 100

# Custom dataset
python scripts/evaluate_new_dataset.py --dataset path/to/questions.json
```

**Output:**
- Detailed metrics in `reports/new_dataset_eval_N_cases.json`
- Confusion matrix in `reports/charts/confusion_matrix.png`
- Performance summary in `reports/charts/performance_summary.png`

### Compare Reasoning Methods

```bash
# Compare CoT, ToT, and Structured on 20 cases
python scripts/compare_reasoning_methods.py --num-cases 20
```

**Output:** `reports/reasoning_method_comparison.json`

### Compare Retrieval Strategies

```bash
# Compare BM25, Concept-First, Hybrid, Multi-Stage
python scripts/compare_retrieval_strategies.py
```

**Output:** `reports/retrieval_strategy_comparison.json`

### Generate New Clinical Cases

```bash
# Generate 100 cases
python scripts/generate_clinical_cases_v5.py 100

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

## Performance Summary

### Current System Performance

**Overall Accuracy:** 60%  
**Target Accuracy:** 80% (achievable with medical embeddings)

**Retrieval Performance:**
- Precision@5: 4%
- Recall@5: 20%
- MAP: 0.118
- MRR: 0.215

**Reasoning Performance:**
- Chain Completeness: 100%
- Evidence Utilization: 100%
- Hallucination Rate: 0%

**Safety Performance:**
- Dangerous Errors: 0
- Safety Score: 1.0

**Calibration:**
- Brier Score: 0.385
- ECE: 0.46

### Known Issues and Solutions

**Issue 1: Low Accuracy (60% vs 80% target)**
- Root Cause: General-purpose embeddings (MiniLM-L6-v2)
- Solution: Switch to medical-domain model (PubMedBERT)
- Location: src/models/embeddings.py lines 116-119

**Issue 2: Low Retrieval Precision (4%)**
- Root Cause: General-purpose embeddings
- Solution: Medical embeddings + better reranking

**Issue 3: High Calibration Error (ECE: 0.46)**
- Root Cause: Overconfident predictions
- Solution: Enhanced confidence calibration (already implemented)

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
**Last Updated:** December 11, 2025
