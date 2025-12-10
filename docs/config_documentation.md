# Configuration Documentation

**Author:** Shreya Uprety  
**Last Updated:** December 11, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration File Structure](#configuration-file-structure)
3. [Configuration Parameters](#configuration-parameters)
4. [Usage](#usage)
5. [Configuration Flow](#configuration-flow)

---

## Overview

The `config/` directory contains the central configuration file for the Medical Question-Answering System. All pipeline parameters, model settings, and feature flags are defined in `pipeline_config.yaml`, allowing for centralized control and easy experimentation.

**Purpose:**
- Centralized configuration management
- Easy parameter tuning without code changes
- Reproducible experiments with versioned configs
- Feature flag control for optional improvements

---

## Configuration File Structure

### File: `pipeline_config.yaml`

Location: `config/pipeline_config.yaml`

The configuration file is organized into logical sections corresponding to different pipeline stages:

```yaml
# Retrieval Configuration
retrieval:
  k_stage1: 150
  k_stage2: 100
  k_stage3: 30
  bm25_weight: 0.5
  semantic_weight: 0.5
  concept_expansion_weight: 0.3
  use_multi_query_expansion: true
  use_symptom_synonyms: true
  use_guideline_prioritization: true
  cross_encoder_rerank: true
  rerank_top_k: 30

# Reasoning Configuration
reasoning:
  method: hybrid  # Options: cot, tot, structured, hybrid
  temperature: 0.0
  max_tokens: 512
  enable_tree_of_thought: true
  tot_complexity_threshold: 0.7
  tot_confidence_threshold: 0.75
  use_structured_fallback: true
  enable_llm_enhancement: true

# Model Configuration
models:
  ollama_model: llama3.1:8b
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  use_gpu: true
  fp16_precision: true
  batch_size: 32
  
# Improvement Modules (Optional)
improvements:
  enable_concept_expansion: true
  enable_clinical_feature_extraction: true
  enable_confidence_calibration: true
  enable_hallucination_detection: true
  enable_safety_verification: true
  enable_context_pruning: true
  enable_specialty_adaptation: true
  enable_terminology_normalization: true

# Evaluation Configuration
evaluation:
  compute_retrieval_metrics: true
  compute_reasoning_metrics: true
  compute_calibration_metrics: true
  enable_error_analysis: true
  enable_confusion_matrix: true
  save_reasoning_chains: true
  
# Logging Configuration
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  save_logs: true
  log_directory: logs/
  verbose_retrieval: false
  verbose_reasoning: true
```

---

## Configuration Parameters

### Retrieval Parameters

#### `k_stage1` (default: 150)
Number of documents retrieved in Stage 1 (broad retrieval).

- **Type:** Integer
- **Range:** 50-300
- **Impact:** Higher values increase recall but add noise
- **Recommendation:** 150 for balanced performance

#### `k_stage2` (default: 100)
Number of documents retained after Stage 2 (focused retrieval).

- **Type:** Integer
- **Range:** 50-200
- **Impact:** Filters Stage 1 results with enhanced queries
- **Recommendation:** 100 for most cases

#### `k_stage3` (default: 30)
Final number of documents after reranking.

- **Type:** Integer
- **Range:** 10-50
- **Impact:** Final context size for reasoning
- **Recommendation:** 30 to balance context quality and LLM window

#### `bm25_weight` (default: 0.5)
Weight for BM25 (lexical) scores in hybrid retrieval.

- **Type:** Float
- **Range:** 0.0-1.0
- **Impact:** Higher favors exact term matching
- **Formula:** `hybrid_score = bm25_weight × bm25 + (1-bm25_weight) × semantic`

#### `semantic_weight` (default: 0.5)
Weight for semantic similarity scores.

- **Type:** Float
- **Range:** 0.0-1.0
- **Impact:** Higher favors semantic meaning over exact terms
- **Note:** `bm25_weight + semantic_weight = 1.0`

#### `use_multi_query_expansion` (default: true)
Enable multi-query expansion for diverse retrieval.

- **Type:** Boolean
- **Impact:** Generates alternative phrasings of the query
- **Example:** "chest pain" → ["chest pain", "cardiac discomfort", "thoracic pain"]

#### `use_symptom_synonyms` (default: true)
Inject medical synonyms into queries.

- **Type:** Boolean
- **Impact:** Expands medical terminology coverage
- **Source:** UMLS synonym database

#### `cross_encoder_rerank` (default: true)
Use cross-encoder for final reranking.

- **Type:** Boolean
- **Impact:** Improves ranking precision but adds latency
- **Model:** Cross-encoder/ms-marco-MiniLM-L-6-v2

---

### Reasoning Parameters

#### `method` (default: hybrid)
Primary reasoning method.

- **Type:** String
- **Options:**
  - `cot`: Chain-of-Thought only (fast, 40% accuracy)
  - `tot`: Tree-of-Thought only (slow, 50% accuracy)
  - `structured`: Structured Medical Reasoning only (35% accuracy)
  - `hybrid`: Adaptive selection based on complexity
- **Recommendation:** `hybrid` for best accuracy-speed tradeoff

#### `temperature` (default: 0.0)
LLM sampling temperature.

- **Type:** Float
- **Range:** 0.0-2.0
- **Impact:** 0.0 = deterministic, higher = more creative
- **Recommendation:** 0.0 for medical QA (deterministic answers required)

#### `max_tokens` (default: 512)
Maximum tokens for reasoning chain.

- **Type:** Integer
- **Range:** 256-1024
- **Impact:** Longer allows more detailed reasoning but increases latency

#### `tot_complexity_threshold` (default: 0.7)
Minimum complexity score to trigger Tree-of-Thought.

- **Type:** Float
- **Range:** 0.0-1.0
- **Impact:** Only complex questions use expensive ToT
- **Complexity Factors:** Multi-system involvement, contradictory symptoms

#### `tot_confidence_threshold` (default: 0.75)
Maximum CoT confidence to escalate to ToT.

- **Type:** Float
- **Range:** 0.0-1.0
- **Impact:** Low-confidence CoT answers escalate to ToT
- **Logic:** `if complexity > 0.7 AND cot_confidence < 0.75: use ToT`

---

### Model Parameters

#### `ollama_model` (default: llama3.1:8b)
Ollama model for reasoning.

- **Type:** String
- **Options:** Any Ollama-compatible model
- **Tested:** llama3.1:8b, llama2:13b
- **Recommendation:** llama3.1:8b for balanced performance

#### `embedding_model` (default: sentence-transformers/all-MiniLM-L6-v2)
Model for semantic embeddings.

- **Type:** String
- **Current Issue:** General-purpose model causes accuracy drop (60% vs 80%)
- **Recommendation:** Switch to `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- **Location to Change:** `src/models/embeddings.py` lines 116-119

#### `use_gpu` (default: true)
Enable GPU acceleration for embeddings.

- **Type:** Boolean
- **Impact:** 10-50x faster embedding computation
- **Fallback:** CPU if GPU unavailable

#### `fp16_precision` (default: true)
Use half-precision (FP16) for embeddings.

- **Type:** Boolean
- **Impact:** 2x faster, 2x less memory, negligible accuracy loss
- **Requires:** CUDA-capable GPU

---

### Improvement Module Flags

#### `enable_concept_expansion` (default: true)
Expand queries with UMLS medical concepts.

- **Impact:** Improves recall for medical terminology
- **Example:** "MI" → "myocardial infarction, heart attack, acute coronary syndrome"

#### `enable_clinical_feature_extraction` (default: true)
Extract structured clinical features (symptoms, vitals, demographics).

- **Impact:** Enhances query understanding
- **Features:** Age, gender, symptoms, vitals, risk factors

#### `enable_confidence_calibration` (default: true)
Calibrate prediction confidence scores.

- **Impact:** Improves Brier score and ECE
- **Method:** Temperature scaling and Platt scaling

#### `enable_hallucination_detection` (default: true)
Detect answers not grounded in context.

- **Impact:** Critical for safety (achieves 0% hallucination rate)
- **Method:** Evidence matching and semantic entailment

#### `enable_safety_verification` (default: true)
Verify medical safety of answers.

- **Impact:** Prevents dangerous recommendations
- **Checks:** Contraindications, drug interactions, guideline violations

#### `enable_context_pruning` (default: true)
Remove irrelevant context before reasoning.

- **Impact:** Reduces noise, improves reasoning focus
- **Method:** Relevance scoring and redundancy removal

---

## Usage

### Loading Configuration

The configuration is automatically loaded by the system:

```python
from src.utils.config_loader import load_config

config = load_config()

# Access parameters
k_stage1 = config['retrieval']['k_stage1']
reasoning_method = config['reasoning']['method']
enable_tot = config['reasoning']['enable_tree_of_thought']
```

### Overriding Configuration

You can override configuration parameters programmatically:

```python
# Override retrieval parameters
config['retrieval']['k_stage1'] = 200
config['retrieval']['bm25_weight'] = 0.7

# Override reasoning method
config['reasoning']['method'] = 'tot'
config['reasoning']['temperature'] = 0.1
```

Or via command-line arguments:

```bash
python scripts/evaluate_new_dataset.py \
    --config config/pipeline_config.yaml \
    --override retrieval.k_stage1=200 \
    --override reasoning.method=tot
```

### Creating Custom Configurations

Create experiment-specific configurations:

```bash
# Copy base config
cp config/pipeline_config.yaml config/experiment_tot_only.yaml

# Edit experiment config
# Set reasoning.method = "tot"
# Set reasoning.enable_tree_of_thought = true

# Run with custom config
python scripts/evaluate_new_dataset.py \
    --config config/experiment_tot_only.yaml
```

---

## Configuration Flow

### Pipeline Initialization

```
config/pipeline_config.yaml
        ↓
src/utils/config_loader.py → load_config()
        ↓
├─ src/retrieval/multi_stage_retriever.py
│  └─ Uses: k_stage1, k_stage2, k_stage3, weights
│
├─ src/reasoning/rag_pipeline.py
│  └─ Uses: method, temperature, max_tokens
│
├─ src/models/embeddings.py
│  └─ Uses: embedding_model, use_gpu, fp16_precision
│
├─ src/improvements/ (19 modules)
│  └─ Uses: Feature flags (enable_*)
│
└─ src/evaluation/pipeline.py
   └─ Uses: Evaluation flags, metric settings
```

---

## Related Documentation

- [Part 2: RAG Implementation](part_2_rag_implementation.md) - How configuration affects pipeline behavior
- [Part 4: Experiments](part_4_experiments.md) - Configuration used in experiments
- [Scripts Documentation](scripts_documentation.md) - Command-line configuration overrides

---

**Documentation Author:** Shreya Uprety
