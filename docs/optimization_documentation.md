# Optimization Documentation

**Author:** Shreya Uprety  
**Last Updated:** December 11, 2025

---

## Overview

The `src/optimization/` module provides parameter tuning and optimization tools for retrieval and reasoning components.

**Key Modules:**
- **Retrieval Tuner:** Optimizes retrieval parameters (weights, thresholds)
- **Parameter Optimizer:** Grid search for systematic optimization
- **Symptom Extractor:** Optimized clinical feature extraction

---

## Retrieval Tuner

**File:** `src/optimization/retrieval_tuner.py`

### Purpose

Tunes retrieval parameters to balance precision and recall. Addresses retrieval performance issues through systematic weight optimization.

### Key Features

1. **Hybrid Weight Tuning:** Find optimal FAISS:BM25 ratio
2. **Similarity Threshold Tuning:** Adjust relevance cutoffs
3. **Multi-Stage Weight Tuning:** Optimize stage fusion weights
4. **Query Expansion Tuning:** Balance recall vs precision

### Implementation

```python
class RetrievalTuner:
    """
    Optimize retrieval parameters based on validation set.
    
    Tests different configurations:
    - Hybrid search weights (FAISS:BM25 ratios)
    - Similarity thresholds
    - Query expansion aggressiveness
    - Multi-stage weights
    """
    
    def __init__(
        self,
        faiss_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        embedding_model: EmbeddingModel
    ):
        """Initialize retrieval tuner."""
        self.faiss_store = faiss_store
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
```

### Methods

#### tune_hybrid_weights()

```python
def tune_hybrid_weights(
    self,
    validation_queries: List[Dict],
    weight_combinations: Optional[List[Tuple[float, float]]] = None
) -> TuningResult:
    """
    Tune hybrid search weights (FAISS vs BM25).
    
    Args:
        validation_queries: List of queries with ground truth
        weight_combinations: List of (semantic_weight, keyword_weight) tuples
        
    Returns:
        TuningResult with best configuration
        
    Default Weight Combinations:
        (0.7, 0.3) - Favor semantic
        (0.6, 0.4) - Slightly favor semantic
        (0.5, 0.5) - Balanced
        (0.4, 0.6) - Slightly favor keyword
        (0.3, 0.7) - Favor keyword
    """
```

#### tune_similarity_threshold()

```python
def tune_similarity_threshold(
    self,
    validation_queries: List[Dict],
    threshold_range: Tuple[float, float] = (0.3, 0.8),
    num_steps: int = 10
) -> TuningResult:
    """
    Optimize similarity threshold for FAISS retrieval.
    
    Args:
        validation_queries: Validation set
        threshold_range: (min, max) threshold values to test
        num_steps: Number of threshold values to test
        
    Returns:
        Optimal threshold configuration
    """
```

#### tune_multi_stage_weights()

```python
def tune_multi_stage_weights(
    self,
    validation_queries: List[Dict],
    stage_weight_combinations: Optional[List[Tuple[float, float, float]]] = None
) -> TuningResult:
    """
    Optimize multi-stage retrieval fusion weights.
    
    Args:
        validation_queries: Validation set
        stage_weight_combinations: (stage1, stage2, stage3) weight tuples
        
    Returns:
        Best multi-stage configuration
    """
```

### Usage Example

```python
from src.optimization.retrieval_tuner import RetrievalTuner

# Initialize tuner
tuner = RetrievalTuner(
    faiss_store=faiss_store,
    bm25_retriever=bm25_retriever,
    embedding_model=embedding_model
)

# Load validation queries
validation_queries = load_validation_set()

# Tune hybrid weights
result = tuner.tune_hybrid_weights(validation_queries)

print(f"Best weights: {result.best_params}")
print(f"MAP: {result.metrics.map_score:.3f}")
print(f"Precision@5: {result.metrics.precision_at_k[5]:.3f}")
print(f"Recall@5: {result.metrics.recall_at_k[5]:.3f}")
```

### Results

**Tested Configurations:**
- 5 weight combinations
- 10 similarity thresholds
- 8 multi-stage weight combinations

**Best Configuration (from experiments):**
- Hybrid weights: (0.6, 0.4) - Semantic-first
- Similarity threshold: 0.45
- Multi-stage weights: (0.5, 0.3, 0.2)

**Performance:**
- MAP: 0.212 (Concept-First strategy)
- Precision@5: 18.0%
- Recall@5: 45.0%

---

## Parameter Optimizer

**File:** `src/optimization/parameter_optimizer.py`

### Purpose

Systematic grid search optimization for all pipeline parameters. Addresses the balance between precision and recall through exhaustive parameter exploration.

### Implementation

```python
class ParameterOptimizer:
    """
    Systematic parameter optimization using grid search.
    
    Optimizes:
    - Hybrid search weights
    - Multi-stage weights
    - Similarity thresholds
    - Query expansion parameters
    """
    
    def __init__(
        self,
        retrieval_tuner: RetrievalTuner,
        validation_queries: List[Dict]
    ):
        """Initialize parameter optimizer."""
        self.retrieval_tuner = retrieval_tuner
        self.validation_queries = validation_queries
```

### Methods

#### optimize()

```python
def optimize(
    self,
    configs: List[OptimizationConfig],
    objective: str = 'balanced'
) -> OptimizationResult:
    """
    Perform grid search optimization.
    
    Args:
        configs: List of optimization configurations
        objective: 'precision', 'recall', 'f1', 'balanced'
        
    Returns:
        OptimizationResult with best parameters
    """
```

### Objective Functions

1. **Precision:** Maximize precision@k
2. **Recall:** Maximize recall@k
3. **F1:** Maximize F1 score
4. **Balanced:** Weighted combination (0.4×P@5 + 0.4×R@5 + 0.2×MAP)

### Usage Example

```python
from src.optimization.parameter_optimizer import (
    ParameterOptimizer,
    OptimizationConfig
)

# Define optimization configurations
configs = [
    OptimizationConfig(
        param_name='semantic_weight',
        param_values=[0.3, 0.4, 0.5, 0.6, 0.7],
        objective='balanced'
    ),
    OptimizationConfig(
        param_name='similarity_threshold',
        param_values=[0.3, 0.4, 0.5, 0.6, 0.7],
        objective='balanced'
    )
]

# Run optimization
optimizer = ParameterOptimizer(tuner, validation_queries)
result = optimizer.optimize(configs, objective='balanced')

print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score:.3f}")
```

---

## Symptom Extractor

**File:** `src/optimization/symptom_extractor.py`

### Purpose

Optimized clinical symptom extraction using pattern matching and medical NER.

### Features

- **Symptom Detection:** Regex + medical ontology
- **Severity Extraction:** Mild, moderate, severe classification
- **Temporal Parsing:** Duration, onset pattern
- **Negative Detection:** "No fever" vs "fever"

### Usage

```python
from src.optimization.symptom_extractor import SymptomExtractor

extractor = SymptomExtractor()

symptoms = extractor.extract(clinical_case)
# Returns: {
#   'symptoms': ['chest_pain', 'dyspnea'],
#   'severity': {'chest_pain': 'severe'},
#   'duration': {'chest_pain': '2 hours'},
#   'onset': 'acute'
# }
```

---

## Configuration

All optimization parameters are stored in `config/pipeline_config.yaml`:

```yaml
optimization:
  retrieval:
    hybrid_weights:
      semantic: 0.6
      keyword: 0.4
    similarity_threshold: 0.45
    multi_stage_weights:
      stage1: 0.5
      stage2: 0.3
      stage3: 0.2
  
  reasoning:
    temperature: 0.0
    max_tokens: 512
    
  validation:
    split: 0.2
    min_samples: 50
```

---

## Running Optimization

### Command Line

```bash
# Tune retrieval parameters
python -m src.optimization.retrieval_tuner \
    --validation data/validation.json \
    --output results/tuning_results.json

# Grid search optimization
python -m src.optimization.parameter_optimizer \
    --configs configs/optimization_configs.json \
    --objective balanced \
    --output results/optimization_results.json
```

### Programmatic

```python
from src.optimization import run_full_optimization

# Run complete optimization pipeline
results = run_full_optimization(
    faiss_store=faiss_store,
    bm25_retriever=bm25_retriever,
    validation_queries=validation_queries,
    objectives=['precision', 'recall', 'balanced']
)

# Apply best configuration
apply_optimized_config(results.best_params)
```

---

## Optimization Results

### Retrieval Parameter Tuning

| Parameter | Default | Optimized | Improvement |
|-----------|---------|-----------|-------------|
| Semantic Weight | 0.5 | 0.6 | +2.5% MAP |
| Keyword Weight | 0.5 | 0.4 | - |
| Similarity Threshold | 0.5 | 0.45 | +1.8% Recall@5 |
| Multi-Stage (S1:S2:S3) | 0.33:0.33:0.33 | 0.5:0.3:0.2 | +1.2% MAP |

### Overall Impact

- **Precision@5:** 16.2% → **18.0%** (+11%)
- **Recall@5:** 42.0% → **45.0%** (+7%)
- **MAP:** 0.195 → **0.212** (+9%)

---

## Related Documentation

- [Retrieval Documentation](retrieval_documentation.md)
- [Part 4: Experiments](part_4_experiments.md)
- [Improvements Documentation](improvements_documentation.md)

---

**Documentation Author:** Shreya Uprety
