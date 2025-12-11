# Reports Documentation

**Author:** Shreya Uprety  
**Updated:** 2025-12-11 (Corrected based on evaluation results)

## Overview

The `reports/` directory contains evaluation results, performance metrics, and analysis reports for the Medical QA system.

**Report Types:**
- **Evaluation Results:** Overall system performance on 50 cases
- **Reasoning Method Comparison:** CoT vs ToT vs Structured Medical
- **Retrieval Strategy Comparison:** 6 retrieval strategies on 100 cases
- **Condition Confusion Analysis:** Error patterns by medical condition
- **Visualizations:** Performance charts and graphs

## Report Files

### 1. evaluation_results.json

**Purpose:** Complete evaluation results on 50 clinical cases

**Actual Structure from Evaluation:**

```json
{
  "metadata": {
    "evaluation_date": "2025-12-11T04:06:17.777485",
    "total_cases": 50,
    "split": "all",
    "evaluation_time_seconds": 5432.80569434166
  },
  "retrieval": {
    "precision_at_k": {
      "1": 0.0,
      "3": 0.10666666666666665,
      "5": 0.11200000000000003,
      "10": 0.07983333333333334
    },
    "recall_at_k": {
      "1": 0.0,
      "3": 0.32,
      "5": 0.56,
      "10": 0.78
    },
    "map_score": 0.2676756556137361,
    "mrr": 0.2676756556137361,
    "context_relevance_scores": [...],
    "medical_concept_coverage": 0.7507612568837058,
    "guideline_coverage": 1.0
  },
  "reasoning": {
    "exact_match_accuracy": 0.52,
    "semantic_accuracy": 0.52,
    "partial_credit_accuracy": 0.14,
    "brier_score": 0.25432525360557,
    "expected_calibration_error": 0.178589036485,
    "reasoning_chain_completeness": 1.0,
    "evidence_utilization_rate": 1.0,
    "confidence_distribution": {
      "90-100%": 7,
      "40-50%": 11,
      "0-10%": 5,
      "30-40%": 13,
      "70-80%": 2,
      "20-30%": 4,
      "10-20%": 7,
      "80-90%": 1
    },
    "hallucination_rate": 0.0,
    "cannot_answer_misuse_rate": 0.04,
    "method_accuracy": {
      "Advanced-structured": 0.52
    },
    "cot_tot_delta": 0.0,
    "verifier_pass_rate": 0.0
  },
  "classification": {
    "per_option": {
      "A": {
        "precision": 0.47619047619047616,
        "recall": 0.9090909090909091,
        "f1": 0.6249999999999999,
        "support": 11
      },
      "B": {
        "precision": 0.4,
        "recall": 0.3076923076923077,
        "f1": 0.34782608695652173,
        "support": 13
      },
      "C": {
        "precision": 0.75,
        "recall": 0.6428571428571429,
        "f1": 0.6923076923076924,
        "support": 14
      },
      "D": {
        "precision": 0.7142857142857143,
        "recall": 0.4166666666666667,
        "f1": 0.5263157894736842,
        "support": 12
      }
    },
    "macro_precision": 0.5851190476190476,
    "macro_recall": 0.5690767565767565,
    "macro_f1": 0.5478623921844745,
    "weighted_f1": 0.5480967259285338,
    "balanced_accuracy": 0.5690767565767565,
    "confusion_matrix": {...},
    "option_distribution": {...}
  },
  "safety": {
    "dangerous_error_count": 2,
    "contraindication_check_accuracy": 0.0,
    "urgency_recognition_accuracy": 0.0,
    "safety_score": 0.96
  },
  "error_analysis": {...},
  "performance_segmentation": {...},
  "chart_paths": {...}
}
```

**Key Metrics from Evaluation:**

| Metric | Value | Issues/Target |
|--------|-------|---------------|
| Overall Accuracy | 52.0% | Target: 65% |
| Precision@5 | 11.2% | Too low, target: 16% |
| Recall@5 | 56.0% | Acceptable |
| MAP | 0.268 | Moderate |
| Brier Score | 0.254 | Needs calibration (target: <0.20) |
| ECE | 0.179 | Needs improvement (target: <0.15) |
| Reasoning Time | 5433s (~90.5min) | 41.4s average per case |
| Dangerous Errors | 2 | Critical safety issue |
| Safety Accuracy | 0.0% | Critical gap (contraindications/urgency) |

### 2. reasoning_method_comparison.json

**Purpose:** Performance comparison of 3 reasoning methods on 50 cases

**Actual Structure from Evaluation:**

```json
{
  "timestamp": "2025-12-10 20:22:35",
  "num_cases": 50,
  "results": [
    {
      "method_name": "Chain-of-Thought",
      "description": "Standard LLM reasoning with step-by-step prompting",
      "accuracy": 0.34,
      "avg_reasoning_time_ms": 4954.734697341919,
      "total_time_seconds": 247.73973488807678,
      "num_cases": 50,
      "correct_predictions": 17,
      "brier_score": 0.42393981381797075,
      "ece": 0.2663503756622089,
      "answer_distribution": {
        "B": 23,
        "C": 7,
        "A": 17,
        "D": 2,
        "Cannot answer from the provided context.": 1
      },
      "avg_reasoning_length": 98.06,
      "avg_evidence_usage": 10.64,
      "reasoning_coherence": 0.32666666666666666
    },
    {
      "method_name": "Tree-of-Thought",
      "description": "Structured multi-branch reasoning with explicit thought paths",
      "accuracy": 0.52,
      "avg_reasoning_time_ms": 41366.63206100464,
      "total_time_seconds": 2068.350494861603,
      "num_cases": 50,
      "correct_predictions": 26,
      "brier_score": 0.3442,
      "ece": 0.3099999999999999,
      "answer_distribution": {
        "C": 14,
        "A": 18,
        "B": 14,
        "D": 2,
        "Cannot answer from the provided context.": 2
      },
      "avg_reasoning_length": 525.04,
      "avg_evidence_usage": 3.9,
      "reasoning_coherence": 0.43333333333333335
    },
    {
      "method_name": "Structured Medical",
      "description": "5-step clinical reasoning: feature extraction → differential → evidence → treatment → selection",
      "accuracy": 0.44,
      "avg_reasoning_time_ms": 26990.508499145508,
      "total_time_seconds": 1349.5254249572754,
      "num_cases": 50,
      "correct_predictions": 22,
      "brier_score": 0.29472068743289254,
      "ece": 0.28264520580952374,
      "answer_distribution": {
        "C": 9,
        "A": 13,
        "D": 5,
        "B": 21,
        "Cannot answer from the provided context.": 2
      },
      "avg_reasoning_length": 40.96,
      "avg_evidence_usage": 6.72,
      "reasoning_coherence": 0.32
    }
  ],
  "summary": {
    "best_accuracy": {
      "method": "Tree-of-Thought",
      "score": 0.52
    },
    "best_calibration": {
      "method": "Structured Medical",
      "brier_score": 0.29472068743289254
    },
    "fastest": {
      "method": "Chain-of-Thought",
      "time_ms": 4954.734697341919
    }
  },
  "recommendations": [
    "Tree-of-Thought achieves the highest accuracy for medical MCQs.",
    "Structured Medical provides the best-calibrated confidence scores.",
    "Chain-of-Thought is recommended for time-sensitive applications."
  ]
}
```

**Comparison Table:**

| Method | Accuracy | Avg Time (ms) | Brier Score | ECE | Coherence | Answer Bias |
|--------|----------|---------------|-------------|-----|-----------|-------------|
| Tree-of-Thought | 52% | 41,367 | 0.344 | 0.310 | 43.3% | Balanced (A:36%, C:28%, B:28%) |
| Structured Medical | 44% | 26,991 | 0.295 | 0.283 | 32.0% | B-heavy (42%) |
| Chain-of-Thought | 34% | 4,955 | 0.424 | 0.266 | 32.7% | B-heavy (46%) |

### 3. retrieval_strategy_comparison.json

**Purpose:** Comparison of 6 retrieval strategies on 100 cases

**Actual Structure from Evaluation:**

```json
{
  "timestamp": "2025-12-09 21:38:10",
  "num_cases": 100,
  "dataset_path": "data/processed/questions/questions_1.json",
  "results": [
    {
      "strategy_name": "Single-Stage FAISS",
      "description": "Pure semantic search using dense embeddings (MiniLM-L6-v2)",
      "map_score": 0.21099611696465245,
      "mrr": 0.4219922339293049,
      "precision_at_1": 0.0,
      "precision_at_3": 0.2533333333333333,
      "precision_at_5": 0.17599999999999993,
      "recall_at_1": 0.0,
      "recall_at_3": 0.38,
      "recall_at_5": 0.44,
      "avg_query_time_ms": 8.583245277404785,
      "total_time_seconds": 0.8645603656768799,
      "num_cases": 100
    },
    {
      "strategy_name": "Single-Stage BM25",
      "description": "Pure keyword-based search using BM25 algorithm",
      "map_score": 0.20722960811118704,
      "mrr": 0.4144592162223741,
      "precision_at_1": 0.0,
      "precision_at_3": 0.23999999999999996,
      "precision_at_5": 0.17399999999999996,
      "recall_at_1": 0.0,
      "recall_at_3": 0.36,
      "recall_at_5": 0.435,
      "avg_query_time_ms": 1.3998913764953613,
      "total_time_seconds": 0.13998913764953613,
      "num_cases": 100
    },
    {
      "strategy_name": "Hybrid Linear",
      "description": "Linear combination of FAISS (0.65) and BM25 (0.35)",
      "map_score": 0.21062643350801247,
      "mrr": 0.42125286701602493,
      "precision_at_1": 0.0,
      "precision_at_3": 0.24666666666666662,
      "precision_at_5": 0.17799999999999994,
      "recall_at_1": 0.0,
      "recall_at_3": 0.37,
      "recall_at_5": 0.445,
      "avg_query_time_ms": 8.325040340423584,
      "total_time_seconds": 0.8325040340423584,
      "num_cases": 100
    },
    {
      "strategy_name": "Multi-Stage (3-stage)",
      "description": "Stage 1: FAISS (k=150) -> Stage 2: BM25 filter (k=100) -> Stage 3: Cross-encoder rerank (k=25)",
      "map_score": 0.20416295721187028,
      "mrr": 0.40832591442374055,
      "precision_at_1": 0.0,
      "precision_at_3": 0.2433333333333333,
      "precision_at_5": 0.16999999999999993,
      "recall_at_1": 0.0,
      "recall_at_3": 0.365,
      "recall_at_5": 0.425,
      "avg_query_time_ms": 2878.1001448631287,
      "total_time_seconds": 287.81001448631287,
      "num_cases": 100
    },
    {
      "strategy_name": "Concept-First",
      "description": "BM25 keyword filter followed by FAISS semantic refinement",
      "map_score": 0.21214898989898992,
      "mrr": 0.42429797979797984,
      "precision_at_1": 0.0,
      "precision_at_3": 0.24999999999999997,
      "precision_at_5": 0.17999999999999994,
      "recall_at_1": 0.0,
      "recall_at_3": 0.375,
      "recall_at_5": 0.45,
      "avg_query_time_ms": 11.620626449584961,
      "total_time_seconds": 1.162062644958496,
      "num_cases": 100
    },
    {
      "strategy_name": "Semantic-First",
      "description": "FAISS semantic search followed by BM25 keyword refinement",
      "map_score": 0.2125936988936989,
      "mrr": 0.4251873977873978,
      "precision_at_1": 0.0,
      "precision_at_3": 0.2533333333333333,
      "precision_at_5": 0.17799999999999996,
      "recall_at_1": 0.0,
      "recall_at_3": 0.38,
      "recall_at_5": 0.445,
      "avg_query_time_ms": 9.648230075836182,
      "total_time_seconds": 0.9648230075836182,
      "num_cases": 100
    }
  ],
  "summary": {
    "best_map": {
      "strategy": "Semantic-First",
      "score": 0.2125936988936989
    },
    "best_recall": {
      "strategy": "Concept-First",
      "score": 0.45
    },
    "fastest": {
      "strategy": "Single-Stage BM25",
      "time_ms": 1.3998913764953613
    }
  },
  "recommendations": [
    "Single-Stage BM25 offers excellent speed for real-time applications."
  ]
}
```

**Retrieval Strategy Rankings:**

| Strategy | MAP | MRR | P@5 | R@5 | Time (ms) | Rank |
|----------|-----|-----|-----|-----|-----------|------|
| Semantic-First | 0.2126 | 0.4252 | 17.8% | 44.5% | 9.65 | 1 |
| Concept-First | 0.2121 | 0.4243 | 18.0% | 45.0% | 11.62 | 2 |
| Single FAISS | 0.2110 | 0.4220 | 17.6% | 44.0% | 8.58 | 3 |
| Hybrid Linear | 0.2106 | 0.4213 | 17.8% | 44.5% | 8.33 | 4 |
| Single BM25 | 0.2072 | 0.4145 | 17.4% | 43.5% | 1.40 | 5 |
| Multi-Stage | 0.2042 | 0.4083 | 17.0% | 42.5% | 2,878 | 6 |

### 4. Error Analysis (Within evaluation_results.json)

**Purpose:** Detailed error analysis by category and root cause

**Actual Structure from Evaluation:**

```json
"error_analysis": {
  "error_categories": {
    "reasoning": {
      "error_type": "reasoning",
      "description": "System retrieved relevant info but made incorrect reasoning",
      "count": 16,
      "examples": [...],
      "root_causes": [
        "Insufficient chain-of-thought reasoning steps",
        "Failure to properly weight evidence from multiple sources",
        "Over-reliance on single retrieved document",
        "Missing critical symptom analysis"
      ],
      "proposed_solutions": [...]
    },
    "knowledge": {
      "error_type": "knowledge",
      "description": "System has incorrect medical knowledge or interpretation",
      "count": 8,
      "examples": [...],
      "root_causes": [
        "Incorrect interpretation of medical guidelines",
        "Missing context about patient-specific factors",
        "Failure to consider contraindications",
        "Incorrect application of treatment protocols"
      ],
      "proposed_solutions": [...]
    }
  }
},
"performance_segmentation": {
  "by_category": {
    "Gastroenterology": {"accuracy": 0.7142857142857143, "correct": 5, "total": 7},
    "Endocrine": {"accuracy": 0.6666666666666666, "correct": 4, "total": 6},
    "Cardiovascular": {"accuracy": 0.5454545454545454, "correct": 6, "total": 11},
    "Infectious Disease": {"accuracy": 0.0, "correct": 0, "total": 3},
    "Respiratory": {"accuracy": 0.625, "correct": 5, "total": 8},
    "Rheumatology": {"accuracy": 0.3333333333333333, "correct": 1, "total": 3},
    "Hematology": {"accuracy": 0.3333333333333333, "correct": 1, "total": 3},
    "Nephrology": {"accuracy": 0.6666666666666666, "correct": 2, "total": 3},
    "Psychiatry": {"accuracy": 0.3333333333333333, "correct": 1, "total": 3},
    "Critical Care": {"accuracy": 1.0, "correct": 1, "total": 1},
    "Neurology": {"accuracy": 0.0, "correct": 0, "total": 2}
  },
  "by_question_type": {
    "diagnosis": {"accuracy": 0.5217391304347826, "correct": 24, "total": 46},
    "treatment": {"accuracy": 1.0, "correct": 2, "total": 2},
    "other": {"accuracy": 0.0, "correct": 0, "total": 2}
  },
  "by_complexity": {
    "moderate": {"accuracy": 0.52, "correct": 13, "total": 25},
    "complex": {"accuracy": 0.46153846153846156, "correct": 6, "total": 13},
    "simple": {"accuracy": 0.5833333333333334, "correct": 7, "total": 12}
  },
  "by_relevance_level": {
    "high": {"accuracy": 0.4482758620689655, "correct": 13, "total": 29},
    "low": {"accuracy": 0.45454545454545453, "correct": 5, "total": 11},
    "medium": {"accuracy": 0.8, "correct": 8, "total": 10}
  }
},
"pitfalls": [
  {
    "pitfall": "Overconfident Wrong Answers",
    "description": "System shows high confidence (>80%) but gives incorrect answers",
    "count": 2,
    "severity": "high",
    "examples": [...],
    "solution": "Implement confidence calibration and add uncertainty estimation"
  },
  {
    "pitfall": "Missing Critical Symptoms",
    "description": "Reasoning fails to consider important symptoms from case description",
    "count": 20,
    "severity": "medium",
    "examples": [...],
    "solution": "Enhance symptom extraction and ensure all symptoms are considered in reasoning"
  },
  {
    "pitfall": "Medical Terminology Misunderstanding",
    "description": "System fails to properly interpret medical abbreviations or terms",
    "count": 24,
    "severity": "medium",
    "examples": [...],
    "solution": "Add medical terminology expansion and abbreviation resolution"
  }
],
"confusion_matrix": {
  "Gastroenterology": {"cardiology": 7},
  "Endocrine": {"endocrinology": 2, "cardiology": 3, "nephrology": 1},
  "Cardiovascular": {"cardiology": 10, "endocrinology": 1},
  "Infectious Disease": {"cardiology": 3},
  "Respiratory": {"cardiology": 8},
  "Rheumatology": {"cardiology": 2, "nephrology": 1},
  "Hematology": {"cardiology": 3},
  "Nephrology": {"nephrology": 3},
  "Psychiatry": {"obstetrics": 1, "cardiology": 2},
  "Critical Care": {"cardiology": 1},
  "Neurology": {"cardiology": 2}
}
```

**Key Error Insights:**

| Category | Issues | Severity |
|----------|--------|----------|
| Reasoning Errors | 16 cases (64% of errors) | High |
| Knowledge Errors | 8 cases (32% of errors) | High |
| Overconfident Wrong | 2 cases (>80% confidence but wrong) | Critical |
| Missing Symptoms | 20 cases (40% of total) | Medium |
| Terminology Issues | 24 cases (48% of total) | Medium |
| Infectious Disease | 0% accuracy (0/3) | Critical gap |
| Neurology | 0% accuracy (0/2) | Critical gap |

## Visualization Charts

**Directory:** `reports/charts/`

### Generated Charts

**1. performance_summary.png**
- Overall performance dashboard
- Includes accuracy, retrieval metrics, calibration, safety scores

**2. confusion_matrix.png**
- 4x4 answer confusion matrix (A/B/C/D predicted vs true)
- Highlights diagonal (correct predictions) in green

**3. error_analysis.png**
- Error category breakdown
- Performance by medical specialty
- Confidence-accuracy calibration plot
- Pitfalls summary visualization

**Chart Paths from Evaluation:**
```json
"chart_paths": {
  "performance_summary": "reports\\charts\\performance_summary.png",
  "confusion_matrix": "reports\\charts\\confusion_matrix.png",
  "error_analysis": "reports\\charts\\error_analysis.png"
}
```

## Interpreting Results

### Accuracy Benchmarks

- **52% accuracy:** Baseline performance with Tree-of-Thought
- **Target:** 65% (achievable with medical embeddings + optimizations)
- **Critical gaps:** Infectious Disease (0%), Neurology (0%), Safety (0%)

### Retrieval Performance

- **Precision@5 (11.2%):** Main bottleneck - too low
- **Recall@5 (56.0%):** Acceptable recall
- **Precision-Recall Tradeoff:** High recall but poor precision
- **Best strategy:** Semantic-First (MAP: 0.213, 9.65ms)

### Calibration Quality

- **Brier Score (0.254):** Needs improvement (target: <0.20)
- **ECE (0.179):** Moderate calibration (target: <0.15)
- **Confidence Issues:**
  - Overconfident errors: 2 cases (>80% confidence wrong)
  - Low-confidence accuracy: 20% (0-10% range)
  - Optimal range: 40-90% confidence (46-86% accuracy)

### Safety Critical Issues

- **Contraindication checks:** 0% accuracy
- **Urgency recognition:** 0% accuracy
- **Dangerous errors:** 2 cases
- **Safety score:** 0.96 (good overall but critical gaps)

### Timing Performance

- **Tree-of-Thought:** 41.4s per query (accurate but slow)
- **Chain-of-Thought:** 5.0s per query (fast but less accurate)
- **Structured Medical:** 27.0s per query (balanced)
- **Retrieval time:** 9.65ms (Semantic-First strategy)

## Using Reports

### Loading Evaluation Results

```python
import json

# Load main evaluation results
with open('reports/evaluation_results.json') as f:
    results = json.load(f)

# Access key metrics
print(f"Accuracy: {results['reasoning']['exact_match_accuracy']:.1%}")
print(f"Precision@5: {results['retrieval']['precision_at_k']['5']:.1%}")
print(f"Brier Score: {results['reasoning']['brier_score']:.3f}")

# Check safety issues
if results['safety']['dangerous_error_count'] > 0:
    print(f"CRITICAL: {results['safety']['dangerous_error_count']} dangerous errors")

# Analyze performance by category
for category, data in results['performance_segmentation']['by_category'].items():
    if data['accuracy'] < 0.5:
        print(f"Low performance in {category}: {data['accuracy']:.1%}")
```

### Analyzing Error Patterns

```python
# Load and analyze error categories
error_categories = results['error_analysis']['error_categories']
total_errors = sum(cat['count'] for cat in error_categories.values())

print(f"Total errors: {total_errors}")
for category, data in error_categories.items():
    percentage = (data['count'] / total_errors) * 100
    print(f"{category}: {data['count']} errors ({percentage:.1f}%)")
    print(f"  Root causes: {', '.join(data['root_causes'][:2])}")
```

### Comparing Reasoning Methods

```python
# Load reasoning comparison
with open('reports/reasoning_method_comparison.json') as f:
    comparison = json.load(f)

# Find best method for different needs
methods = comparison['results']
best_accuracy = max(methods, key=lambda x: x['accuracy'])
best_calibration = min(methods, key=lambda x: x['brier_score'])
fastest = min(methods, key=lambda x: x['avg_reasoning_time_ms'])

print(f"Most accurate: {best_accuracy['method_name']} ({best_accuracy['accuracy']:.1%})")
print(f"Best calibrated: {best_calibration['method_name']} (Brier: {best_calibration['brier_score']:.3f})")
print(f"Fastest: {fastest['method_name']} ({fastest['avg_reasoning_time_ms']/1000:.1f}s avg)")
```

## Report Updates

Reports are generated by:

### Main Evaluation:

```bash
python src/evaluation/evaluate_pipeline.py \
    --dataset data/processed/questions/questions_1.json \
    --num-cases 50 \
    --output reports/evaluation_results.json
```

### Reasoning Comparison:

```bash
python scripts/compare_reasoning_methods.py \
    --num-cases 50 \
    --output reports/reasoning_method_comparison.json
```

### Retrieval Comparison:

```bash
python scripts/compare_retrieval_strategies.py \
    --num-cases 100 \
    --output reports/retrieval_strategy_comparison.json
```

**Frequency:** Reports should be regenerated after major system changes.

## Key Insights from Reports

### 1. Current System State (Tree-of-Thought)

- **Accuracy:** 52% (26/50 correct)
- **Main bottleneck:** Retrieval precision (11.2% @5)
- **Critical gaps:** Safety (0%), Infectious Disease (0%), Neurology (0%)
- **Strength:** No hallucinations, good evidence utilization

### 2. Optimization Priorities

1. Improve retrieval precision (target: 16% @5)
2. Add safety checks (contraindications, urgency recognition)
3. Implement confidence calibration (fix 2 overconfident errors)
4. Enhance symptom extraction (address 20 missed symptom cases)
5. Improve medical terminology (address 24 misunderstanding cases)

### 3. Expected Improvements

- **With medical embeddings:** 52% → 65-70% accuracy
- **With confidence calibration:** Brier 0.254 → 0.22
- **With symptom extraction:** Reduce missed symptoms from 20 to <10 cases

### 4. Configuration Recommendations

- **Retrieval:** Semantic-First (MAP: 0.213, 9.65ms)
- **Reasoning:** Tree-of-Thought for accuracy, Chain-of-Thought for speed
- **Safety:** Require contraindication checks for all treatment questions
- **Calibration:** Implement temperature scaling for confidence scores

## Related Documentation

- **Part 3:** Evaluation Framework - Updated with actual metrics
- **Part 4:** Experiments and Analysis - Detailed experiment results
- **Evaluation Documentation** - Evaluation methodology

---

**Documentation Author:** Shreya Uprety  
**Evaluation Reference:** 2025-12-11 Medical QA Evaluation Results  
**Report Status:** Current reports reflect 52% accuracy with critical gaps in safety and specialty performance