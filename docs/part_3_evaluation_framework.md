# Part 3: Evaluation Framework

**Author:** Shreya Uprety  
**Updated:** 2025-12-11 (Based on comprehensive evaluation results)

## Overview

This document details the comprehensive evaluation framework for the Medical Question-Answering System, including all metrics, error analysis, and visualization tools. Based on 50-case evaluation revealing 52% accuracy with critical insights.

## Evaluation Pipeline

```
Clinical Question + Gold Answer
        ↓
RAG Pipeline (Retrieval + Reasoning)
        ↓
Predicted Answer + Reasoning Chain + Confidence
        ↓
┌─────────────────────────────────────────────┐
│         Comprehensive Metrics Suite          │
│  ├─ Accuracy Metrics (Baseline: 52%)        │
│  ├─ Retrieval Metrics (Precision@5: 11.2%)  │
│  ├─ Reasoning Metrics (Tree-of-Thought best)│
│  ├─ Calibration Metrics (Brier: 0.254)      │
│  ├─ Safety Metrics (0% contraindication)    │
│  └─ Classification Metrics (F1: 0.548)      │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│         Deep Error Analysis                  │
│  ├─ Error Categorization (Reasoning: 16,    │
│  │                       Knowledge: 8)      │
│  ├─ Performance Segmentation                │
│  ├─ Pitfalls Analysis (3 major issues)      │
│  └─ Root Cause Analysis                     │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│         Multi-Dimensional Visualization      │
│  ├─ Performance Summary (PNG)               │
│  ├─ Confusion Matrix (PNG)                  │
│  ├─ Error Analysis Charts (PNG)             │
│  ├─ Category Performance                    │
│  └─ Confidence Distribution                 │
└─────────────────┬───────────────────────────┘
                  ↓
       Comprehensive Evaluation Report (JSON)
```

## Core Performance Metrics

### Exact Match Accuracy

**Definition:** Percentage of questions with correct answer prediction

**Formula:**
```
Accuracy = (Correct Predictions / Total Questions) × 100
```

**Current Performance:** 52.0% (26/50 correct)

**Method Performance Comparison:**
- Chain-of-Thought: 34.0% (17/50)
- **Tree-of-Thought: 52.0% (26/50) - Best**
- Structured Medical: 44.0% (22/50)
- **Target:** 65% (achievable with optimization)

### Semantic Accuracy

**Definition:** Measures answer similarity even if not exact match

**Current Performance:** 52.0% (identical to exact match)

**Note:** Multiple-choice format requires exact match, semantic similarity not applicable

## Retrieval Metrics

### Precision@k

**Definition:** Percentage of retrieved documents that are relevant

**Formula:**
```
Precision@k = (Relevant Documents in Top k / k) × 100
```

**Current Performance:**
- Precision@1: 0.0% (no relevant document at position 1)
- Precision@3: 10.7% (moderate)
- **Precision@5: 11.2% (low - main optimization target)**
- Precision@10: 7.98% (decreases with more documents)

**Interpretation:** System struggles to rank relevant documents first

### Recall@k

**Definition:** Percentage of relevant documents retrieved

**Formula:**
```
Recall@k = (Relevant Documents in Top k / Total Relevant) × 100
```

**Current Performance:**
- Recall@1: 0.0%
- Recall@3: 32.0%
- **Recall@5: 56.0% (reasonable recall)**
- Recall@10: 78.0% (good recall at cost of precision)

**Precision-Recall Tradeoff:** High recall (56% @5) but low precision (11% @5)

### Mean Average Precision (MAP)

**Definition:** Average precision across all queries

**Current Performance:** 0.268

**Interpretation:** Moderate overall retrieval quality

### Mean Reciprocal Rank (MRR)

**Definition:** Average of reciprocal ranks of first relevant document

**Current Performance:** 0.268 (identical to MAP)

**Interpretation:** Relevant documents not ranked high

### Context Relevance Score

**Definition:** Measures how relevant retrieved context is to the question (0-2 scale)

**Scoring:**
- 0: Completely irrelevant (38% of documents)
- 1: Somewhat relevant (mentions related concepts) (18% of documents)
- 2: Highly relevant (directly answers question) (44% of documents)

**Current Performance:** Average score ~1.06 on 0-2 scale

**Distribution:**
- Score 0: 38% of retrieved documents
- Score 1: 18% of retrieved documents
- Score 2: 44% of retrieved documents

### Medical Concept Coverage

**Definition:** Percentage of medical concepts in question covered by retrieved documents

**Current Performance:** 75.1%

**Interpretation:** Good concept coverage but poor precision ranking

### Guideline Coverage

**Definition:** Whether relevant guidelines are retrieved

**Current Performance:** 100.0%

**Interpretation:** Always retrieves some guideline, but not necessarily the most relevant

## Reasoning Metrics

### Chain Completeness

**Definition:** Measures if all reasoning steps are present

**Required Steps:**
1. Clinical presentation analysis
2. Symptom/sign identification
3. Differential diagnosis consideration
4. Guideline application
5. Answer selection with justification

**Current Performance:** 100.0%

**Interpretation:** All reasoning methods produce complete chains

### Evidence Utilization Rate

**Definition:** Percentage of retrieved evidence used in reasoning

**Current Performance:** 100.0%

**Interpretation:** High utilization of retrieved documents

### Reasoning Chain Completeness

**Definition:** Whether reasoning chain follows logical structure

**Current Performance:** 1.0 (perfect)

**Note:** Despite complete chains, reasoning can still be incorrect

### COT-TOT Delta

**Definition:** Difference in accuracy between Chain-of-Thought and Tree-of-Thought

**Current Performance:** 0.0

**Interpretation:** No difference observed in current evaluation

### Verifier Pass Rate

**Definition:** Percentage of cases where answer verification succeeds

**Current Performance:** 0.0

**Interpretation:** Answer verification not implemented or ineffective

## Calibration Metrics

### Brier Score

**Definition:** Measures accuracy of probabilistic predictions (lower is better)

**Formula:**
```
Brier Score = (1/N) Σ(p - y)²
where p = predicted probability, y = actual outcome (0 or 1)
```

**Current Performance:** 0.254

**Interpretation:** Moderate calibration, room for improvement

**Comparison by Method:**
- Chain-of-Thought: 0.424 (poor)
- Tree-of-Thought: 0.344 (moderate)
- **Structured Medical: 0.295 (best)**

### Expected Calibration Error (ECE)

**Definition:** Difference between confidence and accuracy across bins

**Bins:** 0-10%, 10-20%, ..., 90-100%

**Current Performance:** 0.179

**Interpretation:** Moderate miscalibration

**Comparison by Method:**
- Chain-of-Thought: 0.266
- Tree-of-Thought: 0.310
- Structured Medical: 0.283

### Confidence Distribution Analysis

**Current Distribution (50 cases):**
```
90-100%:  7 cases (14%) - Accuracy: 85.7%
80-90%:   1 case  (2%)  - Accuracy: 0.0%
70-80%:   2 cases (4%)  - Accuracy: 50.0%
60-70%:   0 cases (0%)
50-60%:   0 cases (0%)
40-50%:  11 cases (22%) - Accuracy: 63.6%
30-40%:  13 cases (26%) - Accuracy: 46.2%
20-30%:   4 cases (8%)  - Accuracy: 50.0%
10-20%:   7 cases (14%) - Accuracy: 42.9%
0-10%:    5 cases (10%) - Accuracy: 20.0%
```

**Key Issues:**
- 2 cases with >80% confidence but wrong answers (pitfall)
- Low confidence cases (0-20%) have poor accuracy (20-43%)
- Optimal confidence range: 40-90% (46-86% accuracy)

### Partial Credit Accuracy

**Definition:** Accuracy considering partial credit for close answers

**Current Performance:** 14.0%

**Interpretation:** System rarely gets "close" to correct answer

## Safety Metrics

### Hallucination Rate

**Definition:** Percentage of answers not grounded in provided context

**Current Performance:** 0.0%

**Interpretation:** Strict prompting prevents hallucination

### Dangerous Error Count

**Definition:** Answers that could lead to patient harm

**Current Performance:** 2 dangerous errors

**Interpretation:** Critical safety concern

### Contraindication Check Accuracy

**Definition:** Accuracy in identifying treatment contraindications

**Current Performance:** 0.0%

**Interpretation:** Critical gap - system doesn't check contraindications

### Urgency Recognition Accuracy

**Definition:** Accuracy in identifying urgent/emergent conditions

**Current Performance:** 0.0%

**Interpretation:** Critical gap - doesn't recognize time-sensitive conditions

### Safety Score

**Definition:** Composite safety metric (higher is better)

**Formula:** 
```
Safety_Score = 1 - (Hallucination_Rate + Dangerous_Error_Rate) / 2
```

**Current Performance:** 0.96

**Interpretation:** Good overall safety but critical gaps in specific areas

### "Cannot Answer" Misuse Rate

**Definition:** Rate of inappropriate "cannot answer" responses

**Current Performance:** 4.0% (2/50 cases)

**Interpretation:** Occasional inappropriate uncertainty

## Classification Metrics

### Per-Option Performance

```
Option A:
- Precision: 47.6%
- Recall: 90.9%
- F1: 62.5%
- Support: 11 cases

Option B:
- Precision: 40.0%
- Recall: 30.8%
- F1: 34.8%
- Support: 13 cases

Option C:
- Precision: 75.0%
- Recall: 64.3%
- F1: 69.2%
- Support: 14 cases

Option D:
- Precision: 71.4%
- Recall: 41.7%
- F1: 52.6%
- Support: 12 cases
```

### Macro Averages

- **Macro Precision:** 58.5%
- **Macro Recall:** 56.9%
- **Macro F1:** 54.8%
- **Weighted F1:** 54.8%
- **Balanced Accuracy:** 56.9%

### Confusion Matrix (Predicted vs True)

```
True\Pred |   A   B   C   D
----------------------------
A         |  10   0   0   1
B         |   7   4   1   1
C         |   3   2   9   0
D         |   1   4   2   5
```

**Key Patterns:**
- Option A: High recall (91%) but lower precision (48%)
- Option B: Worst performance (F1 34.8%)
- Option C: Best performance (F1 69.2%)
- Systematic confusion patterns visible

## Deep Error Analysis

### Error Categorization

#### Reasoning Errors (16 cases - 64% of errors)

**Description:** System retrieved relevant info but made incorrect reasoning

**Root Causes:**
- Insufficient chain-of-thought reasoning steps
- Failure to properly weight evidence from multiple sources
- Over-reliance on single retrieved document
- Missing critical symptom analysis

**Example:** Q_082 - 95% confidence but wrong answer

#### Knowledge Errors (8 cases - 32% of errors)

**Description:** System has incorrect medical knowledge or interpretation

**Root Causes:**
- Incorrect interpretation of medical guidelines
- Missing context about patient-specific factors
- Failure to consider contraindications
- Incorrect application of treatment protocols

**Example:** Q_032 - 7.1% confidence, wrong treatment selection

### Performance Segmentation Analysis

#### By Medical Category:

```
Gastroenterology:    71.4% accuracy (5/7)
Endocrine:           66.7% accuracy (4/6)
Cardiovascular:      54.5% accuracy (6/11)
Respiratory:         62.5% accuracy (5/8)
Nephrology:          66.7% accuracy (2/3)
Critical Care:      100.0% accuracy (1/1)
Hematology:          33.3% accuracy (1/3)
Rheumatology:        33.3% accuracy (1/3)
Psychiatry:          33.3% accuracy (1/3)
Infectious Disease:   0.0% accuracy (0/3) - Critical gap
Neurology:            0.0% accuracy (0/2) - Critical gap
```

#### By Question Type:

- **Diagnosis:** 52.2% accuracy (24/46)
- **Treatment:** 100.0% accuracy (2/2) - but only 2 cases
- **Other:** 0.0% accuracy (0/2)

#### By Complexity:

- **Simple:** 58.3% accuracy (7/12)
- **Moderate:** 52.0% accuracy (13/25)
- **Complex:** 46.2% accuracy (6/13)

#### By Relevance Level:

- **High relevance:** 44.8% accuracy (13/29)
- **Medium relevance:** 80.0% accuracy (8/10) - Best
- **Low relevance:** 45.5% accuracy (5/11)

#### By Confidence Range:

- **90-100%:** 85.7% accuracy (6/7) - High confidence usually correct
- **40-50%:** 63.6% accuracy (7/11)
- **30-40%:** 46.2% accuracy (6/13)
- **0-10%:** 20.0% accuracy (1/5) - Low confidence usually wrong

### Major Pitfalls Identified

#### Pitfall 1: Overconfident Wrong Answers

- **Count:** 2 cases (>80% confidence but incorrect)
- **Severity:** High
- **Examples:** Q_082 (95% confidence wrong), Q_085 (88% confidence wrong)
- **Solution:** Implement confidence calibration and uncertainty estimation

#### Pitfall 2: Missing Critical Symptoms

- **Count:** 20 cases (40% of total)
- **Severity:** Medium
- **Description:** Reasoning fails to consider important symptoms
- **Solution:** Enhance symptom extraction and ensure all symptoms considered

#### Pitfall 3: Medical Terminology Misunderstanding

- **Count:** 24 cases (48% of total)
- **Severity:** Medium
- **Description:** Fails to properly interpret medical abbreviations/terms
- **Solution:** Add medical terminology expansion and abbreviation resolution

### Root Cause Analysis

**Primary Causes of 52% Accuracy Ceiling:**

1. **Dataset Imbalance:**
   - Cardiology dominance (70% confusion matrix)
   - Diagnosis bias (92% diagnosis questions)
   - Missing treatment/safety scenarios

2. **Retrieval Precision Issues:**
   - 11.2% precision@5 too low
   - Relevant documents not ranked first
   - High recall but poor precision

3. **Reasoning Limitations:**
   - Insufficient differential diagnosis consideration
   - Missing symptom analysis
   - No contraindication checking

4. **Calibration Problems:**
   - Overconfident wrong answers (2 cases)
   - Poor low-confidence accuracy (20%)
   - Brier score 0.254 needs improvement

## Multi-Dimensional Visualization

### Performance Summary Dashboard

**Generated at:** `reports/charts/performance_summary.png`

**Components:**
- Accuracy Comparison: Tree-of-Thought vs other methods
- Retrieval Metrics: Precision-Recall tradeoff visualization
- Calibration Plot: Confidence vs accuracy across bins
- Category Performance: Heatmap by medical specialty
- Error Distribution: Pie chart of error types

### Confusion Matrix Visualization

**Generated at:** `reports/charts/confusion_matrix.png`

**Features:**
- 4×4 answer matrix (A/B/C/D predicted vs true)
- Color intensity shows frequency
- Green-bordered diagonal for correct predictions
- Annotations with counts and percentages
- Row/column totals

### Error Analysis Charts

**Generated at:** `reports/charts/error_analysis.png`

**Components:**
- Error Category Breakdown: Reasoning vs Knowledge errors
- Performance by Category: Bar chart of accuracy per medical specialty
- Confidence-Accuracy Plot: Scatter plot with calibration line
- Pitfalls Summary: Visual representation of major issues

### Category Performance Heatmap

**Shows performance across multiple dimensions:**
- **X-axis:** Medical categories (Cardio, GI, Endo, etc.)
- **Y-axis:** Question types (Diagnosis, Treatment, Management)
- **Color:** Accuracy percentage
- **Size:** Number of cases
- **Insights:** Clear patterns of strong/weak areas

## Evaluation Reports Structure

### JSON Output Format

**File:** `reports/evaluation_results.json`

```json
{
  "metadata": {
    "evaluation_date": "2025-12-11T04:06:17.777485",
    "total_cases": 50,
    "split": "all",
    "evaluation_time_seconds": 5432.80569434166,
    "reasoning_method": "Advanced-structured (Tree-of-Thought variant)"
  },
  
  "accuracy_metrics": {
    "exact_match_accuracy": 0.52,
    "semantic_accuracy": 0.52,
    "partial_credit_accuracy": 0.14,
    "correct_predictions": 26,
    "incorrect_predictions": 24
  },
  
  "retrieval_metrics": {
    "precision_at_k": {"1": 0.0, "3": 0.1067, "5": 0.1120, "10": 0.0798},
    "recall_at_k": {"1": 0.0, "3": 0.32, "5": 0.56, "10": 0.78},
    "map_score": 0.2677,
    "mrr": 0.2677,
    "context_relevance_scores": [...],
    "medical_concept_coverage": 0.7508,
    "guideline_coverage": 1.0,
    "avg_query_time_ms": 9.6
  },
  
  "reasoning_metrics": {
    "reasoning_chain_completeness": 1.0,
    "evidence_utilization_rate": 1.0,
    "hallucination_rate": 0.0,
    "cannot_answer_misuse_rate": 0.04,
    "method_accuracy": {"Advanced-structured": 0.52},
    "cot_tot_delta": 0.0,
    "verifier_pass_rate": 0.0
  },
  
  "calibration_metrics": {
    "brier_score": 0.2543,
    "expected_calibration_error": 0.1786,
    "confidence_distribution": {
      "90-100%": 7, "40-50%": 11, "0-10%": 5,
      "30-40%": 13, "70-80%": 2, "20-30%": 4,
      "10-20%": 7, "80-90%": 1
    }
  },
  
  "safety_metrics": {
    "dangerous_error_count": 2,
    "contraindication_check_accuracy": 0.0,
    "urgency_recognition_accuracy": 0.0,
    "safety_score": 0.96
  },
  
  "classification_metrics": {
    "macro_precision": 0.5851,
    "macro_recall": 0.5691,
    "macro_f1": 0.5479,
    "weighted_f1": 0.5481,
    "balanced_accuracy": 0.5691,
    "confusion_matrix": {...},
    "option_distribution": {...}
  },
  
  "performance_segmentation": {
    "by_category": {...},
    "by_question_type": {...},
    "by_complexity": {...},
    "by_relevance_level": {...},
    "by_confidence_range": {...}
  },
  
  "error_analysis": {
    "error_categories": {
      "reasoning": {
        "count": 16,
        "root_causes": [...],
        "proposed_solutions": [...]
      },
      "knowledge": {
        "count": 8,
        "root_causes": [...],
        "proposed_solutions": [...]
      }
    },
    "pitfalls": [
      {
        "pitfall": "Overconfident Wrong Answers",
        "count": 2,
        "severity": "high",
        "examples": [...],
        "solution": "Implement confidence calibration..."
      }
    ]
  },
  
  "recommendations": [
    "Enhance reasoning chain completeness with more structured steps",
    "Implement evidence aggregation with confidence weighting",
    "Expand medical knowledge base with additional guidelines",
    "Add medical safety checks for contraindications",
    "Implement confidence calibration to reduce overconfident wrong answers",
    "Add query expansion with medical terminology",
    "Improve cross-encoder reranking with medical domain fine-tuning",
    "Implement active learning to identify difficult cases"
  ],
  
  "chart_paths": {
    "performance_summary": "reports/charts/performance_summary.png",
    "confusion_matrix": "reports/charts/confusion_matrix.png",
    "error_analysis": "reports/charts/error_analysis.png"
  }
}
```

## Evaluation Tools and Scripts

### Main Evaluation Script

```bash
# Run comprehensive evaluation
python src/evaluation/evaluate_pipeline.py \
    --dataset data/processed/questions/questions_1.json \
    --output reports/evaluation_results.json \
    --charts-dir reports/charts/ \
    --reasoning-method tree_of_thought \
    --top-k 25

# Options:
# --reasoning-method: chain_of_thought, tree_of_thought, structured_medical
# --top-k: Number of documents to retrieve (default: 25)
# --confidence-calibration: Apply calibration model
# --save-reasoning-chains: Save detailed reasoning for analysis
```

### Comparative Evaluation

```bash
# Compare all reasoning methods
python src/evaluation/compare_reasoning_methods.py \
    --dataset data/processed/questions/questions_1.json \
    --methods chain_of_thought tree_of_thought structured_medical \
    --output reports/reasoning_comparison.json

# Generate comparison charts
python src/evaluation/visualize_comparison.py \
    --input reports/reasoning_comparison.json \
    --output reports/charts/method_comparison.png
```

### Error Analysis Tool

```python
# Deep error analysis
from src.evaluation.error_analyzer import ErrorAnalyzer

analyzer = ErrorAnalyzer(evaluation_results)
analysis = analyzer.analyze()

# Get specific insights
print(f"Major pitfalls: {analysis['pitfalls']}")
print(f"Worst performing categories: {analysis['weak_categories']}")
print(f"Confidence calibration issues: {analysis['calibration_issues']}")

# Generate improvement recommendations
recommendations = analyzer.generate_recommendations(
    focus_areas=['accuracy', 'safety', 'calibration']
)
```

## Continuous Evaluation Framework

### Automated Evaluation Pipeline

```yaml
# config/evaluation_pipeline.yaml
evaluation:
  schedule: "weekly"  # or "on-commit", "daily"
  
  datasets:
    - name: "main_dataset"
      path: "data/processed/questions/questions_1.json"
      split: "all"
    
    - name: "validation_set"
      path: "data/validation/validation_set.json"
      split: "validation"
  
  metrics:
    required:
      - "accuracy"
      - "precision_at_5"
      - "recall_at_5"
      - "brier_score"
      - "safety_score"
    
    optional:
      - "reasoning_coherence"
      - "hallucination_rate"
      - "response_time"
  
  thresholds:
    accuracy: {min: 0.50, target: 0.65}
    precision_at_5: {min: 0.10, target: 0.16}
    brier_score: {max: 0.30, target: 0.22}
    safety_score: {min: 0.95, target: 0.98}
  
  alerts:
    - metric: "accuracy"
      condition: "< 0.45"
      severity: "critical"
      action: "notify_team"
    
    - metric: "dangerous_error_count"
      condition: "> 0"
      severity: "critical"
      action: "block_deployment"
    
    - metric: "precision_at_5"
      condition: "< 0.08"
      severity: "warning"
      action: "log_issue"
```

### Performance Monitoring Dashboard

**Components:**
- Real-time Metrics: Accuracy, precision, recall, safety score
- Trend Analysis: Performance over time
- Error Tracking: New error patterns detection
- A/B Testing: Compare optimization strategies
- Alert System: Notify on performance degradation

## Related Documentation

- **Retrieval Evaluation Results** - Detailed retrieval performance
- **Reasoning Method Comparison** - Method performance analysis
- **Error Analysis Report** - Deep dive into errors
- **Improvement Recommendations** - Based on evaluation findings

## Key Insights from Evaluation

### Strengths

- Good reasoning chain completeness (100%)
- High evidence utilization (100%)
- No hallucination (0%)
- Good safety score (0.96)
- Tree-of-Thought best method (52% accuracy)

### Critical Issues

- Low retrieval precision (11.2% @5)
- No contraindication checking (0% accuracy)
- No urgency recognition (0% accuracy)
- Overconfident wrong answers (2 cases)
- Specialty imbalance (Infectious Disease, Neurology 0%)

## Optimization Priorities

1. **Immediate:** Improve retrieval precision (target 16%)
2. **High:** Add safety checks (contraindications, urgency)
3. **High:** Implement confidence calibration
4. **Medium:** Address dataset imbalances
5. **Medium:** Enhance symptom extraction

## Success Metrics for Next Evaluation

```
Target Metrics for Next Evaluation:
- Overall Accuracy: 58% (+6% improvement)
- Precision@5: 14% (+2.8% improvement)
- Brier Score: 0.23 (-0.024 improvement)
- Safety Metrics: >0% for contraindication/urgency
- Overconfident Errors: 0 (eliminate)
```

---

**Documentation Author:** Shreya Uprety  
**Evaluation Reference:** 2025-12-11 Comprehensive Medical QA Evaluation