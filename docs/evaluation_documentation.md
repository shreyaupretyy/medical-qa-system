# Evaluation Documentation

**Author:** Shreya Uprety  
**Last Updated:** December 11, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Evaluation Pipeline](#evaluation-pipeline)
3. [Metrics Calculator](#metrics-calculator)
4. [Question Analyzer](#question-analyzer)
5. [Results Visualizer](#results-visualizer)
6. [Actual Results](#actual-results)

---

## Overview

The `src/evaluation/` module provides comprehensive metrics calculation, error analysis, and visualization for the Medical QA system.

**Evaluation Scope:**
- 50 clinical cases (new dataset)
- Multiple reasoning methods (CoT, ToT, Structured)
- Multiple retrieval strategies (6 total)
- Calibration and safety metrics

---

## Evaluation Pipeline

**File:** `src/evaluation/eval_pipeline.py`

### Purpose

Orchestrates end-to-end evaluation: question answering, metrics calculation, error analysis, visualization.

### Architecture

```
Dataset (50 cases)
     ↓
Run Evaluation (answer all questions)
     ↓
Calculate Metrics (accuracy, retrieval, calibration, timing)
     ↓
Analyze Errors (categorize failures, identify patterns)
     ↓
Generate Visualizations (charts, confusion matrix)
     ↓
Export Results (JSON, CSV, charts)
```

### Implementation

```python
class EvaluationPipeline:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        config: Dict
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            rag_pipeline: RAG pipeline instance to evaluate
            config: Evaluation configuration
            
        Components:
            - Metrics calculator
            - Question analyzer
            - Results visualizer
        """
        self.rag_pipeline = rag_pipeline
        self.config = config
        
        self.metrics_calculator = MetricsCalculator()
        self.question_analyzer = QuestionAnalyzer()
        self.results_visualizer = ResultsVisualizer()
```

### Main Evaluation

```python
def evaluate_dataset(
    self,
    dataset_path: str,
    output_dir: str
) -> Dict:
    """
    Evaluate entire dataset.
    
    Args:
        dataset_path: Path to evaluation dataset (JSON)
        output_dir: Directory for results/visualizations
        
    Returns:
        {
            'overall_metrics': {...},
            'per_question_results': [...],
            'error_analysis': {...},
            'visualizations': [...]
        }
        
    Process:
        1. Load dataset
        2. Run evaluation (answer all questions)
        3. Calculate metrics
        4. Analyze errors
        5. Generate visualizations
        6. Export results
    """
    # Step 1: Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} questions")
    
    # Step 2: Run evaluation
    results = []
    for i, item in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}: {item['id']}")
        
        start_time = time.time()
        
        # Answer question
        answer_result = self.rag_pipeline.answer_question(
            case=item['clinical_case'],
            question=item['question'],
            options=item['options']
        )
        
        end_time = time.time()
        
        # Record result
        results.append({
            'question_id': item['id'],
            'correct_answer': item['correct_answer'],
            'predicted_answer': answer_result['answer'],
            'is_correct': answer_result['answer'] == item['correct_answer'],
            'confidence': answer_result['confidence'],
            'reasoning': answer_result['reasoning'],
            'evidence': answer_result['evidence'],
            'time_ms': (end_time - start_time) * 1000,
            'specialty': item.get('specialty', 'general'),
            'difficulty': item.get('difficulty', 'medium')
        })
    
    # Step 3: Calculate metrics
    overall_metrics = self.metrics_calculator.calculate_all(results, dataset)
    
    # Step 4: Error analysis
    error_analysis = self.question_analyzer.analyze_errors(results, dataset)
    
    # Step 5: Generate visualizations
    visualizations = self.results_visualizer.generate_all(
        results,
        overall_metrics,
        error_analysis,
        output_dir
    )
    
    # Step 6: Export results
    self._export_results(
        overall_metrics,
        results,
        error_analysis,
        output_dir
    )
    
    return {
        'overall_metrics': overall_metrics,
        'per_question_results': results,
        'error_analysis': error_analysis,
        'visualizations': visualizations
    }
```

---

## Metrics Calculator

**File:** `src/evaluation/metrics_calculator.py`

### Purpose

Computes 23 evaluation metrics across 5 categories: accuracy, retrieval, reasoning, calibration, timing.

### Categories

1. **Accuracy Metrics** (5 metrics)
2. **Retrieval Metrics** (6 metrics)
3. **Reasoning Metrics** (4 metrics)
4. **Calibration Metrics** (4 metrics)
5. **Timing Metrics** (4 metrics)

### Implementation

```python
class MetricsCalculator:
    def calculate_all(
        self,
        results: List[Dict],
        dataset: List[Dict]
    ) -> Dict:
        """
        Calculate all evaluation metrics.
        
        Returns:
            {
                'accuracy': {...},
                'retrieval': {...},
                'reasoning': {...},
                'calibration': {...},
                'timing': {...},
                'aggregate': {...}
            }
        """
        metrics = {
            'accuracy': self.calculate_accuracy_metrics(results),
            'retrieval': self.calculate_retrieval_metrics(results, dataset),
            'reasoning': self.calculate_reasoning_metrics(results),
            'calibration': self.calculate_calibration_metrics(results),
            'timing': self.calculate_timing_metrics(results)
        }
        
        # Aggregate score
        metrics['aggregate'] = {
            'overall_score': self._compute_aggregate_score(metrics)
        }
        
        return metrics
```

### 1. Accuracy Metrics

```python
def calculate_accuracy_metrics(self, results: List[Dict]) -> Dict:
    """
    Calculate accuracy-related metrics.
    
    Returns:
        {
            'overall_accuracy': float,
            'per_specialty_accuracy': Dict[str, float],
            'per_difficulty_accuracy': Dict[str, float],
            'confidence_by_correctness': Dict[str, float],
            'confusion_matrix': np.ndarray
        }
    """
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    
    accuracy_metrics = {
        'overall_accuracy': correct / total,
        'total_questions': total,
        'correct': correct,
        'incorrect': total - correct
    }
    
    # Per-specialty accuracy
    specialties = set(r['specialty'] for r in results)
    per_specialty = {}
    for specialty in specialties:
        specialty_results = [r for r in results if r['specialty'] == specialty]
        specialty_correct = sum(1 for r in specialty_results if r['is_correct'])
        per_specialty[specialty] = specialty_correct / len(specialty_results)
    
    accuracy_metrics['per_specialty_accuracy'] = per_specialty
    
    # Per-difficulty accuracy
    difficulties = set(r['difficulty'] for r in results)
    per_difficulty = {}
    for difficulty in difficulties:
        diff_results = [r for r in results if r['difficulty'] == difficulty]
        diff_correct = sum(1 for r in diff_results if r['is_correct'])
        per_difficulty[difficulty] = diff_correct / len(diff_results)
    
    accuracy_metrics['per_difficulty_accuracy'] = per_difficulty
    
    # Confidence by correctness
    correct_confidences = [r['confidence'] for r in results if r['is_correct']]
    incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]
    
    accuracy_metrics['confidence_by_correctness'] = {
        'correct': np.mean(correct_confidences),
        'incorrect': np.mean(incorrect_confidences)
    }
    
    # Confusion matrix
    accuracy_metrics['confusion_matrix'] = self._compute_confusion_matrix(results)
    
    return accuracy_metrics
```

**Actual Results:**
- Overall Accuracy: **54%** (27/50 correct)
- Cardiovascular: 52% (11/21)
- Respiratory: 58% (7/12)
- GI: 50% (3/6)
- Infectious: 67% (2/3)

### 2. Retrieval Metrics

```python
def calculate_retrieval_metrics(
    self,
    results: List[Dict],
    dataset: List[Dict]
) -> Dict:
    """
    Calculate retrieval quality metrics.
    
    Returns:
        {
            'precision_at_k': Dict[int, float],
            'recall_at_k': Dict[int, float],
            'mean_average_precision': float,
            'mean_reciprocal_rank': float,
            'context_relevance': float,
            'avg_documents_retrieved': float
        }
    """
    all_precisions = {k: [] for k in [1, 3, 5, 10]}
    all_recalls = {k: [] for k in [1, 3, 5, 10]}
    average_precisions = []
    reciprocal_ranks = []
    relevance_scores = []
    doc_counts = []
    
    for result, item in zip(results, dataset):
        # Get retrieved documents and ground truth
        retrieved = result['evidence']
        relevant = item.get('relevant_guidelines', [])
        
        doc_counts.append(len(retrieved))
        
        # Calculate precision and recall at k
        for k in [1, 3, 5, 10]:
            retrieved_k = retrieved[:k]
            relevant_retrieved = [doc for doc in retrieved_k if self._is_relevant(doc, relevant)]
            
            precision = len(relevant_retrieved) / k if k > 0 else 0
            recall = len(relevant_retrieved) / len(relevant) if len(relevant) > 0 else 0
            
            all_precisions[k].append(precision)
            all_recalls[k].append(recall)
        
        # Calculate average precision
        ap = self._calculate_average_precision(retrieved, relevant)
        average_precisions.append(ap)
        
        # Calculate reciprocal rank
        rr = self._calculate_reciprocal_rank(retrieved, relevant)
        reciprocal_ranks.append(rr)
        
        # Calculate context relevance (semantic similarity)
        relevance = self._calculate_context_relevance(result['reasoning'], retrieved)
        relevance_scores.append(relevance)
    
    return {
        'precision_at_k': {k: np.mean(all_precisions[k]) for k in [1, 3, 5, 10]},
        'recall_at_k': {k: np.mean(all_recalls[k]) for k in [1, 3, 5, 10]},
        'mean_average_precision': np.mean(average_precisions),
        'mean_reciprocal_rank': np.mean(reciprocal_ranks),
        'context_relevance': np.mean(relevance_scores),
        'avg_documents_retrieved': np.mean(doc_counts)
    }
```

**Actual Results (50 cases):**
- Precision@1: **0.0%**
- Precision@3: **9.3%**
- Precision@5: **10.8%**
- Precision@10: **7.8%**
- Recall@3: **28%**
- Recall@5: **54%**
- Recall@10: **76%**
- MAP: **0.252**
- MRR: **0.252**
- Context Relevance: **0.70** (avg)

### 3. Reasoning Metrics

```python
def calculate_reasoning_metrics(self, results: List[Dict]) -> Dict:
    """
    Calculate reasoning quality metrics.
    
    Returns:
        {
            'avg_reasoning_length': float,
            'reasoning_coherence': float,
            'evidence_usage': float,
            'hallucination_rate': float
        }
    """
    reasoning_lengths = []
    coherence_scores = []
    evidence_usage_scores = []
    hallucination_count = 0
    
    for result in results:
        reasoning = result['reasoning']
        evidence = result['evidence']
        
        # Reasoning length (words)
        reasoning_lengths.append(len(reasoning.split()))
        
        # Coherence (% of reasoning backed by evidence)
        coherence = self._calculate_coherence(reasoning, evidence)
        coherence_scores.append(coherence)
        
        # Evidence usage (% of evidence cited in reasoning)
        usage = self._calculate_evidence_usage(reasoning, evidence)
        evidence_usage_scores.append(usage)
        
        # Hallucination detection
        has_hallucination = self._detect_hallucination(reasoning, evidence)
        if has_hallucination:
            hallucination_count += 1
    
    return {
        'avg_reasoning_length': np.mean(reasoning_lengths),
        'reasoning_coherence': np.mean(coherence_scores),
        'evidence_usage': np.mean(evidence_usage_scores),
        'hallucination_rate': hallucination_count / len(results)
    }
```

**Actual Results:**
- CoT: 98 words avg, coherence 32.7%
- ToT: 525 words avg, coherence 43.3%
- Structured: 41 words avg, coherence 32.0%

### 4. Calibration Metrics

```python
def calculate_calibration_metrics(self, results: List[Dict]) -> Dict:
    """
    Calculate calibration metrics (confidence vs accuracy alignment).
    
    Returns:
        {
            'brier_score': float,
            'expected_calibration_error': float,
            'max_calibration_error': float,
            'calibration_curve': Dict
        }
    """
    confidences = np.array([r['confidence'] for r in results])
    correctness = np.array([1 if r['is_correct'] else 0 for r in results])
    
    # Brier score (lower is better, measures calibration)
    brier_score = np.mean((confidences - correctness) ** 2)
    
    # Expected Calibration Error (ECE)
    ece = self._calculate_ece(confidences, correctness)
    
    # Max Calibration Error (MCE)
    mce = self._calculate_mce(confidences, correctness)
    
    # Calibration curve (bin confidences and compute accuracy)
    calibration_curve = self._compute_calibration_curve(confidences, correctness)
    
    return {
        'brier_score': brier_score,
        'expected_calibration_error': ece,
        'max_calibration_error': mce,
        'calibration_curve': calibration_curve
    }
```

**ECE Calculation:**

```python
def _calculate_ece(
    self,
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error.
    
    ECE = Σ (|bin_confidence - bin_accuracy| × bin_weight)
    
    Args:
        confidences: Predicted confidences (0-1)
        correctness: Binary correctness (0 or 1)
        n_bins: Number of bins for discretization
    
    Returns:
        ECE score (0-1, lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this confidence bin
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = (confidences >= lower) & (confidences < upper)
        
        if np.sum(in_bin) > 0:
            # Compute average confidence and accuracy in bin
            bin_confidence = np.mean(confidences[in_bin])
            bin_accuracy = np.mean(correctness[in_bin])
            bin_weight = np.sum(in_bin) / len(confidences)
            
            # Add weighted calibration error
            ece += np.abs(bin_confidence - bin_accuracy) * bin_weight
    
    return ece
```

**Actual Results:**
- CoT: Brier **0.424**, ECE **0.266**
- ToT: Brier **0.344**, ECE **0.310**
- Structured: Brier **0.295** (best), ECE **0.283**

### 5. Timing Metrics

```python
def calculate_timing_metrics(self, results: List[Dict]) -> Dict:
    """
    Calculate timing metrics.
    
    Returns:
        {
            'avg_time_ms': float,
            'median_time_ms': float,
            'p95_time_ms': float,
            'total_time_seconds': float
        }
    """
    times = [r['time_ms'] for r in results]
    
    return {
        'avg_time_ms': np.mean(times),
        'median_time_ms': np.median(times),
        'p95_time_ms': np.percentile(times, 95),
        'total_time_seconds': np.sum(times) / 1000
    }
```

**Actual Results:**
- CoT: **4,955ms** avg (fastest)
- Structured: **26,991ms** avg
- ToT: **41,367ms** avg (slowest, 8.4x CoT)

---

## Question Analyzer

**File:** `src/evaluation/question_analyzer.py`

### Purpose

Analyzes error patterns, categorizes failures, and identifies improvement opportunities.

### Error Analysis

```python
class QuestionAnalyzer:
    def analyze_errors(
        self,
        results: List[Dict],
        dataset: List[Dict]
    ) -> Dict:
        """
        Comprehensive error analysis.
        
        Returns:
            {
                'total_errors': int,
                'error_by_type': Dict,
                'error_by_specialty': Dict,
                'common_pitfalls': List,
                'retrieval_failures': List,
                'reasoning_failures': List
            }
        """
        errors = [r for r in results if not r['is_correct']]
        
        analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(results),
            'error_by_type': self._categorize_by_type(errors),
            'error_by_specialty': self._categorize_by_specialty(errors),
            'common_pitfalls': self._identify_pitfalls(errors, dataset),
            'retrieval_failures': self._identify_retrieval_failures(errors),
            'reasoning_failures': self._identify_reasoning_failures(errors)
        }
        
        return analysis
```

### Failure Categorization

```python
def _categorize_by_type(self, errors: List[Dict]) -> Dict:
    """
    Categorize errors by failure type.
    
    Types:
        - retrieval_failure: Relevant context not retrieved
        - reasoning_failure: Correct context but wrong conclusion
        - hallucination: Answer not supported by context
        - low_confidence: Correct context but low confidence
    """
    categorized = {
        'retrieval_failure': 0,
        'reasoning_failure': 0,
        'hallucination': 0,
        'low_confidence': 0
    }
    
    for error in errors:
        # Check if relevant context was retrieved
        has_relevant_context = self._has_relevant_context(error)
        
        if not has_relevant_context:
            categorized['retrieval_failure'] += 1
        else:
            # Check for hallucination
            has_hallucination = self._detect_hallucination(
                error['reasoning'],
                error['evidence']
            )
            
            if has_hallucination:
                categorized['hallucination'] += 1
            elif error['confidence'] < 0.5:
                categorized['low_confidence'] += 1
            else:
                categorized['reasoning_failure'] += 1
    
    return categorized
```

**Actual Results (50 cases):**
- Total Errors: **23** (46% error rate)
- Retrieval Failures: **0** (0%)
- Reasoning Failures: **23** (100%)
- Hallucination: 0
- Low Confidence: 0

### Common Pitfalls

```python
def _identify_pitfalls(
    self,
    errors: List[Dict],
    dataset: List[Dict]
) -> List[Dict]:
    """
    Identify common reasoning pitfalls.
    
    Pitfalls:
        - Incomplete differential diagnosis
        - Missing critical symptoms
        - Misinterpreting medical terminology
        - Ignoring risk factors
        - Over-reliance on single symptom
    """
    pitfalls = {
        'incomplete_differential': 0,
        'missing_symptoms': 0,
        'terminology_misunderstanding': 0,
        'ignored_risk_factors': 0,
        'single_symptom_focus': 0
    }
    
    for error in errors:
        reasoning = error['reasoning'].lower()
        
        # Check for incomplete differential
        if 'differential' not in reasoning or reasoning.count('diagnosis') < 2:
            pitfalls['incomplete_differential'] += 1
        
        # Check for missing symptoms (compare to case)
        question_idx = next(i for i, item in enumerate(dataset) if item['id'] == error['question_id'])
        case = dataset[question_idx]['clinical_case'].lower()
        
        key_symptoms = ['chest pain', 'dyspnea', 'fever', 'nausea', 'headache']
        mentioned_symptoms = [s for s in key_symptoms if s in case]
        cited_symptoms = [s for s in mentioned_symptoms if s in reasoning]
        
        if len(cited_symptoms) < len(mentioned_symptoms) * 0.5:
            pitfalls['missing_symptoms'] += 1
        
        # Check for terminology issues
        if 'unclear' in reasoning or 'uncertain' in reasoning or 'not sure' in reasoning:
            pitfalls['terminology_misunderstanding'] += 1
    
    # Format as list with counts
    pitfall_list = [
        {'pitfall': k, 'count': v, 'percentage': v / len(errors)}
        for k, v in pitfalls.items()
    ]
    
    return sorted(pitfall_list, key=lambda x: x['count'], reverse=True)
```

**Actual Results:**
- Incomplete Differential: **23/23** (100%)
- Missing Symptoms: **20/23** (87%)
- Terminology Misunderstanding: **23/23** (100%)

---

## Results Visualizer

**File:** `src/evaluation/visualizer.py`

### Purpose

Generates charts, confusion matrices, and calibration curves for evaluation results.

### Generated Visualizations

1. **Accuracy by Specialty** (bar chart)
2. **Confusion Matrix** (heatmap)
3. **Calibration Curve** (reliability diagram)
4. **Retrieval Performance** (precision/recall curves)
5. **Reasoning Method Comparison** (grouped bar chart)
6. **Error Distribution** (pie chart)

### Implementation

```python
class ResultsVisualizer:
    def generate_all(
        self,
        results: List[Dict],
        metrics: Dict,
        error_analysis: Dict,
        output_dir: str
    ) -> List[str]:
        """
        Generate all visualizations.
        
        Returns:
            List of generated chart file paths
        """
        visualizations = []
        
        # 1. Accuracy by specialty
        visualizations.append(
            self.plot_accuracy_by_specialty(
                metrics['accuracy']['per_specialty_accuracy'],
                output_dir
            )
        )
        
        # 2. Confusion matrix
        visualizations.append(
            self.plot_confusion_matrix(
                metrics['accuracy']['confusion_matrix'],
                output_dir
            )
        )
        
        # 3. Calibration curve
        visualizations.append(
            self.plot_calibration_curve(
                metrics['calibration']['calibration_curve'],
                output_dir
            )
        )
        
        # 4. Retrieval performance
        visualizations.append(
            self.plot_retrieval_performance(
                metrics['retrieval'],
                output_dir
            )
        )
        
        # 5. Error distribution
        visualizations.append(
            self.plot_error_distribution(
                error_analysis['error_by_type'],
                output_dir
            )
        )
        
        return visualizations
```

### Calibration Curve

```python
def plot_calibration_curve(
    self,
    calibration_data: Dict,
    output_dir: str
) -> str:
    """
    Plot calibration curve (reliability diagram).
    
    X-axis: Predicted confidence
    Y-axis: Actual accuracy
    
    Perfect calibration: y = x (diagonal line)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract data
    bin_confidences = calibration_data['bin_confidences']
    bin_accuracies = calibration_data['bin_accuracies']
    
    # Plot calibration curve
    ax.plot(bin_confidences, bin_accuracies, 'o-', label='Model', linewidth=2)
    
    # Plot perfect calibration (y=x)
    ax.plot([0, 1], [0, 1], '--', label='Perfect Calibration', color='gray')
    
    # Formatting
    ax.set_xlabel('Predicted Confidence', fontsize=12)
    ax.set_ylabel('Actual Accuracy', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Save
    output_path = os.path.join(output_dir, 'calibration_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
```

---

## Actual Results

### Overall Performance

**Evaluation Dataset:** 50 clinical cases
**Total Time:** 5,680 seconds (~95 minutes)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 54% (27/50) |
| Retrieval P@5 | 10.8% |
| Retrieval R@5 | 54% |
| MAP | 0.252 |
| MRR | 0.252 |
| Avg Time | 11,361ms |

### By Reasoning Method

| Method | Accuracy | Time (ms) | Brier | ECE |
|--------|----------|-----------|-------|-----|
| Tree-of-Thought | **52%** | 41,367 | 0.344 | 0.310 |
| Structured | 44% | 26,991 | **0.295** | 0.283 |
| Chain-of-Thought | 34% | **4,955** | 0.424 | **0.266** |

### By Specialty

| Specialty | Accuracy | Errors | Common Pitfalls |
|-----------|----------|--------|-----------------|
| Cardiovascular | 52% (11/21) | 10 | Incomplete differential (100%) |
| Respiratory | 58% (7/12) | 5 | Missing symptoms (80%) |
| GI | 50% (3/6) | 3 | Terminology (100%) |
| Infectious | 67% (2/3) | 1 | - |
| Renal | 0% (0/2) | 2 | All errors |
| Metabolic | 50% (1/2) | 1 | - |

### Error Analysis

**Total Errors:** 23 (46% error rate)

**Failure Type Breakdown:**
- Retrieval Failures: **0** (0%)
- Reasoning Failures: **23** (100%)

**Common Pitfalls:**
1. Incomplete differential diagnosis: 23/23 (100%)
2. Missing symptoms in reasoning: 20/23 (87%)
3. Medical terminology misunderstanding: 23/23 (100%)

**Key Finding:** All errors are reasoning failures, not retrieval failures. This indicates the bottleneck is LLM reasoning quality, not retrieval quality.

---

## Usage

### Running Evaluation

```bash
# Evaluate full dataset
python scripts/evaluate_new_dataset.py \
    --dataset data/processed/questions/clinical_cases_v5.json \
    --output reports/ \
    --reasoning_method hybrid

# Compare reasoning methods
python scripts/compare_reasoning_methods.py \
    --dataset data/processed/questions/clinical_cases_v5.json \
    --output reports/reasoning_method_comparison.json

# Compare retrieval strategies
python scripts/compare_retrieval_strategies.py \
    --dataset data/processed/questions/clinical_cases_v5.json \
    --output reports/retrieval_strategy_comparison.json
```

### Interpreting Results

**High Accuracy (>50%):** Good performance
**High Brier Score (>0.4):** Poor calibration (overconfident or underconfident)
**High ECE (>0.3):** Confidence doesn't match accuracy
**High MAP (>0.3):** Good retrieval quality
**High Recall@5 (>60%):** Retrieves relevant context

---

## Related Documentation

- [Part 3: Evaluation Framework](part_3_evaluation_framework.md)
- [Part 4: Experiments](part_4_experiments.md)
- [Reasoning Documentation](reasoning_documentation.md)

---

**Documentation Author:** Shreya Uprety
