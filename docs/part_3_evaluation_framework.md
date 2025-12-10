# Part 3: Evaluation Framework

**Author:** Shreya Uprety  
**Last Updated:** December 11, 2025

---

## Overview

This document details the comprehensive evaluation framework for the Medical Question-Answering System, including all metrics, error analysis, and visualization tools.

---

## Evaluation Pipeline

```
Clinical Question + Gold Answer
        ↓
RAG Pipeline (Retrieval + Reasoning)
        ↓
Predicted Answer + Reasoning Chain + Confidence
        ↓
┌─────────────────────────────────────────────┐
│         Metrics Calculation                  │
│  ├─ Accuracy Metrics                        │
│  ├─ Retrieval Metrics                       │
│  ├─ Reasoning Metrics                       │
│  ├─ Calibration Metrics                     │
│  └─ Safety Metrics                          │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│         Error Analysis                       │
│  ├─ Condition Confusion Matrix              │
│  ├─ Error Categorization                    │
│  ├─ Concept-Level Analysis                  │
│  └─ High-Confidence Errors                  │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│         Visualization                        │
│  ├─ Confusion Matrix (PNG)                  │
│  ├─ Performance Summary (PNG)               │
│  └─ Error Analysis Charts (PNG)             │
└─────────────────┬───────────────────────────┘
                  ↓
       Evaluation Report (JSON)
```

---

## Accuracy Metrics

### Exact Match Accuracy

**Definition:** Percentage of questions with correct answer prediction

**Formula:**
```
Accuracy = (Correct Predictions / Total Questions) × 100
```

**Implementation:**
```python
def calculate_accuracy(predictions, gold_answers):
    correct = sum(1 for pred, gold in zip(predictions, gold_answers) if pred == gold)
    return correct / len(predictions)
```

**Current Performance:** 54% (hybrid reasoning on 50 cases)  
**Target:** 80% (achievable with medical embeddings + PubMedBERT)

---

### Semantic Accuracy

**Definition:** Measures answer similarity even if not exact match

**Use Case:** Allows partial credit for close answers

**Formula:**
```
Semantic_Accuracy = avg(cosine_similarity(pred_embedding, gold_embedding))
```

**Not currently used** (multiple-choice format requires exact match)

---

## Retrieval Metrics

### Precision@k

**Definition:** Percentage of retrieved documents that are relevant

**Formula:**
```
Precision@k = (Relevant Documents in Top k / k) × 100
```

**Implementation:**
```python
def precision_at_k(retrieved_docs, relevant_docs, k=5):
    top_k = retrieved_docs[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_docs))
    return relevant_in_top_k / k
```

**Current Performance:**
- Precision@1: 0%
- Precision@3: 9.3%
- Precision@5: 10.8%
- Precision@10: 7.8%

---

### Recall@k

**Definition:** Percentage of relevant documents retrieved

**Formula:**
```
Recall@k = (Relevant Documents in Top k / Total Relevant) × 100
```

**Implementation:**
```python
def recall_at_k(retrieved_docs, relevant_docs, k=5):
    top_k = retrieved_docs[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_docs))
    return relevant_in_top_k / len(relevant_docs)
```

**Current Performance:**
- Recall@1: 0%
- Recall@3: 28%
- Recall@5: 54%
- Recall@10: 76%

---

### Mean Average Precision (MAP)

**Definition:** Average precision across all queries

**Formula:**
```
AP = (1/R) Σ(Precision@k × Relevance@k)
MAP = avg(AP across all queries)
```

**Current Performance:** 0.252

---

### Mean Reciprocal Rank (MRR)

**Definition:** Average of reciprocal ranks of first relevant document

**Formula:**
```
RR = 1 / rank_of_first_relevant_document
MRR = avg(RR across all queries)
```

**Current Performance:** 0.252

---

### Context Relevance Score

**Definition:** Measures how relevant retrieved context is to the question

**Scale:** 0 (irrelevant) to 2 (highly relevant)

**Scoring:**
- 0: Completely irrelevant
- 1: Somewhat relevant (mentions related concepts)
- 2: Highly relevant (directly answers question)

**Implementation:**
```python
def calculate_context_relevance(context, question):
    # Extract key concepts from question
    question_concepts = extract_concepts(question)
    
    # Count concept matches in context
    matches = sum(1 for concept in question_concepts if concept in context)
    coverage = matches / len(question_concepts)
    
    # Scale to 0-2
    if coverage >= 0.8:
        return 2.0
    elif coverage >= 0.4:
        return 1.0
    else:
        return 0.0
```

**Current Performance:** Average 0.70 (moderate relevance, scale 0-2)

---

## Reasoning Metrics

### Chain Completeness

**Definition:** Measures if all reasoning steps are present

**Required Steps:**
1. Clinical presentation analysis
2. Symptom/sign identification
3. Differential diagnosis consideration
4. Guideline application
5. Answer selection with justification

**Implementation:**
```python
def calculate_chain_completeness(reasoning_chain):
    required_steps = [
        "clinical presentation",
        "symptoms|signs|findings",
        "differential|diagnosis|condition",
        "guideline|protocol|recommendation",
        "answer|conclusion|therefore"
    ]
    
    completeness = 0
    for step in required_steps:
        if re.search(step, reasoning_chain, re.IGNORECASE):
            completeness += 1
    
    return completeness / len(required_steps)
```

**Current Performance:** 100% (all methods produce complete chains)

---

### Evidence Utilization Rate

**Definition:** Percentage of retrieved evidence used in reasoning

**Formula:**
```
Evidence_Utilization = (Documents Referenced in Reasoning / Total Retrieved) × 100
```

**Implementation:**
```python
def calculate_evidence_utilization(reasoning_chain, retrieved_docs):
    referenced_docs = 0
    for doc in retrieved_docs:
        if doc['guideline'] in reasoning_chain or doc['content'][:50] in reasoning_chain:
            referenced_docs += 1
    
    return referenced_docs / len(retrieved_docs)
```

**Current Performance:** 100% (high utilization)

---

### Reasoning Coherence

**Definition:** Measures logical flow and consistency

**Not yet implemented** (future work: LLM-based coherence scoring)

---

## Calibration Metrics

### Brier Score

**Definition:** Measures accuracy of probabilistic predictions

**Formula:**
```
Brier Score = (1/N) Σ(p - y)²

where:
  p = predicted probability
  y = actual outcome (0 or 1)
```

**Interpretation:**
- 0 = perfect calibration
- 1 = worst calibration

**Implementation:**
```python
def calculate_brier_score(confidences, correctness):
    brier = 0
    for conf, correct in zip(confidences, correctness):
        y = 1 if correct else 0
        brier += (conf - y) ** 2
    return brier / len(confidences)
```

**Current Performance:**  
- Chain-of-Thought: 0.424
- Tree-of-Thought: 0.344
- Structured Medical: 0.295 (best)

---

### Expected Calibration Error (ECE)

**Definition:** Difference between confidence and accuracy across bins

**Formula:**
```
ECE = Σ(|Accuracy_bin - Confidence_bin| × n_bin / N)
```

**Bins:** [0-10%, 10-20%, ..., 90-100%]

**Implementation:**
```python
def calculate_ece(confidences, correctness, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        # Get predictions in this bin
        in_bin = (confidences >= bins[i]) & (confidences < bins[i+1])
        
        if in_bin.sum() > 0:
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = correctness[in_bin].mean()
            ece += abs(bin_confidence - bin_accuracy) * (in_bin.sum() / len(confidences))
    
    return ece
```

**Current Performance:**  
- Chain-of-Thought: 0.266 (best)
- Tree-of-Thought: 0.310
- Structured Medical: 0.283

---

### Confidence Distribution

**Tracks prediction confidence ranges**

**Example:**
```
Confidence Range | Count
-----------------|------
0-10%            | 0
10-20%           | 1
20-30%           | 0
30-40%           | 3
40-50%           | 1
50-60%           | 2
60-70%           | 8
70-80%           | 15
80-90%           | 12
90-100%          | 8
```

---

## Safety Metrics

### Hallucination Rate

**Definition:** Percentage of answers not grounded in provided context

**Detection Method:**
```python
def detect_hallucination(answer, reasoning_chain, context):
    # Extract key claims from reasoning
    claims = extract_claims(reasoning_chain)
    
    # Check if each claim is supported by context
    unsupported_claims = 0
    for claim in claims:
        if not is_claim_supported(claim, context):
            unsupported_claims += 1
    
    # Hallucination if >50% claims unsupported
    return unsupported_claims / len(claims) > 0.5
```

**Current Performance:** 0.0% (strict prompting successful)

---

### Dangerous Error Count

**Definition:** Answers that could lead to patient harm

**Categories:**
- Contraindicated treatment
- Missed critical diagnosis
- Delayed urgent intervention
- Incorrect drug dosing

**Current Performance:** 0 dangerous errors

---

### Safety Score

**Definition:** Composite safety metric

**Formula:**
```
Safety_Score = 1 - (Hallucination_Rate + Dangerous_Error_Rate) / 2
```

**Current Performance:** 1.0 (perfect)

---

## Confusion Matrix Analysis

### Answer-Level Confusion Matrix

**Visualization:** 4x4 matrix (A/B/C/D predicted vs true)

![Confusion Matrix](../reports/charts/confusion_matrix.png)

**Features:**
- Green-bordered diagonal for correct predictions
- Color intensity shows frequency
- Annotations show counts

**Implementation:**
```python
# src/evaluation/condition_confusion_analyzer.py

def visualize_confusion_matrix(results, output_path):
    # Create 4x4 matrix for answers
    answer_labels = ['A', 'B', 'C', 'D']
    answer_cm = np.zeros((4, 4), dtype=int)
    
    for result in results:
        true_idx = answer_labels.index(result['gold_answer'])
        pred_idx = answer_labels.index(result['predicted_answer'])
        answer_cm[true_idx, pred_idx] += 1
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(answer_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=answer_labels, yticklabels=answer_labels)
    
    # Highlight diagonal
    for i in range(4):
        plt.gca().add_patch(Rectangle((i, i), 1, 1, fill=False, 
                                      edgecolor='green', lw=3))
    
    plt.xlabel('Predicted Answer')
    plt.ylabel('True Answer')
    plt.title('Answer Confusion Matrix')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
```

---

### Condition-Level Confusion Analysis

**Medical Similarity Groups:**
1. Respiratory infections (Pneumonia, Bronchitis, URI)
2. Respiratory obstructive (Asthma, COPD)
3. Cardiac ischemic (ACS, Angina, MI)
4. Cardiac failure (Heart Failure, Cardiogenic shock)
5. Cardiac arrhythmia (Atrial Fibrillation, VT, SVT)
6. Stroke (Ischemic stroke, Hemorrhagic stroke, TIA)
7. Thrombotic (DVT, Pulmonary Embolism)
8. Infection systemic (Sepsis, Bacteremia)
9. Infection urinary (UTI, Pyelonephritis)
10. Gastrointestinal bleeding (Upper GI bleed, Lower GI bleed)
11. Gastrointestinal inflammatory (Pancreatitis, Gastritis)
12. Renal (AKI, CKD, Nephritis)
13. Metabolic diabetes (Type 1 DM, Type 2 DM, DKA)
14. Hepatic (Cirrhosis, Hepatitis, Liver failure)
15. Rheumatologic (RA, SLE, Gout)
16. Hypertensive (Hypertension, Hypertensive emergency)

**Error Classification:**
- **Similar-condition error:** Predicted and true conditions in same group
- **Unrelated-condition error:** Different groups

**Example:**
```
Predicted: Pneumonia (Respiratory infection)
True: Bronchitis (Respiratory infection)
→ Similar-condition error

Predicted: Heart Failure (Cardiac failure)
True: Pneumonia (Respiratory infection)
→ Unrelated-condition error
```

**Current Performance:**
- Similar-condition errors: 33.3%
- Unrelated-condition errors: 66.7%

---

## Error Analysis

### Error Categorization

**Retrieval Failures:**
- Relevant guideline not retrieved
- Low-quality context

**Current:** 0% (all cases have relevant context)

**Reasoning Failures:**
- Incorrect interpretation of symptoms
- Missing differential diagnosis
- Guideline misapplication

**Current:** 100% of errors

---

### Error Breakdown by Concept

**Medical Domain Distribution:**
```
Cardiovascular: 8 errors
Respiratory: 5 errors
Gastrointestinal: 3 errors
Infectious: 2 errors
Renal: 1 error
Metabolic: 1 error
```

---

### Top Confusion Pairs

**Most Common Confusions:**
1. Heart Failure ↔ Angina (3 cases)
2. DVT ↔ Pulmonary Embolism (2 cases)
3. Gastritis ↔ GI Bleed (2 cases)
4. Pneumonia ↔ COPD Exacerbation (1 case)
5. AKI ↔ CKD (1 case)

---

## Visualization

### Confusion Matrix Heatmap

Generated at: `reports/charts/confusion_matrix.png`

**Features:**
- 4x4 answer matrix (A/B/C/D)
- Color gradient (blue intensity)
- Annotated counts
- Green-bordered diagonal

---

### Performance Summary

Generated at: `reports/charts/performance_summary.png`

**Panels:**
1. Accuracy bar chart
2. Retrieval metrics (Precision/Recall)
3. Calibration (Brier/ECE)
4. Safety metrics

---

## Evaluation Reports

### JSON Output Format

**File:** `reports/new_dataset_eval_N_cases.json`

```json
{
    "overall_metrics": {
        "accuracy": 0.60,
        "total_questions": 50,
        "correct": 30,
        "incorrect": 20
    },
    "retrieval_metrics": {
        "precision_at_5": 0.04,
        "recall_at_5": 0.20,
        "map": 0.118,
        "mrr": 0.215,
        "avg_context_relevance": 0.428
    },
    "reasoning_metrics": {
        "chain_completeness": 1.0,
        "evidence_utilization": 1.0
    },
    "calibration_metrics": {
        "brier_score": 0.385,
        "ece": 0.46,
        "confidence_distribution": {...}
    },
    "safety_metrics": {
        "hallucination_rate": 0.0,
        "dangerous_errors": 0,
        "safety_score": 1.0
    },
    "error_analysis": {
        "retrieval_failures": 0,
        "reasoning_failures": 20,
        "high_confidence_errors": 5
    },
    "per_question_results": [...]
}
```

---

## Related Documentation

- [Evaluation Documentation](evaluation_documentation.md)
- [Part 2: RAG Implementation](part_2_rag_implementation.md)
- [Part 4: Experiments](part_4_experiments.md)

---

**Documentation Author:** Shreya Uprety
