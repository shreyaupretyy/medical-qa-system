# Part 4: Experiments and Analysis

**Author:** Shreya Uprety  
**Updated:** 2025-12-11 (Corrected based on actual evaluation results)

## Table of Contents

1. [Overview](#overview)
2. [Retrieval Strategy Comparison](#retrieval-strategy-comparison)
3. [Reasoning Method Comparison](#reasoning-method-comparison)
4. [Error Analysis](#error-analysis)
5. [Performance Insights](#performance-insights)
6. [Recommendations](#recommendations)

## Overview

This document presents comprehensive experimental results from evaluating the Medical Question-Answering System across different retrieval strategies and reasoning methods. All experiments were conducted on the clinical case dataset.

**Evaluation Date:** December 9-11, 2025  
**Dataset:** Clinical MCQs (50 cases for full evaluation, 100 cases for retrieval-only)  
**Hardware:** GPU-accelerated (NVIDIA CUDA)  
**LLM:** Ollama Llama 3.1 8B  
**Reasoning Method:** Advanced-structured (Tree-of-Thought variant)

## Retrieval Strategy Comparison

**Experiment Goal:** Identify the most effective retrieval strategy for medical question answering.

**Date:** December 9, 2025  
**Cases Evaluated:** 100  
**Report:** `reports/retrieval_strategy_comparison.json`

### Strategies Tested

#### 1. Single-Stage FAISS

**Description:** Pure semantic search using dense embeddings (MiniLM-L6-v2)

**Results:**
- MAP: 0.2110
- MRR: 0.42199
- Precision@1: 0%
- Precision@3: 25.3%
- Precision@5: 17.6%
- Recall@3: 38%
- Recall@5: 44%
- Avg Query Time: 8.58ms

**Analysis:**
- **Strengths:** Fast retrieval, captures semantic meaning
- **Weaknesses:** General-purpose embeddings miss medical terminology
- **Best For:** Paraphrased queries, conceptual questions

#### 2. Single-Stage BM25

**Description:** Pure keyword-based search using BM25 algorithm

**Results:**
- MAP: 0.2072
- MRR: 0.4145
- Precision@1: 0%
- Precision@3: 24.0%
- Precision@5: 17.4%
- Recall@3: 36%
- Recall@5: 43.5%
- Avg Query Time: 1.40ms (fastest)

**Analysis:**
- **Strengths:** Extremely fast, excellent for exact medical term matching
- **Weaknesses:** Misses semantic relationships, vocabulary-dependent
- **Best For:** Questions with specific medical terminology

#### 3. Hybrid Linear

**Description:** Linear combination of FAISS (65%) and BM25 (35%)

**Results:**
- MAP: 0.2106
- MRR: 0.4213
- Precision@1: 0%
- Precision@3: 24.7%
- Precision@5: 17.8%
- Recall@3: 37%
- Recall@5: 44.5%
- Avg Query Time: 8.33ms

**Analysis:**
- **Strengths:** Balances lexical and semantic matching
- **Weaknesses:** Weighted combination doesn't always improve over single methods
- **Best For:** Diverse query types

**Weight Tuning:**
```python
# Tested configurations
weights = [
    (0.5, 0.5),  # Equal
    (0.65, 0.35),  # FAISS-heavy (used)
    (0.35, 0.65),  # BM25-heavy
]
# Best: 0.65 FAISS, 0.35 BM25
```

#### 4. Multi-Stage (3-stage)

**Description:** Stage 1: FAISS (k=150) → Stage 2: BM25 filter (k=100) → Stage 3: Cross-encoder rerank (k=25)

**Results:**
- MAP: 0.2042
- MRR: 0.4083
- Precision@1: 0%
- Precision@3: 24.3%
- Precision@5: 17.0%
- Recall@3: 36.5%
- Recall@5: 42.5%
- Avg Query Time: 2,878ms (slowest)

**Analysis:**
- **Strengths:** Comprehensive, maximizes recall in early stages
- **Weaknesses:** Cross-encoder reranking adds significant latency (2.8s), slightly lower precision
- **Root Cause:** Cross-encoder model (ms-marco-MiniLM) is general-purpose, not medical domain
- **Best For:** High-stakes applications where accuracy > speed

**Performance Breakdown:**
```
Stage 1 (FAISS): ~8ms, k=150
Stage 2 (BM25 filter): ~2ms, k=100
Stage 3 (Cross-encoder): ~2,868ms, k=25
Total: ~2,878ms
Bottleneck: Cross-encoder processes 100 documents × query
```

#### 5. Concept-First

**Description:** BM25 keyword filter followed by FAISS semantic refinement

**Results:**
- MAP: 0.2121 (highest)
- MRR: 0.4243 (highest)
- Precision@1: 0%
- Precision@3: 25.0%
- Precision@5: 18.0% (highest)
- Recall@3: 37.5%
- Recall@5: 45% (highest)
- Avg Query Time: 11.62ms

**Analysis:**
- **Strengths:** Best recall performance, good balance
- **Mechanism:** BM25 filters for medical terms, FAISS refines with semantics
- **Best For:** Maximizing recall

#### 6. Semantic-First

**Description:** FAISS semantic search followed by BM25 keyword refinement

**Results:**
- MAP: 0.2126 (highest)
- MRR: 0.4252 (highest)
- Precision@1: 0%
- Precision@3: 25.3%
- Precision@5: 17.8%
- Recall@3: 38%
- Recall@5: 44.5%
- Avg Query Time: 9.65ms

**Analysis:**
- **Strengths:** Best MAP score, slightly better semantic matching
- **Mechanism:** FAISS casts wide net, BM25 refines with keywords
- **Best For:** Overall performance balance

### Retrieval Strategy Comparison Summary

| Strategy | MAP | MRR | P@5 | R@5 | Time (ms) | Rank |
|----------|-----|-----|-----|-----|-----------|------|
| Semantic-First | 0.2126 | 0.4252 | 17.8% | 44.5% | 9.65 | 1 |
| Concept-First | 0.2121 | 0.4243 | 18.0% | 45.0% | 11.62 | 2 |
| Single FAISS | 0.2110 | 0.4220 | 17.6% | 44.0% | 8.58 | 3 |
| Hybrid Linear | 0.2106 | 0.4213 | 17.8% | 44.5% | 8.33 | 4 |
| Single BM25 | 0.2072 | 0.4145 | 17.4% | 43.5% | 1.40 | 5 |
| Multi-Stage | 0.2042 | 0.4083 | 17.0% | 42.5% | 2,878 | 6 |

**Key Findings:**
- **Semantic-First** has best MAP (0.2126) and fastest among top performers (9.65ms)
- **Concept-First** has best recall (45.0% @5) and precision (18.0% @5)
- **Multi-Stage** significantly underperforms due to cross-encoder bottleneck
- **BM25** is fastest (1.40ms) but has lower accuracy

**Recommendation:** Use **Semantic-First** for best overall performance (MAP 0.213, 9.65ms)

## Reasoning Method Comparison

**Experiment Goal:** Compare Chain-of-Thought, Tree-of-Thought, and Structured Medical Reasoning.

**Date:** December 10, 2025  
**Cases Evaluated:** 50  
**Report:** `reports/reasoning_method_comparison.json`

### Method 1: Chain-of-Thought (CoT)

**Description:** Standard LLM reasoning with step-by-step prompting

**Results:**
- **Accuracy:** 34% (17/50 correct)
- **Avg Time:** 4,955ms (~5 seconds)
- **Total Time:** 247.7 seconds
- **Brier Score:** 0.424
- **ECE:** 0.266 (best calibration)
- **Avg Reasoning Length:** 98 words
- **Avg Evidence Usage:** 10.64 sources
- **Reasoning Coherence:** 32.7%

**Answer Distribution:**
- B: 23 (46%)
- A: 17 (34%)
- C: 7 (14%)
- D: 2 (4%)
- Cannot answer: 1 (2%)

**Analysis:**
- **Strengths:** Fast, straightforward, best calibrated
- **Weaknesses:** Lowest accuracy, significant bias toward option B (46%)
- **Best For:** Time-sensitive applications

### Method 2: Tree-of-Thought (ToT)

**Description:** Structured multi-branch reasoning with explicit thought paths

**Results:**
- **Accuracy:** 52% (26/50 correct) - HIGHEST
- **Avg Time:** 41,367ms (~41 seconds)
- **Total Time:** 2,068 seconds
- **Brier Score:** 0.344
- **ECE:** 0.310
- **Avg Reasoning Length:** 525 words
- **Avg Evidence Usage:** 3.9 sources
- **Reasoning Coherence:** 43.3% (best)

**Answer Distribution:**
- A: 18 (36%)
- C: 14 (28%)
- B: 14 (28%)
- D: 2 (4%)
- Cannot answer: 2 (4%)

**Analysis:**
- **Strengths:** Highest accuracy, best reasoning coherence, most balanced distribution
- **Weaknesses:** 8.4x slower than CoT, verbose output
- **Best For:** Complex cases requiring differential diagnosis

### Method 3: Structured Medical

**Description:** 5-step clinical reasoning: feature extraction → differential → evidence → treatment → selection

**Results:**
- **Accuracy:** 44% (22/50 correct)
- **Avg Time:** 26,991ms (~27 seconds)
- **Total Time:** 1,349.5 seconds
- **Brier Score:** 0.295 (BEST)
- **ECE:** 0.283
- **Avg Reasoning Length:** 41 words
- **Avg Evidence Usage:** 6.72 sources
- **Reasoning Coherence:** 32.0%

**Answer Distribution:**
- B: 21 (42%)
- A: 13 (26%)
- C: 9 (18%)
- D: 5 (10%)
- Cannot answer: 2 (4%)

**Analysis:**
- **Strengths:** Best calibration, concise reasoning, systematic approach
- **Weaknesses:** Middle accuracy, bias toward option B (42%)
- **Best For:** Confidence-critical applications

### Reasoning Method Comparison Summary

| Method | Accuracy | Avg Time (ms) | Brier Score | ECE | Coherence |
|--------|----------|---------------|-------------|-----|-----------|
| Tree-of-Thought | 52% | 41,367 | 0.344 | 0.310 | 43.3% |
| Structured Medical | 44% | 26,991 | 0.295 | 0.283 | 32.0% |
| Chain-of-Thought | 34% | 4,955 | 0.424 | 0.266 | 32.7% |

**Performance Insights:**
- ToT is most accurate (52%) but slowest (41.4s)
- Structured Medical has best calibration (Brier: 0.295)
- CoT is fastest (5.0s) but least accurate (34%)
- Accuracy-Calibration Tradeoff: More accurate methods have worse calibration

**Recommendations:**
- **For Accuracy:** Use Tree-of-Thought (52%)
- **For Speed:** Use Chain-of-Thought (5.0s avg)
- **For Calibration:** Use Structured Medical (Brier: 0.295)
- **For Production:** Hybrid approach with complexity-based escalation

## Error Analysis

**Source:** `reports/evaluation_results.json` (50 cases with Advanced-structured/Tree-of-Thought)  
**Total Errors:** 24 (48% error rate)

### Error Categorization

**Primary Error Types (Based on Full Evaluation):**

#### 1. Reasoning Errors (16 cases - 64% of errors)

**Description:** System retrieved relevant info but made incorrect reasoning

**Root Causes:**
- Insufficient chain-of-thought reasoning steps
- Failure to properly weight evidence from multiple sources
- Over-reliance on single retrieved document
- Missing critical symptom analysis

**Examples:**
- Q_082: 95% confidence but incorrect (A instead of B)
- Q_036: 34.7% confidence, wrong treatment selection

#### 2. Knowledge Errors (8 cases - 32% of errors)

**Description:** System has incorrect medical knowledge or interpretation

**Root Causes:**
- Incorrect interpretation of medical guidelines
- Missing context about patient-specific factors
- Failure to consider contraindications
- Incorrect application of treatment protocols

**Examples:**
- Q_032: 7.1% confidence, wrong treatment (B instead of D)
- Q_070: 71.3% confidence, incorrect diagnosis

### Performance Segmentation

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
- **Medium relevance:** 80.0% accuracy (8/10) - Best performance
- **Low relevance:** 45.5% accuracy (5/11)

### Major Pitfalls Identified

#### Pitfall 1: Overconfident Wrong Answers

- **Count:** 2 cases (>80% confidence but incorrect)
- **Severity:** High
- **Examples:** Q_082 (95% confidence wrong), Q_085 (87.8% confidence wrong)
- **Solution:** Implement confidence calibration and uncertainty estimation

#### Pitfall 2: Missing Critical Symptoms

- **Count:** 20 cases (40% of total)
- **Severity:** Medium
- **Description:** Reasoning fails to consider important symptoms
- **Examples:** Q_004, Q_032, Q_087 - all missing symptom extraction
- **Solution:** Enhance symptom extraction and ensure all symptoms are considered

#### Pitfall 3: Medical Terminology Misunderstanding

- **Count:** 24 cases (48% of total)
- **Severity:** Medium
- **Description:** Fails to properly interpret medical abbreviations/terms
- **Examples:** Reasoning excerpts show poor symptom extraction (e.g., "Extracted 0 symptoms")
- **Solution:** Add medical terminology expansion and abbreviation resolution

## Performance Insights

### Insight 1: Retrieval Precision is Critical Bottleneck

**Current State:**
- Precision@5: 11.2% (too low)
- Recall@5: 56.0% (reasonable)
- MAP: 0.268 (moderate)

**Impact on Accuracy:**
- 52% accuracy ceiling suggests retrieval issues limit reasoning
- Even with perfect reasoning, poor retrieval would limit performance

**Root Cause:** General-purpose embeddings (MiniLM) not medical-domain

**Projected Improvement with Medical Embeddings:**
```
Current:                With PubMedBERT:
  Precision@5: 11.2%      Precision@5: 25-30%
  Recall@5: 56.0%         Recall@5: 70-75%
  Accuracy: 52%           Accuracy: 65-70%
```

### Insight 2: Reasoning Method Selection Depends on Use Case

**Tradeoffs Identified:**
```
Method          Accuracy  Speed     Calibration  Use Case
Tree-of-Thought   52%     41.4s     Moderate     Complex cases, high stakes
Structured Med    44%     27.0s     Best         Confidence-critical apps
Chain-of-Thought  34%      5.0s     Good         Time-sensitive, simple cases
```

**Hybrid Strategy Recommended:**
```python
def hybrid_reasoning_strategy(question_complexity, time_constraint):
    if time_constraint < 10:  # seconds
        return "chain_of_thought"
    elif question_complexity > 0.7:  # complex case
        return "tree_of_thought"
    else:
        return "structured_medical"
```

### Insight 3: Confidence Calibration Needs Improvement

**Current Calibration Metrics:**
- Brier Score: 0.254 (target: <0.20)
- ECE: 0.179 (target: <0.15)
- Overconfident Errors: 2 cases with >80% confidence but wrong

**Calibration Analysis by Confidence Range:**
```
Confidence    Cases  Accuracy  Calibration Error
90-100%         7     85.7%     -9.3% (slightly underconfident)
80-90%          1      0.0%    -80.0% (severely overconfident)
70-80%          2     50.0%    -25.0% (overconfident)
40-50%         11     63.6%    -21.4% (overconfident)
30-40%         13     46.2%    -18.8% (overconfident)
0-10%           5     20.0%    +10.0% (underconfident)
```

**Pattern:** System is overconfident in mid-range (30-80%) and has one severely overconfident case

### Insight 4: Dataset Imbalances Affect Performance

**Dataset Issues Identified:**
- **Specialty Imbalance:** Cardiology dominates (70% confusion), Infectious Disease/Neurology at 0%
- **Question Type Bias:** 92% diagnosis (46/50), only 4% treatment (2/50)
- **Safety Coverage Gap:** 0% accuracy in contraindication and urgency recognition
- **Complexity Distribution:** Complex cases perform worst (46% vs 58% simple)

**Impact:** Current 52% accuracy ceiling partly due to dataset limitations

## Recommendations

### Immediate Priority (High Impact)

#### 1. Address Retrieval Precision (Target: 16% @5)

- **Action:** Switch to medical embeddings (PubMedBERT/BioBERT)
- **Location:** Replace MiniLM-L6-v2 in embedding model
- **Expected Impact:** +15-20% precision, +5-10% accuracy
- **Timeline:** 1 week

#### 2. Implement Confidence Calibration

- **Action:** Add temperature scaling or Platt scaling
- **Target:** Reduce ECE from 0.179 to <0.15, eliminate >80% wrong answers
- **Expected Impact:** Better uncertainty estimation, safer deployment
- **Timeline:** 2 weeks

#### 3. Fix Critical Safety Gaps

- **Action:** Add contraindication and urgency recognition
- **Target:** >50% accuracy on safety metrics (currently 0%)
- **Expected Impact:** Reduce dangerous error count from 2 to 0
- **Timeline:** 2 weeks

### Medium Priority (Moderate Impact)

#### 4. Improve Symptom Extraction

- **Action:** Implement medical NER (scispacy or BioBERT)
- **Target:** Reduce "missing critical symptoms" from 20 to <10 cases
- **Expected Impact:** +5% accuracy
- **Timeline:** 3 weeks

#### 5. Rebalance Dataset

- **Action:** Generate more treatment questions, reduce cardiology dominance
- **Target:** Treatment questions: 4% → 30%, Cardiology: 70% → 30%
- **Expected Impact:** More representative performance measurement
- **Timeline:** 4 weeks

#### 6. Optimize Hybrid Retrieval

- **Action:** Fine-tune weights for Semantic-First/Concept-First
- **Target:** Precision@5: 11.2% → 14%
- **Expected Impact:** Better retrieval for reasoning
- **Timeline:** 2 weeks

### Long-Term Priority (Foundation)

#### 7. Fine-tune Medical Cross-Encoder

- **Action:** Fine-tune on medical QA pairs
- **Target:** Replace general-purpose cross-encoder hurting performance
- **Expected Impact:** +5-10% precision
- **Timeline:** 6 weeks

#### 8. Implement Active Learning

- **Action:** Identify difficult cases for targeted improvement
- **Target:** Continuous accuracy improvement
- **Expected Impact:** +1-2% accuracy per iteration
- **Timeline:** 8 weeks

#### 9. Build Medical Knowledge Integration

- **Action:** Integrate UMLS/SNOMED for concept expansion
- **Target:** Improve medical terminology understanding
- **Expected Impact:** Reduce terminology misunderstandings (24→12 cases)
- **Timeline:** 10 weeks

### Expected Performance Improvements

**With Immediate+Medium Priorities (8 weeks):**
```
Metric           Current  Target    Improvement
Accuracy          52%      58-60%    +6-8%
Precision@5       11.2%    14-16%   +2.8-4.8%
Brier Score       0.254    0.23     -0.024
ECE               0.179    0.15     -0.029
Safety Accuracy   0%       >50%     +50%
```

**With All Priorities (16 weeks):**
```
Metric           Current  Target    Improvement
Accuracy          52%      65%       +13%
Precision@5       11.2%    20%      +8.8%
Dangerous Errors  2        0        -2
```

## Experiment Reproduction

### Retrieval Strategy Comparison

```bash
python scripts/compare_retrieval_strategies.py \
    --num-cases 100 \
    --output reports/retrieval_strategy_comparison.json
```

### Reasoning Method Comparison

```bash
python scripts/compare_reasoning_methods.py \
    --num-cases 50 \
    --output reports/reasoning_method_comparison.json
```

### Full Evaluation with Current Best Configuration

```bash
python src/evaluation/evaluate_pipeline.py \
    --dataset data/processed/questions/questions_1.json \
    --num-cases 50 \
    --retrieval-strategy semantic_first \
    --reasoning-method tree_of_thought \
    --top-k 25 \
    --output reports/evaluation_results.json
```

## Related Documentation

- **Part 2:** RAG Implementation
- **Part 3:** Evaluation Framework - Updated with actual results
- **Retrieval Documentation**
- **Reasoning Documentation**

---

**Documentation Author:** Shreya Uprety  
**Evaluation Reference:** 2025-12-11 Comprehensive Medical QA Evaluation Results  
**Key Finding:** System achieves 52% accuracy with Tree-of-Thought reasoning, limited by retrieval precision (11.2% @5) and dataset imbalances