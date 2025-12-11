# Reasoning Documentation

**Author:** Shreya Uprety  
**Updated:** 2025-12-11 (Corrected based on evaluation results)

## Table of Contents

1. [Overview](#overview)
2. [RAG Pipeline](#rag-pipeline)
3. [Advanced-Structured Reasoning (Tree-of-Thought Variant)](#advanced-structured-reasoning-tree-of-thought-variant)
4. [Performance Analysis](#performance-analysis)
5. [Error Analysis and Improvement Plan](#error-analysis-and-improvement-plan)
6. [Recommendations](#recommendations)

## Overview

The `src/reasoning/` module implements reasoning strategies for medical question answering. Based on evaluation results, the current best-performing method is the **Advanced-Structured approach** (Tree-of-Thought variant) achieving **52% accuracy**.

**Key Experimental Results:**
- **Advanced-Structured (Tree-of-Thought variant):** 52% accuracy, 41.4s avg time
- **Chain-of-Thought:** 34% accuracy, 5.0s avg time
- **Structured Medical:** 44% accuracy, 27.0s avg time

**Method Used in Evaluation:** Advanced-structured (Tree-of-Thought variant)

## RAG Pipeline

**File:** `src/reasoning/rag_pipeline.py`

### Purpose

Main RAG (Retrieval-Augmented Generation) pipeline that orchestrates retrieval, context processing, reasoning, and answer selection.

### Architecture (Based on Evaluation Results)

```
Input Question + Case Description
     ↓
Multi-Stage Retrieval (Semantic-First strategy - Best MAP: 0.213)
     ↓
Context Processing (top-k: 25, guideline coverage: 100%)
     ↓
Advanced-Structured Reasoning (Tree-of-Thought variant)
     ↓
Answer Selection with Confidence Scoring
     ↓
Output: Answer + Reasoning Chain + Confidence (0-1)
```

### Implementation

```python
class RAGPipeline:
    def __init__(
        self,
        retriever,
        reasoner,
        config: Dict
    ):
        """
        Initialize RAG pipeline as used in evaluation.
        
        Configuration from evaluation:
        - Retrieval: Semantic-First strategy (MAP: 0.213)
        - Reasoning: Advanced-structured (Tree-of-Thought variant)
        - Top-k: 25 documents
        - Confidence calibration: Not applied (Brier: 0.254, ECE: 0.179)
        """
        self.retriever = retriever
        self.reasoner = reasoner
        self.config = config
        
        # Evaluation results show these need improvement:
        self.retrieval_precision_at_5 = 0.112  # From evaluation
        self.retrieval_recall_at_5 = 0.56      # From evaluation
```

### Methods

#### answer_question(case: str, question: str, options: Dict) -> Dict

```python
def answer_question(
    self,
    case: str,
    question: str,
    options: Dict[str, str]
) -> Dict:
    """
    Answer clinical MCQ question as implemented in evaluation.
    
    Args:
        case: Clinical case description
        question: Question text
        options: Dictionary of {A/B/C/D: option_text}
        
    Returns (based on evaluation output):
        {
            'selected_answer': 'A|B|C|D',
            'confidence': float (0-1),
            'reasoning_steps': List[str],
            'answer_scores': Dict[str, float],
            'retrieved_contexts': List[Any]  # From evaluation
        }
        
    Performance from evaluation:
        - Accuracy: 52% (26/50 correct)
        - Avg reasoning time: ~41.4 seconds
        - Evidence utilization: 100%
        - Reasoning chain completeness: 100%
    """
    # Retrieval (Semantic-First - best from experiments)
    retrieved_docs = self.retriever.retrieve(
        query=question,
        top_k=25  # From evaluation configuration
    )
    
    # Reasoning (Advanced-structured/Tree-of-Thought variant)
    result = self.reasoner.reason_and_select_answer(
        question=question,
        case_description=case,
        options=options,
        retrieved_contexts=retrieved_docs,
        correct_answer=None  # For inference
    )
    
    return {
        'selected_answer': result.selected_answer,
        'confidence': result.confidence_score,
        'reasoning_steps': result.reasoning_steps,
        'answer_scores': result.answer_scores if hasattr(result, 'answer_scores') else {},
        'retrieved_contexts': retrieved_docs,
        'method_used': 'advanced_structured'  # From evaluation
    }
```

## Advanced-Structured Reasoning (Tree-of-Thought Variant)

**File:** `src/reasoning/medical_reasoning.py` (Implementation used in evaluation)

### Purpose

The reasoning method that achieved 52% accuracy in evaluation. Based on evaluation output, this is a Tree-of-Thought variant with structured clinical reasoning steps.

### Experimental Results

- **Accuracy:** 52.0% (26/50 correct)
- **Avg Reasoning Time:** 41,367ms (~41.4 seconds)
- **Reasoning Chain Completeness:** 100.0%
- **Evidence Utilization Rate:** 100.0%
- **Brier Score:** 0.2543
- **Expected Calibration Error (ECE):** 0.1786
- **Hallucination Rate:** 0.0%
- **Cannot Answer Misuse Rate:** 4.0% (2/50 cases)

### Implementation (Based on Evaluation Output)

```python
class AdvancedStructuredReasoner:
    """
    Tree-of-Thought variant with structured clinical reasoning.
    
    Based on evaluation reasoning excerpts:
    - Symptom extraction
    - Demographics analysis
    - Differential diagnosis generation
    - Evidence scoring
    - Guideline matching
    """
    
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.client = ollama.Client()
        
    def reason_and_select_answer(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        retrieved_contexts: List[Any],
        correct_answer: Optional[str] = None
    ):
        """
        Main reasoning method from evaluation.
        
        Based on evaluation reasoning excerpts:
        "Extracted X symptoms, demographics: {...}, acuity: ..."
        "Generated Y differential diagnoses"
        "Scored evidence for 4 options"
        "Matched guidelines for top Z"
        """
        # Step 1: Extract clinical features (from evaluation output)
        clinical_features = self._extract_clinical_features(case_description)
        
        # Step 2: Generate differential diagnoses
        differentials = self._generate_differentials(clinical_features, retrieved_contexts)
        
        # Step 3: Score evidence for each option
        evidence_scores = self._score_evidence(options, retrieved_contexts, clinical_features)
        
        # Step 4: Match guidelines (from evaluation output)
        guideline_matches = self._match_guidelines(options, retrieved_contexts)
        
        # Step 5: Select answer with confidence
        selected_answer, confidence = self._select_answer(
            evidence_scores, 
            guideline_matches,
            clinical_features
        )
        
        # Build reasoning steps (format from evaluation)
        reasoning_steps = self._build_reasoning_steps(
            clinical_features,
            differentials,
            evidence_scores,
            guideline_matches
        )
        
        # Calculate answer scores (if available)
        answer_scores = self._calculate_answer_scores(evidence_scores)
        
        return {
            'selected_answer': selected_answer,
            'confidence_score': confidence,
            'reasoning_steps': reasoning_steps,
            'answer_scores': answer_scores
        }
```

### Clinical Feature Extraction (From Evaluation)

Based on evaluation reasoning excerpts:

```python
def _extract_clinical_features(self, case_description: str) -> Dict:
    """
    Extract clinical features as shown in evaluation.
    
    Evaluation examples:
    - "Extracted 2 symptoms, demographics: {'age': 47, 'gender': 'male', 'age_group': 'adult'}, acuity: emergency"
    - "Extracted 0 symptoms, demographics: {'age': 70, 'gender': 'female', 'age_group': 'geriatric'}, acuity: emergency"
    - "Extracted 3 symptoms, demographics: {'age': 72, 'gender': 'male', 'age_group': 'geriatric'}, acuity: emergency"
    
    Issues identified:
    - 20 cases missed critical symptoms
    - 24 cases had medical terminology misunderstanding
    - Symptom extraction needs improvement
    """
    features = {
        'symptoms': [],
        'demographics': {},
        'acuity': 'routine',
        'vitals': {}
    }
    
    # Simple extraction (needs improvement based on evaluation)
    import re
    
    # Extract age
    age_match = re.search(r'(\d+)-year-old', case_description)
    if age_match:
        age = int(age_match.group(1))
        features['demographics']['age'] = age
        if age >= 65:
            features['demographics']['age_group'] = 'geriatric'
        elif age >= 18:
            features['demographics']['age_group'] = 'adult'
        else:
            features['demographics']['age_group'] = 'pediatric'
    
    # Extract gender
    if 'male' in case_description.lower():
        features['demographics']['gender'] = 'male'
    elif 'female' in case_description.lower():
        features['demographics']['gender'] = 'female'
    
    # Determine acuity
    emergency_keywords = ['emergency', 'acute', 'sudden', 'severe', 'urgent']
    if any(keyword in case_description.lower() for keyword in emergency_keywords):
        features['acuity'] = 'emergency'
    
    return features
```

### Reasoning Steps Format (From Evaluation)

```python
def _build_reasoning_steps(
    self,
    clinical_features: Dict,
    differentials: List,
    evidence_scores: Dict,
    guideline_matches: Dict
) -> List[str]:
    """
    Build reasoning steps in format seen in evaluation.
    
    Evaluation format:
    "Extracted X symptoms, demographics: {...}, acuity: ..."
    "Generated Y differential diagnoses"
    "Scored evidence for 4 options"
    "Matched guidelines for top Z"
    """
    steps = []
    
    # Step 1: Clinical feature extraction
    symptom_count = len(clinical_features.get('symptoms', []))
    steps.append(
        f"Extracted {symptom_count} symptoms, "
        f"demographics: {clinical_features['demographics']}, "
        f"acuity: {clinical_features['acuity']}"
    )
    
    # Step 2: Differential diagnosis
    diff_count = len(differentials)
    steps.append(f"Generated {diff_count} differential diagnoses")
    
    # Step 3: Evidence scoring
    steps.append("Scored evidence for 4 options")
    
    # Step 4: Guideline matching
    match_count = len(guideline_matches)
    steps.append(f"Matched guidelines for top {match_count}")
    
    return steps
```

### Confidence Calculation (Current Implementation Issues)

Based on evaluation confidence distribution:

```python
def _calculate_confidence(
    self,
    evidence_scores: Dict,
    clinical_features: Dict
) -> float:
    """
    Calculate confidence score (0-1).
    
    Current issues from evaluation:
    - Brier Score: 0.254 (needs improvement)
    - ECE: 0.179 (needs improvement)
    - 2 cases with >80% confidence but wrong answers
    - Confidence distribution shows calibration issues
    
    Current confidence distribution (50 cases):
    90-100%:  7 cases (14%) - Accuracy: 85.7%
    80-90%:   1 case  (2%)  - Accuracy: 0.0% (overconfident error)
    70-80%:   2 cases (4%)  - Accuracy: 50.0%
    40-50%:  11 cases (22%) - Accuracy: 63.6%
    30-40%:  13 cases (26%) - Accuracy: 46.2%
    20-30%:   4 cases (8%)  - Accuracy: 50.0%
    10-20%:   7 cases (14%) - Accuracy: 42.9%
    0-10%:    5 cases (10%) - Accuracy: 20.0%
    """
    # Simple confidence calculation (needs calibration)
    max_score = max(evidence_scores.values()) if evidence_scores else 0
    min_score = min(evidence_scores.values()) if evidence_scores else 0
    
    if max_score == min_score:
        return 0.5  # Default confidence
    
    # Normalize to 0-1 range
    confidence = (max_score - min_score) / (max_score + 0.001)
    
    # Apply adjustment based on symptom count
    symptom_count = len(clinical_features.get('symptoms', []))
    if symptom_count < 2:
        confidence *= 0.7  # Reduce confidence with few symptoms
    
    return min(max(confidence, 0.0), 1.0)
```

## Performance Analysis

### Current Performance Summary

**Overall Accuracy:** 52.0% (26/50 correct)

**By Question Type:**
- Diagnosis: 52.2% (24/46 correct)
- Treatment: 100.0% (2/2 correct) - but only 2 cases
- Other: 0.0% (0/2 correct)

**By Medical Category (Accuracy):**
- Critical Care: 100.0% (1/1)
- Gastroenterology: 71.4% (5/7)
- Endocrine: 66.7% (4/6)
- Nephrology: 66.7% (2/3)
- Respiratory: 62.5% (5/8)
- Cardiovascular: 54.5% (6/11)
- Hematology: 33.3% (1/3)
- Rheumatology: 33.3% (1/3)
- Psychiatry: 33.3% (1/3)
- **Infectious Disease: 0.0% (0/3) - Critical gap**
- **Neurology: 0.0% (0/2) - Critical gap**

**By Complexity:**
- Simple: 58.3% (7/12)
- Moderate: 52.0% (13/25)
- Complex: 46.2% (6/13)

**By Relevance Level:**
- High relevance: 44.8% (13/29)
- **Medium relevance: 80.0% (8/10) - Best performance**
- Low relevance: 45.5% (5/11)

### Time Performance

**Reasoning Time (Advanced-Structured):**
- Average: 41,367ms (~41.4 seconds)
- Total for 50 cases: 2,068 seconds (~34.5 minutes)

**Breakdown (from earlier experiments):**
- Branch generation: ~5,000ms
- Branch evaluation: ~30,000ms
- Branch pruning: ~4,000ms
- Final selection: ~2,367ms

**Comparison with Other Methods:**
- Chain-of-Thought: 4,955ms avg (8.4x faster, 34% accuracy)
- Structured Medical: 26,991ms avg (1.5x faster, 44% accuracy)

### Calibration Performance

**Current Issues:**
- Brier Score: 0.2543 (target: <0.20)
- ECE: 0.1786 (target: <0.15)
- Overconfident Errors: 2 cases with >80% confidence but wrong

**Confidence-Accuracy Mismatch:**
- 90-100% confidence: 85.7% accuracy (good)
- **80-90% confidence: 0.0% accuracy (critical issue)**
- 40-50% confidence: 63.6% accuracy (overconfident)
- 0-10% confidence: 20.0% accuracy (underconfident)

### Safety Performance

**Critical Gaps Identified:**
- Contraindication Check Accuracy: 0.0%
- Urgency Recognition Accuracy: 0.0%
- Dangerous Error Count: 2 cases
- Safety Score: 0.96 (good overall but critical gaps)

**Strengths:**
- Hallucination Rate: 0.0%
- Evidence Utilization: 100.0%
- Reasoning Chain Completeness: 100.0%

## Error Analysis and Improvement Plan

### Error Categories (From Evaluation)

#### 1. Reasoning Errors (16 cases - 64% of errors)

**Description:** Retrieved relevant info but made incorrect reasoning

**Root Causes:**
- Insufficient chain-of-thought reasoning steps
- Failure to properly weight evidence from multiple sources
- Over-reliance on single retrieved document
- Missing critical symptom analysis

**Examples:** Q_082 (95% confidence wrong), Q_036 (34.7% confidence wrong)

#### 2. Knowledge Errors (8 cases - 32% of errors)

**Description:** Incorrect medical knowledge or interpretation

**Root Causes:**
- Incorrect interpretation of medical guidelines
- Missing context about patient-specific factors
- Failure to consider contraindications
- Incorrect application of treatment protocols

**Examples:** Q_032 (7.1% confidence wrong), Q_070 (71.3% confidence wrong)

### Major Pitfalls in Current Reasoning

#### Pitfall 1: Missing Critical Symptoms (20 cases)

- **Issue:** Reasoning fails to consider important symptoms
- **Examples from evaluation:** Q_004, Q_032, Q_087 cases
- **Solution:** Enhance symptom extraction with medical NER

#### Pitfall 2: Medical Terminology Misunderstanding (24 cases)

- **Issue:** Fails to interpret medical abbreviations/terms
- **Examples:** "Extracted 0 symptoms" when symptoms present
- **Solution:** Add medical terminology expansion

#### Pitfall 3: Overconfident Wrong Answers (2 cases)

- **Issue:** High confidence (>80%) but incorrect answers
- **Examples:** Q_082 (95% confidence wrong), Q_085 (87.8% confidence wrong)
- **Solution:** Implement confidence calibration

### Improvement Plan for Reasoning

#### Immediate Improvements (1-2 weeks):

**1. Enhance Symptom Extraction**

```python
# Replace current simple extraction with:
def enhanced_symptom_extraction(case_description: str) -> List[str]:
    # Use scispacy or BioBERT for medical NER
    # Addresses 20 cases of missing critical symptoms
    pass
```

**2. Add Medical Terminology Expansion**

```python
def expand_medical_terms(text: str) -> str:
    # Expand abbreviations: "ACS" → "Acute Coronary Syndrome"
    # Addresses 24 cases of terminology misunderstanding
    pass
```

**3. Implement Basic Confidence Calibration**

```python
def calibrate_confidence(raw_confidence: float, features: Dict) -> float:
    # Apply temperature scaling based on historical performance
    # Addresses 2 overconfident wrong answers
    pass
```

#### Medium-term Improvements (3-4 weeks):

**4. Add Safety Checks**

```python
def check_contraindications(answer: str, case: str, context: str) -> bool:
    # Check for contraindications in treatment options
    # Addresses 0% contraindication check accuracy
    pass
```

**5. Improve Differential Diagnosis Generation**

```python
def generate_comprehensive_differentials(case: str) -> List[Dict]:
    # Generate more comprehensive differentials
    # Addresses insufficient reasoning steps
    pass
```

**6. Implement Evidence Weighting**

```python
def weight_evidence_by_guideline_strength(evidence: List) -> Dict:
    # Weight evidence based on guideline strength
    # Addresses over-reliance on single document
    pass
```

## Recommendations

### Immediate Actions for Production

#### 1. Use Current Best Configuration:

- **Reasoning Method:** Advanced-structured (Tree-of-Thought variant)
- **Retrieval Strategy:** Semantic-First (MAP: 0.213, 9.65ms)
- **Expected Performance:** 52% accuracy, ~41s per query

#### 2. Add Confidence Thresholds for Safety:

```python
# Reject low-confidence answers for safety
if confidence < 0.3:
    return "Cannot answer from the provided context."

# Flag high-confidence answers for review
if confidence > 0.8:
    log_for_expert_review(answer, reasoning, confidence)
```

#### 3. Implement Hybrid Approach for Speed:

```python
# Use faster method for simple cases
def select_reasoning_method(complexity: float):
    if complexity < 0.5:
        return "chain_of_thought"  # 5s avg
    else:
        return "advanced_structured"  # 41s avg
```

### Performance Targets

**With Immediate Improvements:**
- Accuracy: 52% → 55-57%
- Precision@5: 11.2% → 14-16%
- Brier Score: 0.254 → 0.23
- Overconfident Errors: 2 → 0

**With Medium-term Improvements:**
- Accuracy: 52% → 60-65%
- Safety Metrics: 0% → >50% for contraindication/urgency
- Reasoning Time: 41s → 25-30s with optimizations

### Configuration for Next Evaluation

```python
recommended_config = {
    "retrieval": {
        "strategy": "semantic_first",
        "top_k": 25,
        "similarity_threshold": 0.45
    },
    "reasoning": {
        "method": "advanced_structured",
        "enhancements": {
            "symptom_extraction": "medical_ner",
            "confidence_calibration": "temperature_scaling",
            "safety_checks": "contraindication_verification"
        }
    },
    "safety": {
        "reject_confidence_below": 0.1,
        "review_confidence_above": 0.8,
        "require_contraindication_check": True
    }
}
```

### Monitoring and Evaluation

**Key Metrics to Track:**
- Accuracy by Medical Category (watch Infectious Disease, Neurology)
- Confidence Calibration (Brier Score, ECE)
- Safety Metrics (contraindication checks, dangerous errors)
- Reasoning Time (target: <30s average)

**Automated Checks:**

```python
def validate_reasoning_output(result: Dict) -> bool:
    """Validate reasoning output meets quality standards."""
    checks = [
        result['confidence'] >= 0.0 and result['confidence'] <= 1.0,
        result['selected_answer'] in ['A', 'B', 'C', 'D', 'Cannot answer...'],
        len(result['reasoning_steps']) >= 3,  # Minimum reasoning steps
        'contraindication_check' in result.get('safety_checks', {}),
    ]
    return all(checks)
```

## Related Documentation

- **Part 4:** Experiments and Analysis - Updated with actual results
- **Evaluation Framework** - Performance metrics
- **Improvements Documentation** - Enhancement plans

---

**Documentation Author:** Shreya Uprety  
**Evaluation Reference:** 2025-12-11 Medical QA Evaluation Results  
**Current Status:** Advanced-structured reasoning achieves 52% accuracy, needs improvement in symptom extraction (20 cases), terminology (24 cases), and confidence calibration (2 overconfident errors)