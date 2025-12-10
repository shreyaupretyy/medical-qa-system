# Reasoning Documentation

**Author:** Shreya Uprety  
**Last Updated:** December 11, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [RAG Pipeline](#rag-pipeline)
3. [Chain-of-Thought Reasoning](#chain-of-thought-reasoning)
4. [Tree-of-Thought Reasoning](#tree-of-thought-reasoning)
5. [Structured Medical Reasoning](#structured-medical-reasoning)
6. [Hybrid Reasoning](#hybrid-reasoning)
7. [Query Understanding](#query-understanding)
8. [Performance Analysis](#performance-analysis)

---

## Overview

The `src/reasoning/` module implements multiple reasoning strategies for medical question answering. The module supports Chain-of-Thought (CoT), Tree-of-Thought (ToT), Structured Medical Reasoning, and hybrid approaches.

**Key Finding:** ToT achieves highest accuracy (52%) but ToT is 8.4x slower than CoT (34%). Hybrid approach balances both.

---

## RAG Pipeline

**File:** `src/reasoning/rag_pipeline.py`

### Purpose

Main RAG (Retrieval-Augmented Generation) pipeline that orchestrates retrieval, context processing, reasoning, and answer selection.

### Architecture

```
Input Question
     ↓
Query Understanding (extract features, expand concepts)
     ↓
Multi-Stage Retrieval (BM25, FAISS, Concept-First)
     ↓
Context Processing (pruning, reranking, formatting)
     ↓
Reasoning Engine (CoT / ToT / Structured / Hybrid)
     ↓
Safety Verification (hallucination detection, safety checks)
     ↓
Confidence Calibration
     ↓
Answer + Reasoning Chain + Confidence
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
        Initialize RAG pipeline.
        
        Args:
            retriever: Retrieval strategy instance
            reasoner: Reasoning method instance
            config: Pipeline configuration
            
        Components:
            - Query understanding
            - Multi-stage retrieval
            - Context pruning
            - Reasoning (CoT/ToT/Structured)
            - Hallucination detection
            - Safety verification
            - Confidence calibration
        """
        self.retriever = retriever
        self.reasoner = reasoner
        self.config = config
        
        # Initialize improvement modules
        self.clinical_feature_extractor = ClinicalFeatureExtractor()
        self.concept_expander = MedicalConceptExpander()
        self.context_pruner = ContextPruner()
        self.hallucination_detector = HallucinationDetector()
        self.safety_verifier = SafetyVerifier()
        self.confidence_calibrator = ConfidenceCalibrator()
```

### Methods

#### `answer_question(case: str, question: str, options: Dict) -> Dict`

```python
def answer_question(
    self,
    case: str,
    question: str,
    options: Dict[str, str]
) -> Dict:
    """
    Answer clinical MCQ question.
    
    Args:
        case: Clinical case description
        question: Question text
        options: Dictionary of {A/B/C/D: option_text}
        
    Returns:
        {
            'answer': 'A|B|C|D',
            'reasoning': 'Step-by-step reasoning',
            'confidence': float (0-1),
            'evidence': List of supporting documents,
            'safety_verified': bool,
            'hallucination_detected': bool
        }
        
    Process:
        1. Query understanding
        2. Retrieval
        3. Context processing
        4. Reasoning
        5. Safety checks
        6. Confidence calibration
    """
    # Step 1: Query understanding
    query_features = self._understand_query(case, question)
    
    # Step 2: Retrieval
    retrieved_docs = self._retrieve_context(query_features)
    
    # Step 3: Context processing
    processed_context = self._process_context(retrieved_docs, query_features)
    
    # Step 4: Reasoning
    reasoning_result = self.reasoner.reason(
        case=case,
        question=question,
        options=options,
        context=processed_context
    )
    
    # Step 5: Safety verification
    is_safe = self.safety_verifier.verify(
        answer=reasoning_result['answer'],
        reasoning=reasoning_result['reasoning'],
        context=processed_context
    )
    
    # Step 6: Hallucination detection
    has_hallucination = self.hallucination_detector.detect(
        reasoning=reasoning_result['reasoning'],
        context=processed_context
    )
    
    # Step 7: Confidence calibration
    calibrated_confidence = self.confidence_calibrator.calibrate(
        confidence=reasoning_result['confidence'],
        features={
            'reasoning_length': len(reasoning_result['reasoning']),
            'evidence_count': len(processed_context),
            'retrieval_quality': self._compute_retrieval_quality(retrieved_docs)
        }
    )
    
    return {
        'answer': reasoning_result['answer'],
        'reasoning': reasoning_result['reasoning'],
        'confidence': calibrated_confidence,
        'evidence': processed_context,
        'safety_verified': is_safe,
        'hallucination_detected': has_hallucination
    }
```

---

## Chain-of-Thought Reasoning

**File:** `src/reasoning/medical_reasoning.py`

### Purpose

Implements linear step-by-step reasoning for medical question answering.

### Experimental Results

- **Accuracy:** 34% (17/50 correct)
- **Avg Time:** 4,955ms (~5 seconds)
- **Brier Score:** 0.424
- **ECE:** 0.266 (best calibration)
- **Reasoning Coherence:** 32.7%

### Implementation

```python
class ChainOfThoughtReasoner:
    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        max_tokens: int = 512
    ):
        """
        Initialize CoT reasoner.
        
        Args:
            model: Ollama model name
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum reasoning chain length
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ollama.Client()
```

### Prompt Template

```python
COT_PROMPT = """
You are a medical expert answering a clinical question. Reason step-by-step using ONLY the provided context.

Context (Medical Guidelines):
{context}

Clinical Case:
{case}

Question:
{question}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Instructions:
1. Analyze the clinical presentation
2. Identify key symptoms, signs, and risk factors
3. Consider differential diagnoses
4. Apply relevant guidelines from the context
5. Evaluate each option against the guidelines
6. Select the most appropriate answer

CRITICAL: Base your reasoning ONLY on the provided context. Do not use external medical knowledge.

Reasoning (step-by-step):
"""
```

### Reasoning Process

```python
def reason(
    self,
    case: str,
    question: str,
    options: Dict[str, str],
    context: str
) -> Dict:
    """
    Generate Chain-of-Thought reasoning.
    
    Returns:
        {
            'answer': 'A|B|C|D',
            'reasoning': 'Step-by-step explanation',
            'confidence': float
        }
    """
    # Format prompt
    prompt = COT_PROMPT.format(
        context=context,
        case=case,
        question=question,
        option_a=options['A'],
        option_b=options['B'],
        option_c=options['C'],
        option_d=options['D']
    )
    
    # Generate reasoning
    response = self.client.generate(
        model=self.model,
        prompt=prompt,
        options={
            'temperature': self.temperature,
            'num_predict': self.max_tokens
        }
    )
    
    reasoning_text = response['response']
    
    # Extract answer
    answer = self._extract_answer(reasoning_text)
    
    # Estimate confidence
    confidence = self._estimate_confidence(reasoning_text, answer)
    
    return {
        'answer': answer,
        'reasoning': reasoning_text,
        'confidence': confidence
    }
```

### Example Output

```
Step 1: Analyze Clinical Presentation
The patient is a 58-year-old male presenting with sudden onset of severe, crushing chest pain radiating to the left arm. The pain started 45 minutes ago during physical activity (climbing stairs). He appears diaphoretic and anxious.

Step 2: Identify Key Symptoms and Risk Factors
- Chest pain: crushing, radiating to left arm (classic ACS symptom)
- Diaphoresis: autonomic response to cardiac ischemia
- Risk factors: age 58, male, hypertension, hyperlipidemia

Step 3: Consider Differential Diagnoses
Based on the presentation, the primary differential diagnosis is:
- Acute Coronary Syndrome (ACS) - most likely given classic symptoms
- Musculoskeletal pain - less likely given radiation pattern
- GERD - less likely given severity and radiation

Step 4: Apply Guideline Recommendations
From the context, the guidelines for ACS management state:
"Immediate Management (MONA + Antiplatelet): Aspirin 325mg chewed immediately, obtain ECG to assess for ST-segment changes."

Step 5: Evaluate Options
Option A: Obtain troponin levels and observe
- Delays definitive management, not immediate action

Option B: Administer aspirin 325mg and obtain ECG immediately
- Aligns with MONA protocol and ACS guidelines
- Aspirin provides immediate antiplatelet effect
- ECG required for STEMI vs NSTEMI differentiation

Option C: Start IV fluids and schedule stress test
- Stress test contraindicated in acute setting
- Not appropriate immediate action

Option D: Give sublingual nitroglycerin only
- Incomplete management, missing aspirin (critical)

Step 6: Select Answer
Based on guideline recommendations, Option B provides the most appropriate immediate management for suspected ACS.

Answer: B
Confidence: High (guideline-based recommendation)
```

---

## Tree-of-Thought Reasoning

**File:** `src/reasoning/tree_of_thought.py`

### Purpose

Implements multi-branch reasoning with explicit exploration of alternative diagnostic pathways.

### Experimental Results

- **Accuracy:** 52% (26/50 correct) - HIGHEST
- **Avg Time:** 41,367ms (~41 seconds)
- **Brier Score:** 0.344
- **ECE:** 0.310
- **Reasoning Coherence:** 43.3% (best)

### Implementation

```python
class TreeOfThoughtReasoner:
    def __init__(
        self,
        model: str = "llama3.1:8b",
        num_branches: int = 5,
        max_depth: int = 3,
        temperature: float = 0.3  # Slightly higher for diversity
    ):
        """
        Initialize ToT reasoner.
        
        Args:
            model: Ollama model name
            num_branches: Number of reasoning branches to explore
            max_depth: Maximum reasoning depth
            temperature: Sampling temperature for diversity
        """
        self.model = model
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.temperature = temperature
        self.client = ollama.Client()
```

### Multi-Branch Reasoning

```python
def reason(
    self,
    case: str,
    question: str,
    options: Dict[str, str],
    context: str
) -> Dict:
    """
    Tree-of-Thought reasoning with branch exploration.
    
    Process:
        1. Generate initial branches (differential diagnoses)
        2. Expand each branch with evidence evaluation
        3. Prune low-confidence branches
        4. Select highest-confidence conclusion
    
    Returns:
        {
            'answer': 'A|B|C|D',
            'reasoning': 'Multi-branch reasoning tree',
            'confidence': float,
            'branches_explored': int
        }
    """
    # Step 1: Generate initial branches
    branches = self._generate_branches(case, question, options, context)
    
    # Step 2: Expand and evaluate branches
    evaluated_branches = []
    for branch in branches:
        branch_score = self._evaluate_branch(branch, context)
        evaluated_branches.append((branch, branch_score))
    
    # Step 3: Prune low-confidence branches
    evaluated_branches.sort(key=lambda x: x[1], reverse=True)
    top_branches = evaluated_branches[:self.num_branches // 2]
    
    # Step 4: Select best branch
    best_branch, best_score = top_branches[0]
    
    # Step 5: Extract answer from best branch
    answer = self._extract_answer_from_branch(best_branch)
    
    # Format reasoning tree
    reasoning_tree = self._format_reasoning_tree(evaluated_branches, best_branch)
    
    return {
        'answer': answer,
        'reasoning': reasoning_tree,
        'confidence': best_score,
        'branches_explored': len(branches)
    }
```

### Branch Generation

```python
def _generate_branches(
    self,
    case: str,
    question: str,
    options: Dict[str, str],
    context: str
) -> List[Dict]:
    """
    Generate reasoning branches for each answer option.
    
    Returns:
        List of branch dictionaries, each containing:
        - hypothesis: "Option A is correct because..."
        - evidence: Supporting evidence from context
        - reasoning: Logical chain
    """
    branches = []
    
    for option_key, option_text in options.items():
        prompt = TOT_BRANCH_PROMPT.format(
            context=context,
            case=case,
            question=question,
            option_key=option_key,
            option_text=option_text
        )
        
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': self.temperature}
        )
        
        branches.append({
            'option': option_key,
            'hypothesis': f"Option {option_key} is correct",
            'reasoning': response['response']
        })
    
    return branches
```

### Example Output

```
TREE OF THOUGHT REASONING

Branch 1: Option A (Obtain troponin levels and observe)
Hypothesis: This is a conservative approach that gathers more data.
Evidence:
  - Troponin is the gold standard for MI diagnosis
  - Serial troponins can detect evolving MI
Reasoning:
  - Pro: Confirms diagnosis definitively
  - Con: Delays treatment (aspirin, reperfusion)
  - Con: Guideline recommends immediate aspirin
Guideline Match: Low (delays critical interventions)
Confidence: 0.20

Branch 2: Option B (Administer aspirin 325mg and obtain ECG immediately)
Hypothesis: Immediate antiplatelet therapy and diagnostic ECG.
Evidence:
  - Guideline: "Aspirin 325mg chewed immediately"
  - Guideline: "Obtain ECG to assess for ST-segment changes"
  - MONA protocol includes aspirin as first-line
Reasoning:
  - Pro: Immediate antiplatelet effect (reduces mortality)
  - Pro: ECG required for STEMI/NSTEMI differentiation
  - Pro: Aligns with evidence-based guidelines
  - Con: None identified
Guideline Match: High (exact protocol match)
Confidence: 0.95

Branch 3: Option C (Start IV fluids and schedule stress test)
Hypothesis: Rehydration and delayed diagnostic testing.
Evidence:
  - No guideline support for IV fluids as primary intervention
  - Stress test is contraindicated in acute ACS
Reasoning:
  - Pro: None for acute ACS management
  - Con: Stress test dangerous in acute setting (may precipitate MI)
  - Con: Delays definitive treatment
Guideline Match: None (contraindicated)
Confidence: 0.05

Branch 4: Option D (Give sublingual nitroglycerin only)
Hypothesis: Vasodilation for chest pain relief.
Evidence:
  - Guideline: "Nitroglycerin 0.4mg SL"
  - But also states: "MONA + Antiplatelet"
Reasoning:
  - Pro: Provides symptomatic relief
  - Con: Missing aspirin (critical antiplatelet)
  - Con: Incomplete management
Guideline Match: Partial (missing aspirin)
Confidence: 0.40

BRANCH EVALUATION:
Explored: 4 branches
Pruned: 2 branches (A, C)
Remaining: 2 branches (B, D)

FINAL SELECTION:
Best Branch: Branch 2 (Option B)
Reasoning: Highest guideline match, complete MONA protocol
Confidence: 0.95

Answer: B
```

---

## Structured Medical Reasoning

**File:** `src/improvements/structured_reasoner.py`

### Purpose

5-step clinical decision framework: Patient Profile → Differential → Evidence → Treatment → Selection

### Experimental Results

- **Accuracy:** 44% (22/50 correct)
- **Avg Time:** 26,991ms (~27 seconds)
- **Brier Score:** 0.295 (BEST calibration)
- **ECE:** 0.283
- **Reasoning Coherence:** 32.0%

### 5-Step Process

```python
class StructuredMedicalReasoner:
    def __init__(self, model: str = "llama3.1:8b"):
        """Initialize structured reasoner with 5-step framework."""
        self.model = model
        self.client = ollama.Client()
        
    def reason(
        self,
        case: str,
        question: str,
        options: Dict[str, str],
        context: str
    ) -> Dict:
        """
        5-step structured medical reasoning.
        
        Steps:
            1. Patient Profile: Extract demographics, symptoms, vitals
            2. Differential Diagnosis: Generate possible conditions (LLM)
            3. Evidence Analysis: Match symptoms to conditions
            4. Guideline Application: Apply treatment protocols
            5. Final Decision: Select answer with confidence
        """
        # Step 1: Patient Profile
        profile = self._extract_patient_profile(case)
        
        # Step 2: Differential Diagnosis (LLM-enhanced)
        differential = self._generate_differential(case, profile, context)
        
        # Step 3: Evidence Analysis
        evidence_scores = self._analyze_evidence(differential, options, context)
        
        # Step 4: Guideline Application
        guideline_matches = self._match_guidelines(options, context)
        
        # Step 5: Final Decision with LLM Verification
        final_decision = self._make_decision(
            profile,
            differential,
            evidence_scores,
            guideline_matches,
            options
        )
        
        # Verify reasoning coherence with LLM
        verified = self._verify_coherence(final_decision)
        
        return final_decision
```

### Step 1: Patient Profile Extraction

```python
def _extract_patient_profile(self, case: str) -> Dict:
    """
    Extract structured clinical features.
    
    Returns:
        {
            'demographics': {'age': int, 'gender': str},
            'symptoms': List[str],
            'vitals': Dict[str, float],
            'risk_factors': List[str],
            'acuity': 'emergency|urgent|routine'
        }
    """
    import re
    
    profile = {
        'demographics': {},
        'symptoms': [],
        'vitals': {},
        'risk_factors': [],
        'acuity': 'routine'
    }
    
    # Extract age
    age_match = re.search(r'(\d+)-year-old', case)
    if age_match:
        profile['demographics']['age'] = int(age_match.group(1))
    
    # Extract gender
    if 'male' in case.lower():
        profile['demographics']['gender'] = 'male'
    elif 'female' in case.lower():
        profile['demographics']['gender'] = 'female'
    
    # Extract vitals
    bp_match = re.search(r'BP[:\s]+(\d+)/(\d+)', case)
    if bp_match:
        profile['vitals']['BP_systolic'] = int(bp_match.group(1))
        profile['vitals']['BP_diastolic'] = int(bp_match.group(2))
    
    # Extract symptoms (keyword matching)
    symptom_keywords = [
        'chest pain', 'dyspnea', 'fever', 'headache', 'nausea',
        'vomiting', 'diarrhea', 'fatigue', 'cough', 'palpitations'
    ]
    for symptom in symptom_keywords:
        if symptom in case.lower():
            profile['symptoms'].append(symptom)
    
    # Determine acuity
    emergency_keywords = ['acute', 'sudden', 'severe', 'emergency']
    if any(keyword in case.lower() for keyword in emergency_keywords):
        profile['acuity'] = 'emergency'
    
    return profile
```

### Step 2: Differential Diagnosis Generation

```python
def _generate_differential(
    self,
    case: str,
    profile: Dict,
    context: str
) -> List[Dict]:
    """
    Generate differential diagnoses using LLM.
    
    Returns:
        List of {
            'condition': str,
            'likelihood': float,
            'supporting_evidence': List[str]
        }
    """
    prompt = f"""
Based on the following clinical case, generate a differential diagnosis list.

Case: {case}

Patient Profile:
- Age: {profile['demographics'].get('age', 'unknown')}
- Gender: {profile['demographics'].get('gender', 'unknown')}
- Symptoms: {', '.join(profile['symptoms'])}
- Acuity: {profile['acuity']}

Context (Guidelines):
{context[:2000]}  # Truncate for prompt length

Generate a differential diagnosis with likelihood scores (0-1).
Format: JSON array of {"condition": "...", "likelihood": 0.0-1.0}
"""
    
    response = self.client.generate(
        model=self.model,
        prompt=prompt,
        format='json'
    )
    
    import json
    differential = json.loads(response['response'])
    
    return differential
```

### Example Output

```
STRUCTURED MEDICAL REASONING

Step 1: Patient Profile Extraction
Extracted 3 symptoms: chest pain, diaphoresis, dyspnea
Demographics: {'age': 58, 'gender': 'male', 'age_group': 'middle-aged'}
Vitals: {'BP': '145/90', 'HR': 98, 'RR': 22, 'SpO2': 96}
Acuity: emergency

Step 2: Differential Diagnosis Generation (LLM-Enhanced)
Generated 3 differential diagnoses:
  1. Acute Coronary Syndrome (likelihood: 0.90)
     - Supporting: chest pain, radiation, diaphoresis, risk factors
  2. Pulmonary Embolism (likelihood: 0.15)
     - Supporting: dyspnea, sudden onset
  3. Panic Attack (likelihood: 0.05)
     - Supporting: anxiety, diaphoresis

Step 3: Evidence Analysis
Scored evidence for 4 options:
  Option A: 0.30 (delayed diagnosis)
  Option B: 0.90 (matches ACS guideline)
  Option C: 0.10 (contraindicated stress test)
  Option D: 0.50 (incomplete management)

Step 4: Guideline Application
Matched guidelines for top 4 options:
  Option B: ACS management guideline (MONA + Antiplatelet)
    - "Aspirin 325mg chewed immediately"
    - "Obtain ECG to assess for ST-segment changes"

Step 5: Final Decision with LLM Verification
Selected: Option B
Confidence: 0.90
Reasoning: Aligns with highest-likelihood differential (ACS) and evidence-based guideline
LLM Verification: PASS (reasoning coherent and evidence-based)

Answer: B
Calibrated Confidence: 0.88 (adjusted for historical performance)
```

---

## Hybrid Reasoning

### Strategy

Adaptive reasoning method selection based on question complexity and confidence:

```
1. Primary: Chain-of-Thought (fast, general cases)
2. Escalation: If complexity > 0.7 AND confidence < 0.75, use Tree-of-Thought
3. Fallback: If CoT/ToT unavailable, use Structured Medical Reasoning
```

### Implementation

```python
class HybridReasoner:
    def __init__(self):
        self.cot_reasoner = ChainOfThoughtReasoner()
        self.tot_reasoner = TreeOfThoughtReasoner()
        self.structured_reasoner = StructuredMedicalReasoner()
        
    def reason(
        self,
        case: str,
        question: str,
        options: Dict[str, str],
        context: str
    ) -> Dict:
        """
        Hybrid reasoning with adaptive method selection.
        """
        # Step 1: Assess question complexity
        complexity = self._assess_complexity(case, question)
        
        # Step 2: Try Chain-of-Thought first
        cot_result = self.cot_reasoner.reason(case, question, options, context)
        
        # Step 3: Escalate to ToT if needed
        if complexity > 0.7 and cot_result['confidence'] < 0.75:
            print(f"Escalating to ToT (complexity={complexity:.2f}, confidence={cot_result['confidence']:.2f})")
            tot_result = self.tot_reasoner.reason(case, question, options, context)
            return tot_result
        
        return cot_result
```

### Complexity Assessment

```python
def _assess_complexity(self, case: str, question: str) -> float:
    """
    Assess question complexity (0-1 scale).
    
    Factors:
        - Multi-system involvement
        - Contradictory symptoms
        - Rare conditions
        - Ambiguous presentation
        - Length of case description
    
    Returns:
        Complexity score (0-1)
    """
    complexity = 0.0
    
    # Factor 1: Case length (longer = more complex)
    words = len(case.split())
    complexity += min(words / 200, 0.3)
    
    # Factor 2: Multiple systems mentioned
    systems = ['cardiovascular', 'respiratory', 'gastrointestinal', 
              'neurological', 'renal', 'endocrine']
    system_count = sum(1 for sys in systems if sys in case.lower())
    complexity += min(system_count * 0.15, 0.3)
    
    # Factor 3: Contradictory findings
    if 'however' in case.lower() or 'but' in case.lower():
        complexity += 0.2
    
    # Factor 4: Question type
    if 'differential' in question.lower():
        complexity += 0.2
    
    return min(complexity, 1.0)
```

---

## Query Understanding

**File:** `src/reasoning/query_understanding.py`

### Purpose

Enhances queries with clinical feature extraction, medical concept expansion, and specialty detection.

### Implementation

```python
class QueryUnderstanding:
    def __init__(self):
        self.clinical_extractor = ClinicalFeatureExtractor()
        self.concept_expander = MedicalConceptExpander()
        self.specialty_detector = SpecialtyAdapter()
        
    def understand(self, case: str, question: str) -> Dict:
        """
        Comprehensive query understanding.
        
        Returns:
            {
                'clinical_features': {...},
                'expanded_concepts': [...],
                'specialty': 'cardiology|neurology|...',
                'acuity': 'emergency|urgent|routine',
                'question_type': 'diagnosis|treatment|management'
            }
        """
        features = {
            'clinical_features': self.clinical_extractor.extract(case),
            'expanded_concepts': self.concept_expander.expand(case),
            'specialty': self.specialty_detector.detect(case),
            'acuity': self._detect_acuity(case),
            'question_type': self._classify_question(question)
        }
        
        return features
```

---

## Performance Analysis

### Method Comparison

| Method | Accuracy | Time (ms) | Brier | ECE | Best For |
|--------|----------|-----------|-------|-----|----------|
| Tree-of-Thought | **52%** | 41,367 | 0.344 | 0.310 | Complex cases |
| Structured Medical | 44% | 26,991 | **0.295** | 0.283 | Calibration |
| Chain-of-Thought | 34% | **4,955** | 0.424 | **0.266** | Speed |

### Recommendations

**Production System:** Use Hybrid Approach
- Primary: CoT (fast for 70% of questions)
- Escalation: ToT (accuracy for complex 30%)
- Expected: 46-48% accuracy, 12-15s avg time

**Future Improvements:**
1. Fine-tune LLM on medical reasoning examples
2. Add medical knowledge graph integration
3. Implement multi-agent reasoning
4. Add self-consistency checking

---

## Related Documentation

- [Part 2: RAG Implementation](part_2_rag_implementation.md)
- [Part 4: Experiments](part_4_experiments.md)
- [Improvements Documentation](improvements_documentation.md)

---

**Documentation Author:** Shreya Uprety
