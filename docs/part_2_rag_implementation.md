# Part 2: RAG Implementation

**Author:** Shreya Uprety  
**Last Updated:** December 11, 2025

---

## Overview

This document details the multi-stage Retrieval-Augmented Generation (RAG) pipeline implementation, including retrieval strategies, context processing, and integration with reasoning engines.

---

## Multi-Stage Retrieval Architecture

```
Input Query
    ↓
┌───────────────────────────────────────────────┐
│        Stage 1: Broad Retrieval (k=150)       │
│  ┌─────────┐  ┌─────────┐  ┌──────────────┐ │
│  │  BM25   │  │  FAISS  │  │ Concept-First│ │
│  │ Lexical │  │Semantic │  │ UMLS Expand  │ │
│  └────┬────┘  └────┬────┘  └──────┬───────┘ │
│       └────────────┼───────────────┘         │
│                    ↓                          │
│         Merge & Deduplicate (150 docs)       │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│      Stage 2: Focused Retrieval (k=100)       │
│  • Multi-Query Expansion                      │
│  • Symptom Synonym Injection                  │
│  • Guideline Prioritization                   │
│  • Filter to top 100                          │
└───────────────────┬───────────────────────────┘
                    ↓
┌───────────────────────────────────────────────┐
│       Stage 3: Reranking (k=30)               │
│  • Cross-Encoder Reranking                    │
│  • Context Pruning                            │
│  • Evidence Quality Scoring                   │
│  • Select final 30 documents                  │
└───────────────────┬───────────────────────────┘
                    ↓
            Final Context (30 docs)
                    ↓
            Reasoning Engine
```

---

## Retrieval Strategies

### 1. BM25 Retrieval (Lexical)

**Algorithm:** BM25Okapi with k1=1.5, b=0.75

**Strengths:**
- Excellent for exact medical term matching
- Fast (milliseconds)
- No embedding computation

**Weaknesses:**
- Misses semantic relationships
- Sensitive to vocabulary mismatch

**Implementation:**
```python
# src/retrieval/bm25_retriever.py

from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents):
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
        
    def retrieve(self, query, k=150):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [(i, scores[i]) for i in top_k_indices]
```

---

### 2. FAISS Retrieval (Semantic)

**Model:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

**Strengths:**
- Captures semantic meaning
- Robust to paraphrasing
- Finds conceptually related content

**Weaknesses:**
- General-purpose model (not medical-domain)
- Misses exact terminology
- Computationally expensive

**Implementation:**
```python
# src/retrieval/faiss_store.py

import faiss
from sentence_transformers import SentenceTransformer

class FAISSStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = faiss.IndexFlatL2(384)  # L2 distance
        
    def build_index(self, documents):
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        self.index.add(embeddings.astype('float32'))
        
    def search(self, query, k=150):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return [(idx, 1/(1+dist)) for idx, dist in zip(indices[0], distances[0])]
```

**Known Issue:**
- Using general-purpose embeddings causes accuracy drop (60% vs 80%)
- Root cause identified in `src/models/embeddings.py` lines 116-119
- Recommended fix: Switch to PubMedBERT

---

### 3. Concept-First Retrieval (Medical Domain)

**Process:**
1. Extract medical entities from query (NER)
2. Expand with UMLS synonyms and related concepts
3. Retrieve using expanded query
4. Score by concept coverage

**Example:**
```
Query: "chest pain radiating to left arm"
    ↓
Entities: ["chest pain", "left arm"]
    ↓
UMLS Expansion:
  - chest pain → angina, cardiac pain, myocardial ischemia
  - left arm radiation → referred pain, cardiac radiation
    ↓
Expanded Query: "chest pain angina cardiac pain myocardial ischemia left arm referred pain"
    ↓
Retrieve documents matching expanded concepts
```

**Implementation:**
```python
# src/retrieval/concept_first_retriever.py

class ConceptFirstRetriever:
    def __init__(self, umls_synonyms, umls_expansion, faiss_store):
        self.synonyms = umls_synonyms
        self.expansion = umls_expansion
        self.faiss = faiss_store
        
    def retrieve(self, query, k=150):
        # Extract medical concepts
        concepts = self.extract_concepts(query)
        
        # Expand with UMLS
        expanded_concepts = []
        for concept in concepts:
            expanded_concepts.extend(self.synonyms.get(concept, []))
            expanded_concepts.extend(self.expansion.get(concept, {}).get('is_a', []))
            expanded_concepts.extend(self.expansion.get(concept, {}).get('related', []))
        
        # Create expanded query
        expanded_query = f"{query} {' '.join(expanded_concepts)}"
        
        # Retrieve with expanded query
        return self.faiss.search(expanded_query, k)
```

---

### 4. Hybrid Retrieval

**Formula:**
```
hybrid_score = α × bm25_score + (1-α) × semantic_score
```

**Parameters:**
- α = 0.5 (equal weighting)
- Adjustable based on query type

**Implementation:**
```python
# src/retrieval/hybrid_retriever.py

class HybridRetriever:
    def __init__(self, bm25_retriever, faiss_store, alpha=0.5):
        self.bm25 = bm25_retriever
        self.faiss = faiss_store
        self.alpha = alpha
        
    def retrieve(self, query, k=150):
        bm25_results = self.bm25.retrieve(query, k=300)
        faiss_results = self.faiss.search(query, k=300)
        
        # Normalize scores
        bm25_scores = self.normalize_scores(bm25_results)
        faiss_scores = self.normalize_scores(faiss_results)
        
        # Combine scores
        combined = {}
        for idx, score in bm25_scores.items():
            combined[idx] = self.alpha * score
        for idx, score in faiss_scores.items():
            combined[idx] = combined.get(idx, 0) + (1-self.alpha) * score
        
        # Return top k
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
```

---

## Multi-Stage Pipeline

### Stage 1: Broad Retrieval (k=150)

**Goal:** Maximize recall, cast wide net

**Process:**
1. Run BM25 retrieval → 150 docs
2. Run FAISS retrieval → 150 docs
3. Run Concept-First retrieval → 150 docs
4. Merge and deduplicate → ~150 unique docs
5. Sort by highest score from any retriever

**Implementation:**
```python
def stage1_retrieval(query, k=150):
    bm25_docs = bm25_retriever.retrieve(query, k)
    faiss_docs = faiss_store.search(query, k)
    concept_docs = concept_first_retriever.retrieve(query, k)
    
    # Merge with score tracking
    doc_scores = {}
    for doc_id, score in bm25_docs + faiss_docs + concept_docs:
        doc_scores[doc_id] = max(doc_scores.get(doc_id, 0), score)
    
    # Sort by highest score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:k]
```

---

### Stage 2: Focused Retrieval (k=100)

**Goal:** Refine with enhanced queries

**Enhancements:**
1. **Multi-Query Expansion:** Generate alternative phrasings
2. **Symptom Synonym Injection:** Add medical synonyms
3. **Guideline Prioritization:** Boost guideline-based documents

**Multi-Query Expansion Example:**
```
Original: "chest pain radiating to left arm"
    ↓
Alternative Queries:
  - "cardiac chest pain with arm radiation"
  - "angina with referred pain to upper extremity"
  - "chest discomfort spreading to left side"
```

**Implementation:**
```python
def stage2_retrieval(stage1_docs, query, k=100):
    # Generate alternative queries
    alt_queries = multi_query_expander.expand(query)
    
    # Inject symptom synonyms
    synonym_queries = [symptom_synonym_injector.inject(q) for q in alt_queries]
    
    # Retrieve for each query
    all_docs = []
    for q in [query] + synonym_queries:
        docs = faiss_store.search(q, k=50)
        all_docs.extend(docs)
    
    # Deduplicate and filter to stage1 docs
    doc_scores = {}
    for doc_id, score in all_docs:
        if doc_id in [d[0] for d in stage1_docs]:
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0), score)
    
    # Boost guideline documents
    for doc_id in doc_scores:
        if is_guideline_document(doc_id):
            doc_scores[doc_id] *= 1.2
    
    # Return top 100
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:k]
```

---

### Stage 3: Reranking (k=30)

**Goal:** Precision optimization with cross-encoder

**Process:**
1. **Cross-Encoder Reranking:** Compute query-document relevance
2. **Context Pruning:** Remove redundant/irrelevant chunks
3. **Evidence Quality Scoring:** Prioritize high-quality evidence

**Cross-Encoder:**
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Computes relevance score for (query, document) pairs

**Implementation:**
```python
from sentence_transformers import CrossEncoder

def stage3_reranking(stage2_docs, query, k=30):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Get document texts
    doc_texts = [get_document_text(doc_id) for doc_id, _ in stage2_docs]
    
    # Compute cross-encoder scores
    pairs = [[query, doc] for doc in doc_texts]
    ce_scores = cross_encoder.predict(pairs)
    
    # Combine with stage2 scores (70% cross-encoder, 30% stage2)
    combined_scores = []
    for i, (doc_id, s2_score) in enumerate(stage2_docs):
        combined = 0.7 * ce_scores[i] + 0.3 * s2_score
        combined_scores.append((doc_id, combined))
    
    # Prune context (remove redundant)
    pruned_docs = context_pruner.prune(combined_scores)
    
    # Sort and return top 30
    sorted_docs = sorted(pruned_docs, key=lambda x: x[1], reverse=True)
    return sorted_docs[:k]
```

---

## Context Processing

### Context Pruning

**Goal:** Remove irrelevant and redundant information

**Methods:**
1. **Relevance Filtering:** Remove chunks with low query-relevance
2. **Redundancy Removal:** Deduplicate similar chunks
3. **Length Optimization:** Fit within LLM context window

**Implementation:**
```python
# src/improvements/context_pruner.py

class ContextPruner:
    def prune(self, documents, query, max_tokens=4000):
        # Filter by relevance threshold
        relevant_docs = [(d, s) for d, s in documents if s > 0.3]
        
        # Remove redundant documents (cosine similarity > 0.9)
        unique_docs = self.remove_redundancy(relevant_docs)
        
        # Truncate to max tokens
        truncated_docs = self.truncate_to_max_tokens(unique_docs, max_tokens)
        
        return truncated_docs
```

---

## Integration with Reasoning

### Context Formatting

Final retrieved context is formatted for reasoning:

```python
def format_context(documents):
    context = "Relevant Medical Guidelines:\n\n"
    for i, (doc_id, score) in enumerate(documents):
        doc_text = get_document_text(doc_id)
        metadata = get_document_metadata(doc_id)
        
        context += f"[Source {i+1}]: {metadata['guideline']}\n"
        context += f"{doc_text}\n\n"
    
    return context
```

### Reasoning Prompt

```
Context:
{formatted_context}

Clinical Case:
{case_description}

Question:
{question}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Instructions:
Based ONLY on the provided context, reason step-by-step to select the most appropriate answer.
Do not use external medical knowledge.

Reasoning:
```

---

## Performance Metrics

### Current Performance (50 cases)

**Retrieval Metrics:**
- Precision@5: 4%
- Recall@5: 20%
- MAP: 0.118
- MRR: 0.215
- Context Relevance: 0.428

**Root Cause:** General-purpose embeddings (MiniLM-L6-v2)

**Target Performance (with medical embeddings):**
- Precision@5: 30-40%
- Recall@5: 75-85%
- MAP: 0.500-0.600
- Context Relevance: 1.5+

---

## Known Issues and Solutions

### Issue 1: Low Retrieval Precision (4%)

**Root Cause:** General-purpose embeddings don't capture medical semantics

**Solution:**
```python
# In src/models/embeddings.py (lines 116-119)

# Current (REMOVE):
# model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Recommended (ADD):
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
# OR
model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
```

### Issue 2: Redundant Retrieved Documents

**Root Cause:** Multiple retrieval strategies return similar documents

**Solution:** Enhanced context pruning with cosine similarity deduplication

### Issue 3: Long Context Window

**Root Cause:** 30 documents exceed LLM window for some models

**Solution:** Adaptive k_stage3 based on document lengths

---

## Related Documentation

- [Retrieval Documentation](retrieval_documentation.md)
- [Reasoning Documentation](reasoning_documentation.md)
- [Part 3: Evaluation Framework](part_3_evaluation_framework.md)

---

**Documentation Author:** Shreya Uprety
