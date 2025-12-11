# Retrieval Documentation

**Author:** Shreya Uprety  

---

## Table of Contents

1. [Overview](#overview)
2. [BM25 Retriever](#bm25-retriever)
3. [FAISS Store](#faiss-store)
4. [Concept-First Retriever](#concept-first-retriever)
5. [Hybrid Retriever](#hybrid-retriever)
6. [Multi-Stage Retriever](#multi-stage-retriever)
7. [Performance Comparison](#performance-comparison)

---

## Overview

The `src/retrieval/` module implements multiple retrieval strategies for finding relevant medical guidelines given a clinical question. The module supports lexical (BM25), semantic (FAISS), concept-based, hybrid, and multi-stage retrieval.

**Key Insight:** Retrieval strategy choice significantly impacts final accuracy. Best performers: Concept-First (MAP: 0.212) and Semantic-First (MAP: 0.213).

---

## BM25 Retriever

**File:** `src/retrieval/bm25_retriever.py`

### Purpose

Implements BM25 (Best Matching 25) algorithm for lexical retrieval based on term frequency and inverse document frequency.

### Implementation

```python
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    def __init__(
        self,
        documents: List[str],
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of document texts
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            
        BM25 Parameters:
            k1: Controls term frequency saturation
                - Higher k1: More weight to term frequency
                - Lower k1: Earlier saturation
                - Typical: 1.2-2.0
            
            b: Controls length normalization
                - b=1: Full normalization
                - b=0: No normalization
                - Typical: 0.75
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
```

### Methods

#### `retrieve(query: str, k: int = 150) -> List[Tuple[int, float]]`

```python
def retrieve(
    self,
    query: str,
    k: int = 150
) -> List[Tuple[int, float]]:
    """
    Retrieve top-k documents for query using BM25.
    
    Args:
        query: Search query
        k: Number of documents to retrieve
        
    Returns:
        List of (document_index, score) tuples sorted by score
        
    Process:
        1. Tokenize query
        2. Compute BM25 scores for all documents
        3. Sort documents by score
        4. Return top k
    """
    # Tokenize query
    tokenized_query = query.lower().split()
    
    # Get BM25 scores
    scores = self.bm25.get_scores(tokenized_query)
    
    # Sort and get top k
    top_k_indices = np.argsort(scores)[-k:][::-1]
    
    return [(idx, scores[idx]) for idx in top_k_indices]
```

### BM25 Scoring Formula

```
BM25(q, d) = Σ IDF(qi) × (f(qi, d) × (k1 + 1)) / (f(qi, d) + k1 × (1 - b + b × |d| / avgdl))

where:
  q = query
  d = document
  qi = query term i
  f(qi, d) = term frequency of qi in d
  |d| = document length
  avgdl = average document length
  IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
  N = total documents
  df(qi) = documents containing qi
```

### Performance Characteristics

**Strengths:**
- Extremely fast (1.4ms avg query time)
- Excellent for exact medical term matching
- No embedding computation required
- Memory efficient

**Weaknesses:**
- Misses semantic relationships (e.g., "MI" ≠ "myocardial infarction")
- Vocabulary mismatch problem
- Case-sensitive without preprocessing

**Best Use Cases:**
- Queries with specific medical terminology
- Known abbreviations or drug names
- Exact protocol matching

**Experimental Results:**
- MAP: 0.207
- MRR: 0.414
- Precision@5: 17.4%
- Recall@5: 43.5%

---

## FAISS Store

**File:** `src/retrieval/faiss_store.py`

### Purpose

Implements dense vector retrieval using Facebook AI Similarity Search (FAISS) for semantic search based on sentence embeddings.

### Implementation

```python
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class FAISSStore:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = True,
        fp16: bool = True
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_model: SentenceTransformer model name
            use_gpu: Use GPU acceleration
            fp16: Use half-precision (FP16) for embeddings
            
        Current Model Issue:
            all-MiniLM-L6-v2 is general-purpose (not medical domain)
            → Causes 20% accuracy drop
            
        Recommended Fix:
            Use microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
            → Expected +20% accuracy improvement
        """
        self.model_name = embedding_model
        self.use_gpu = use_gpu
        self.fp16 = fp16
        
        # Load embedding model
        self.model = SentenceTransformer(embedding_model)
        
        if use_gpu:
            self.model = self.model.cuda()
        
        # FAISS index (initialized on build_index)
        self.index = None
        self.documents = None
```

### Methods

#### `build_index(documents: List[str])`

```python
def build_index(self, documents: List[str]):
    """
    Build FAISS index from documents.
    
    Args:
        documents: List of document texts
        
    Process:
        1. Generate embeddings for all documents
        2. Create FAISS index (IndexFlatL2)
        3. Add embeddings to index
        4. Store document metadata
    """
    self.documents = documents
    
    print(f"Generating embeddings for {len(documents)} documents...")
    
    # Generate embeddings
    embeddings = self.model.encode(
        documents,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=False  # L2 distance doesn't require normalization
    )
    
    # Convert to FP16 if requested
    if self.fp16:
        embeddings = embeddings.astype('float16')
    
    # Create FAISS index
    dimension = embeddings.shape[1]  # 384 for MiniLM-L6-v2
    self.index = faiss.IndexFlatL2(dimension)
    
    if self.use_gpu:
        # Move index to GPU
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    
    # Add embeddings to index
    self.index.add(embeddings.astype('float32'))
    
    print(f"Built index with {self.index.ntotal} vectors")
```

#### `search(query: str, k: int = 150) -> List[Tuple[int, float]]`

```python
def search(
    self,
    query: str,
    k: int = 150
) -> List[Tuple[int, float]]:
    """
    Search for top-k most similar documents.
    
    Args:
        query: Search query
        k: Number of results
        
    Returns:
        List of (document_index, similarity_score) tuples
        
    Process:
        1. Encode query to embedding
        2. Search FAISS index for nearest neighbors
        3. Convert L2 distances to similarity scores
        4. Return top k results
    """
    # Encode query
    query_embedding = self.model.encode([query], convert_to_numpy=True)
    
    # Search index
    distances, indices = self.index.search(query_embedding.astype('float32'), k)
    
    # Convert L2 distances to similarity scores (smaller distance = higher similarity)
    # Similarity = 1 / (1 + distance)
    similarities = [1 / (1 + dist) for dist in distances[0]]
    
    # Return (index, score) tuples
    return [(int(idx), float(sim)) for idx, sim in zip(indices[0], similarities)]
```

### FAISS Index Types

**IndexFlatL2 (Current):**
- Brute-force exhaustive search
- Exact nearest neighbors
- No approximation
- Best for small datasets (<1M vectors)

**Alternative Indexes:**
```python
# For faster search on larger datasets:

# IVF (Inverted File Index)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
# nlist: number of clusters
# Approximate search, 10-100x faster

# HNSW (Hierarchical Navigable Small World)
index = faiss.IndexHNSWFlat(dimension, M=32)
# M: number of connections per layer
# Best accuracy/speed tradeoff

# Product Quantization (memory efficient)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist=100, m=8, bits=8)
# Compresses vectors, 8-64x memory reduction
```

### Performance Characteristics

**Strengths:**
- Captures semantic meaning
- Robust to paraphrasing
- Finds conceptually related content

**Weaknesses:**
- General-purpose model (not medical-domain)
- Computationally expensive (8.58ms vs 1.4ms for BM25)
- Requires GPU for fast inference
- Misses exact medical terminology

**Experimental Results:**
- MAP: 0.211
- MRR: 0.422
- Precision@5: 17.6%
- Recall@5: 44.0%

**Known Issue - Root Cause of Accuracy Drop:**

Location: `src/models/embeddings.py` lines 116-119

```python
# CURRENT (CAUSES ACCURACY DROP):
self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# RECOMMENDED FIX (EXPECTED +20% ACCURACY):
self.model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

# OR:
self.model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
```

**Impact of Medical Embeddings (Projected):**
```
Current (MiniLM-L6-v2):
  Accuracy: 54%
  Precision@5: 10.8%
  Recall@5: 54%

With PubMedBERT:
  Accuracy: 75-80% (+21-26%)
  Precision@5: 30-40% (+19-29%)
  Recall@5: 75-85% (+21-31%)
```

---

## Concept-First Retriever

**File:** `src/retrieval/concept_first_retriever.py`

### Purpose

Implements medical concept expansion using UMLS (Unified Medical Language System) followed by semantic retrieval.

### Implementation

```python
import json
from typing import List, Dict, Tuple
from src.retrieval.faiss_store import FAISSStore

class ConceptFirstRetriever:
    def __init__(
        self,
        faiss_store: FAISSStore,
        umls_synonyms_path: str = "data/umls_synonyms.json",
        umls_expansion_path: str = "data/umls_expansion.json"
    ):
        """
        Initialize concept-first retriever.
        
        Args:
            faiss_store: FAISS store for semantic search
            umls_synonyms_path: Path to UMLS synonym mappings
            umls_expansion_path: Path to UMLS concept expansion
            
        UMLS Integration:
            - 500+ medical concepts
            - 3,000+ synonyms
            - Hierarchical relationships (is_a, related, causes)
        """
        self.faiss = faiss_store
        
        # Load UMLS data
        with open(umls_synonyms_path) as f:
            self.synonyms = json.load(f)
        
        with open(umls_expansion_path) as f:
            self.expansion = json.load(f)
```

### Methods

#### `retrieve(query: str, k: int = 150) -> List[Tuple[int, float]]`

```python
def retrieve(
    self,
    query: str,
    k: int = 150
) -> List[Tuple[int, float]]:
    """
    Retrieve documents using concept expansion.
    
    Args:
        query: Search query
        k: Number of results
        
    Returns:
        List of (document_index, score) tuples
        
    Process:
        1. Extract medical concepts from query
        2. Expand with UMLS synonyms and relationships
        3. Create enriched query
        4. Perform semantic search with FAISS
        5. Return top k results
    """
    # Step 1: Extract medical concepts
    concepts = self.extract_concepts(query)
    
    # Step 2: Expand concepts
    expanded_concepts = []
    for concept in concepts:
        # Add synonyms
        expanded_concepts.extend(self.synonyms.get(concept, []))
        
        # Add hierarchical relationships
        if concept in self.expansion:
            expanded_concepts.extend(self.expansion[concept].get('is_a', []))
            expanded_concepts.extend(self.expansion[concept].get('related', []))
            expanded_concepts.extend(self.expansion[concept].get('causes', []))
    
    # Step 3: Create enriched query
    enriched_query = f"{query} {' '.join(expanded_concepts)}"
    
    # Step 4: Semantic search
    return self.faiss.search(enriched_query, k)
```

#### `extract_concepts(query: str) -> List[str]`

```python
def extract_concepts(self, query: str) -> List[str]:
    """
    Extract medical concepts from query.
    
    Args:
        query: Search query
        
    Returns:
        List of extracted medical concepts
        
    Methods:
        1. Rule-based extraction (regex patterns)
        2. Dictionary matching (UMLS concepts)
        3. Named Entity Recognition (future: scispacy)
    """
    concepts = []
    query_lower = query.lower()
    
    # Check against UMLS concept dictionary
    for concept in self.synonyms.keys():
        if concept in query_lower:
            concepts.append(concept)
        
        # Check synonyms
        for synonym in self.synonyms[concept]:
            if synonym in query_lower:
                concepts.append(concept)
                break
    
    return list(set(concepts))  # Deduplicate
```

### Example Concept Expansion

**Input Query:**
```
"A 58-year-old man with chest pain radiating to left arm"
```

**Step 1: Extract Concepts:**
```python
concepts = ["chest_pain"]
```

**Step 2: Expand Concepts:**
```python
# From umls_synonyms.json:
synonyms = ["angina", "cardiac pain", "thoracic pain"]

# From umls_expansion.json:
is_a = ["angina", "cardiac pain"]
related = ["dyspnea", "palpitations", "myocardial_infarction"]
causes = ["acute_coronary_syndrome", "pulmonary_embolism", "pneumonia"]

expanded_concepts = synonyms + is_a + related + causes
```

**Step 3: Enriched Query:**
```
"A 58-year-old man with chest pain radiating to left arm angina cardiac pain thoracic pain dyspnea palpitations myocardial_infarction acute_coronary_syndrome pulmonary_embolism"
```

**Result:** Retrieves guidelines for ACS, chest pain, MI, PE, etc.

### Performance Characteristics

**Experimental Results (BEST PERFORMER):**
- MAP: 0.212 (highest)
- MRR: 0.424 (highest)
- Precision@5: 18.0% (highest)
- Recall@5: 45.0% (highest)
- Avg Query Time: 11.62ms

**Why It Works:**
1. Medical concepts ensure domain-specific expansion
2. UMLS provides high-quality medical relationships
3. Combines lexical precision with semantic recall
4. Minimal latency overhead

**Best For:** Medical question answering (current best choice)

---

## Hybrid Retriever

**File:** `src/retrieval/hybrid_retriever.py`

### Purpose

Combines BM25 (lexical) and FAISS (semantic) scores with weighted fusion.

### Implementation

```python
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_store import FAISSStore
from typing import List, Tuple, Dict
import numpy as np

class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        faiss_store: FAISSStore,
        alpha: float = 0.65  # Weight for FAISS (0.65 best in experiments)
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_retriever: BM25 retriever instance
            faiss_store: FAISS store instance
            alpha: Weight for semantic scores (1-alpha for BM25)
            
        Fusion Formula:
            hybrid_score = alpha × semantic_score + (1-alpha) × bm25_score
            
        Weight Tuning Results:
            alpha=0.5: MAP=0.210
            alpha=0.65: MAP=0.211 (best)
            alpha=0.35: MAP=0.209
        """
        self.bm25 = bm25_retriever
        self.faiss = faiss_store
        self.alpha = alpha
```

### Methods

#### `retrieve(query: str, k: int = 150) -> List[Tuple[int, float]]`

```python
def retrieve(
    self,
    query: str,
    k: int = 150
) -> List[Tuple[int, float]]:
    """
    Retrieve using hybrid BM25 + FAISS fusion.
    
    Args:
        query: Search query
        k: Number of results
        
    Returns:
        List of (document_index, score) tuples
        
    Process:
        1. Retrieve from BM25 (k=300 for coverage)
        2. Retrieve from FAISS (k=300 for coverage)
        3. Normalize scores to [0, 1]
        4. Combine with weighted fusion
        5. Sort and return top k
    """
    # Retrieve from both methods (retrieve more for better coverage)
    bm25_results = self.bm25.retrieve(query, k=300)
    faiss_results = self.faiss.search(query, k=300)
    
    # Convert to dictionaries for easy lookup
    bm25_scores = {idx: score for idx, score in bm25_results}
    faiss_scores = {idx: score for idx, score in faiss_results}
    
    # Normalize scores to [0, 1]
    bm25_normalized = self.normalize_scores(bm25_scores)
    faiss_normalized = self.normalize_scores(faiss_scores)
    
    # Combine scores
    all_doc_ids = set(bm25_normalized.keys()) | set(faiss_normalized.keys())
    
    combined_scores = {}
    for doc_id in all_doc_ids:
        bm25_score = bm25_normalized.get(doc_id, 0.0)
        faiss_score = faiss_normalized.get(doc_id, 0.0)
        
        combined_scores[doc_id] = (
            self.alpha * faiss_score + (1 - self.alpha) * bm25_score
        )
    
    # Sort by combined score
    sorted_docs = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_docs[:k]

def normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
    """
    Normalize scores to [0, 1] range using min-max normalization.
    
    Args:
        scores: Dictionary of {doc_id: score}
        
    Returns:
        Normalized scores
    """
    if not scores:
        return {}
    
    min_score = min(scores.values())
    max_score = max(scores.values())
    
    if max_score == min_score:
        return {doc_id: 1.0 for doc_id in scores.keys()}
    
    return {
        doc_id: (score - min_score) / (max_score - min_score)
        for doc_id, score in scores.items()
    }
```

### Performance Characteristics

**Experimental Results:**
- MAP: 0.211
- MRR: 0.421
- Precision@5: 17.8%
- Recall@5: 44.5%
- Avg Query Time: 8.33ms

**Analysis:**
- Marginal improvement over single methods
- Balanced performance across query types
- Not significantly better than Concept-First

---

## Multi-Stage Retriever

**File:** `src/retrieval/multi_stage_retriever.py`

### Purpose

Three-stage pipeline: broad retrieval → focused retrieval → cross-encoder reranking.

### Implementation

```python
from sentence_transformers import CrossEncoder
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.bm25_retriever import BM25Retriever

class MultiStageRetriever:
    def __init__(
        self,
        faiss_store: FAISSStore,
        bm25_retriever: BM25Retriever,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize multi-stage retriever.
        
        Args:
            faiss_store: FAISS store
            bm25_retriever: BM25 retriever
            cross_encoder_model: Cross-encoder for reranking
            
        Issue:
            General-purpose cross-encoder performs WORSE than no reranking
            → Needs medical domain fine-tuning
        """
        self.faiss = faiss_store
        self.bm25 = bm25_retriever
        self.cross_encoder = CrossEncoder(cross_encoder_model)
```

### Methods

#### `retrieve(query: str, k: int = 25) -> List[Tuple[int, float]]`

```python
def retrieve(
    self,
    query: str,
    k: int = 25
) -> List[Tuple[int, float]]:
    """
    Multi-stage retrieval pipeline.
    
    Args:
        query: Search query
        k: Final number of results
        
    Returns:
        List of (document_index, score) tuples
        
    Process:
        Stage 1 (Broad Retrieval, k=150):
          - FAISS semantic search
          - Cast wide net for recall
        
        Stage 2 (BM25 Filter, k=100):
          - Apply BM25 to stage 1 results
          - Refine with keyword matching
        
        Stage 3 (Cross-Encoder Rerank, k=25):
          - Compute query-document relevance
          - Precision optimization
    """
    # Stage 1: Broad semantic retrieval
    stage1_results = self.faiss.search(query, k=150)
    stage1_doc_ids = [idx for idx, _ in stage1_results]
    
    # Stage 2: BM25 filter
    stage2_results = self._filter_with_bm25(
        query,
        stage1_doc_ids,
        k=100
    )
    stage2_doc_ids = [idx for idx, _ in stage2_results]
    
    # Stage 3: Cross-encoder reranking
    stage3_results = self._rerank_with_cross_encoder(
        query,
        stage2_doc_ids,
        k=k
    )
    
    return stage3_results

def _rerank_with_cross_encoder(
    self,
    query: str,
    doc_ids: List[int],
    k: int
) -> List[Tuple[int, float]]:
    """
    Rerank documents using cross-encoder.
    
    Args:
        query: Search query
        doc_ids: Document IDs to rerank
        k: Number of results
        
    Returns:
        Reranked (document_index, score) tuples
    """
    # Get document texts
    doc_texts = [self.faiss.documents[idx] for idx in doc_ids]
    
    # Create query-document pairs
    pairs = [[query, doc] for doc in doc_texts]
    
    # Compute cross-encoder scores
    scores = self.cross_encoder.predict(pairs)
    
    # Sort by score
    scored_docs = list(zip(doc_ids, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_docs[:k]
```

### Performance Characteristics

**Experimental Results:**
- MAP: 0.204 (WORST)
- MRR: 0.408
- Precision@5: 17.0%
- Recall@5: 42.5%
- Avg Query Time: 2,878ms (SLOWEST)

**Analysis:**
- General-purpose cross-encoder hurts performance
- 290x slower than single-stage methods
- Needs medical domain cross-encoder

**Recommendation:** Remove or replace with medical cross-encoder

---

## Performance Comparison

### Experimental Results Summary

| Strategy | MAP | MRR | P@5 | R@5 | Time (ms) | Rank |
|----------|-----|-----|-----|-----|-----------|------|
| **Concept-First** | **0.212** | **0.424** | **18.0%** | **45.0%** | 11.62 | **1** |
| **Semantic-First** | **0.213** | **0.425** | 17.8% | 44.5% | 9.65 | **2** |
| Hybrid Linear | 0.211 | 0.421 | 17.8% | 44.5% | 8.33 | 3 |
| Single FAISS | 0.211 | 0.422 | 17.6% | 44.0% | 8.58 | 4 |
| Single BM25 | 0.207 | 0.414 | 17.4% | 43.5% | **1.40** | 5 |
| Multi-Stage | 0.204 | 0.408 | 17.0% | 42.5% | 2,878 | 6 |

### Recommendations

**For Production:** Use Concept-First Retriever
- Best accuracy (MAP: 0.212)
- Reasonable speed (11.62ms)
- Medical domain optimization

**For Speed-Critical:** Use BM25
- Fastest (1.40ms)
- Acceptable accuracy (MAP: 0.207)

**For Future:** Medical Embeddings + Concept Expansion
- Expected MAP: 0.500-0.600 (+90-100% improvement)
- Switch to PubMedBERT in `src/models/embeddings.py`

---

## Related Documentation

- [Part 2: RAG Implementation](part_2_rag_implementation.md)
- [Part 4: Experiments](part_4_experiments.md)
- [Models Documentation](models_documentation.md)

---

**Documentation Author:** Shreya Uprety
