"""
Concept-first retrieval pipeline:
1) BM25 keyword retrieval
2) Concept expansion (rule-based + UMLS synonyms)
3) Embedding retrieval (FAISS) on expanded query
4) Merge candidates
5) Cross-encoder rerank → top_k
"""

from typing import List, Optional
from dataclasses import dataclass
import time
import numpy as np

from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_store import FAISSVectorStore
from retrieval.document_processor import Document
from improvements.medical_concept_expander import MedicalConceptExpander
from retrieval.multi_stage_retriever import RetrievalResult


@dataclass
class ScoredDoc:
    doc: Document
    score: float
    source: str


class ConceptFirstRetriever:
    """
    BM25 → concept expansion → embedding search → merge → cross-encoder rerank.
    Designed to showcase concept-based query expansion (incl. UMLS synonyms).
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        faiss_store: FAISSVectorStore,
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",  # overridden by config in pipeline
        bm25_k: int = 40,
        embed_k: int = 40,
        stage3_k: int = 15,
        top_k: int = 5,
    ):
        self.bm25 = bm25
        self.faiss_store = faiss_store
        self.cross_encoder_name = cross_encoder_name
        self.bm25_k = bm25_k
        self.embed_k = embed_k
        self.stage3_k = stage3_k
        self.top_k = top_k
        self._cross_encoder = None
        self.expander = MedicalConceptExpander()

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(
                    self.cross_encoder_name,
                    max_length=512,
                    device="cpu"  # stay on CPU to reduce GPU/paging pressure
                )
            except Exception as e:
                print(f"[WARN] Cross-encoder load failed: {e}; using score merge only")
                self._cross_encoder = "fallback"
        return self._cross_encoder

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if top_k is None:
            top_k = self.top_k

        t0 = time.time()

        # Stage 1: BM25
        bm25_hits = self.bm25.search(query, top_k=self.bm25_k)

        # Stage 2: Concept expansion (rule-based + UMLS)
        expanded_query = self.expander.expand(query, max_expansions=4)

        # Stage 3: Embedding retrieval on expanded query
        faiss_hits = self.faiss_store.search(expanded_query, top_k=self.embed_k, score_threshold=-1.0)

        # Stage 4: Merge (simple max-score merge with source tagging)
        merged = {}
        for doc, score in bm25_hits:
            merged[id(doc)] = ScoredDoc(doc=doc, score=float(score), source="bm25")
        for doc, score in faiss_hits:
            if id(doc) in merged:
                merged[id(doc)].score = max(merged[id(doc)].score, float(score))
                merged[id(doc)].source = "bm25+faiss"
            else:
                merged[id(doc)] = ScoredDoc(doc=doc, score=float(score), source="faiss")

        candidates = list(merged.values())

        # Stage 5: Cross-encoder rerank to top_k
        ce = self._get_cross_encoder()
        if ce != "fallback" and candidates:
            pairs = [(expanded_query, sd.doc.content) for sd in candidates]
            scores = ce.predict(pairs)
            for sd, s in zip(candidates, scores):
                sd.score = float(s)
        else:
            # fallback: keep existing scores
            pass

        candidates.sort(key=lambda x: x.score, reverse=True)
        selected = candidates[: max(top_k, self.stage3_k)]

        # Build RetrievalResult list (reuse structure from MultiStageRetriever)
        results: List[RetrievalResult] = []
        for rank, sd in enumerate(selected, 1):
            results.append(
                RetrievalResult(
                    document=sd.doc,
                    final_score=sd.score,
                    stage1_score=sd.score if sd.source.startswith("bm25") else 0.0,
                    stage2_score=sd.score if sd.source.endswith("faiss") else 0.0,
                    stage3_score=sd.score,
                    stage1_rank=rank,
                    stage2_rank=rank,
                    stage3_rank=rank,
                    retrieval_metadata={
                        "source": sd.source,
                        "expanded_query": expanded_query,
                        "bm25_k": self.bm25_k,
                        "embed_k": self.embed_k,
                        "cross_encoder": ce if isinstance(ce, str) else self.cross_encoder_name,
                        "elapsed_ms": (time.time() - t0) * 1000.0,
                    },
                )
            )

        return results[:top_k]

