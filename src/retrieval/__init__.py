"""
Retrieval Module for Medical QA System

This module handles document retrieval using multiple strategies:
- Semantic search with FAISS vector database
- Keyword search with BM25
- Hybrid search combining both approaches
- Multi-stage retrieval with reranking

Components:
- document_processor: Loads and chunks medical guidelines
- embeddings: Converts text to vector representations
- faiss_store: Vector database for semantic search
- bm25_retriever: Keyword-based search
- hybrid_retriever: Combined retrieval strategy
- multi_stage_retriever: Three-stage retrieval with reranking
"""

from .document_processor import DocumentProcessor, Document
from .faiss_store import FAISSVectorStore
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .multi_stage_retriever import MultiStageRetriever, RetrievalResult

__all__ = [
    'DocumentProcessor', 
    'Document',
    'FAISSVectorStore',
    'BM25Retriever',
    'HybridRetriever',
    'MultiStageRetriever',
    'RetrievalResult'
]
