"""
LangChain Wrappers for Existing Components

These wrappers expose existing functionality through LangChain interfaces
WITHOUT modifying any internal logic, parameters, or behavior.
"""

from typing import List, Optional, Any, Dict
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
import numpy as np


class MedicalEmbeddingWrapper(Embeddings):
    """
    LangChain wrapper for existing EmbeddingModel.
    
    Passes all calls directly to the underlying model without modification.
    """
    
    def __init__(self, embedding_model):
        """
        Args:
            embedding_model: Your existing EmbeddingModel instance
        """
        self.model = embedding_model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents using existing embed_batch method.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (identical to original behavior)
        """
        # Call existing method with identical parameters
        embeddings = self.model.embed_batch(texts, batch_size=None, show_progress=False)
        # Convert numpy arrays to lists for LangChain compatibility
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query using existing embed method.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector (identical to original behavior)
        """
        # Call existing method
        embedding = self.model.embed(text)
        # Convert numpy array to list for LangChain compatibility
        return embedding.tolist()


class MedicalVectorStoreWrapper(VectorStore):
    """
    LangChain wrapper for existing FAISSVectorStore.
    
    Provides LangChain VectorStore interface while using your exact FAISS implementation.
    """
    
    def __init__(self, faiss_store, embedding_model):
        """
        Args:
            faiss_store: Your existing FAISSVectorStore instance
            embedding_model: Your existing EmbeddingModel instance
        """
        self.store = faiss_store
        self.embedding_model = embedding_model
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Not implemented - your store is pre-built."""
        raise NotImplementedError("Vector store is pre-built, use existing add_documents")
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        """
        Search using existing FAISS search method.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of LangChain Documents (wrapping existing results)
        """
        # Call existing search method with identical parameters
        results = self.store.search(query, k=k)
        
        # Convert to LangChain Documents without modifying content
        documents = []
        for doc, score in results:
            documents.append(
                Document(
                    page_content=doc.content,
                    metadata={
                        "source": doc.metadata.get("source", ""),
                        "guideline_id": doc.metadata.get("guideline_id", ""),
                        "score": float(score)
                    }
                )
            )
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[tuple]:
        """
        Search with scores using existing FAISS search method.
        
        Returns:
            List of (Document, score) tuples
        """
        results = self.store.search(query, k=k)
        
        documents_with_scores = []
        for doc, score in results:
            documents_with_scores.append((
                Document(
                    page_content=doc.content,
                    metadata={
                        "source": doc.metadata.get("source", ""),
                        "guideline_id": doc.metadata.get("guideline_id", ""),
                    }
                ),
                float(score)
            ))
        return documents_with_scores
    
    @classmethod
    def from_texts(cls, texts: List[str], embedding, metadatas=None, **kwargs):
        """Not implemented - use existing store initialization."""
        raise NotImplementedError("Use existing FAISSVectorStore initialization")


class MedicalRetrieverWrapper(BaseRetriever):
    """
    LangChain wrapper for existing MultiStageRetriever.
    
    Executes your exact multi-stage retrieval logic without modification.
    """
    
    retriever: Any = Field(description="The underlying MultiStageRetriever instance")
    
    def __init__(self, multi_stage_retriever, **kwargs):
        """
        Args:
            multi_stage_retriever: Your existing MultiStageRetriever instance
        """
        super().__init__(retriever=multi_stage_retriever, **kwargs)
    
    def _get_relevant_documents(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Execute existing multi-stage retrieval.
        
        Args:
            query: Search query
            
        Returns:
            List of LangChain Documents (wrapping existing results)
        """
        # Call existing retrieve method with identical logic
        # Returns List[RetrievalResult] directly
        retrieval_results = self.retriever.retrieve(query)
        
        # Convert to LangChain Documents without modifying content
        documents = []
        for result in retrieval_results:
            documents.append(
                Document(
                    page_content=result.document.content,
                    metadata={
                        "source": result.document.metadata.get("source", ""),
                        "guideline_id": result.document.metadata.get("guideline_id", ""),
                        "final_score": float(result.final_score),
                        "stage1_score": float(result.stage1_score),
                        "stage2_score": float(result.stage2_score),
                        "stage3_score": float(result.stage3_score)
                    }
                )
            )
        return documents
    
    async def _aget_relevant_documents(
        self,
        query: str,
        **kwargs: Any
    ) -> List[Document]:
        """Async version - calls sync method."""
        return self._get_relevant_documents(query, **kwargs)


class MedicalRerankerWrapper:
    """
    Wrapper for existing reranking logic.
    
    Applies your exact reranking/reordering without changing scores or logic.
    """
    
    def __init__(self, guideline_reranker=None, context_pruner=None):
        """
        Args:
            guideline_reranker: Your existing GuidelineReranker (optional)
            context_pruner: Your existing ContextPruner (optional)
        """
        self.guideline_reranker = guideline_reranker
        self.context_pruner = context_pruner
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Apply existing reranking logic.
        
        Args:
            query: Original query
            documents: Retrieved documents
            top_k: Number of documents to return
            
        Returns:
            Reranked documents (using existing logic)
        """
        # If no reranker, return as-is
        if not self.guideline_reranker and not self.context_pruner:
            return documents[:top_k] if top_k else documents
        
        # Convert LangChain Documents back to your format
        retrieved_docs = []
        for doc in documents:
            # Preserve exact structure and scores
            retrieved_docs.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': doc.metadata.get('score', 0.0)
            })
        
        # Apply existing reranker if available
        if self.guideline_reranker:
            reranked = self.guideline_reranker.rerank(query, retrieved_docs)
        else:
            reranked = retrieved_docs
        
        # Apply existing context pruner if available
        if self.context_pruner:
            pruned = self.context_pruner.prune(query, reranked)
        else:
            pruned = reranked
        
        # Convert back to LangChain Documents
        result_documents = []
        for doc_data in pruned[:top_k] if top_k else pruned:
            result_documents.append(
                Document(
                    page_content=doc_data['content'],
                    metadata=doc_data['metadata']
                )
            )
        
        return result_documents


class MedicalLLMWrapper(LLM):
    """
    LangChain wrapper for existing Ollama LLM.
    
    Routes all generation requests to your exact LLM with identical parameters.
    """
    
    reasoning_engine: Any = Field(description="The underlying MedicalReasoningEngine instance")
    
    def __init__(self, reasoning_engine, **kwargs):
        """
        Args:
            reasoning_engine: Your existing MedicalReasoningEngine instance
        """
        super().__init__(reasoning_engine=reasoning_engine, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return identifier."""
        return "medical_ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate answer using existing reasoning engine.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences (not used in existing implementation)
            
        Returns:
            Generated text (identical to original behavior)
        """
        # Call existing generate method with identical parameters
        response = self.reasoning_engine.llm_client.generate(
            prompt=prompt,
            max_tokens=self.reasoning_engine.llm_client.max_tokens,
            temperature=self.reasoning_engine.llm_client.temperature
        )
        return response
    
    def reason_and_select(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        retrieved_contexts: List[Any],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute existing reason_and_select_answer logic.
        
        Args:
            question: Medical question
            case_description: Patient case description
            options: Answer options as dict {"A": "text", ...}
            retrieved_contexts: List of retrieved documents
            
        Returns:
            Answer selection with reasoning (identical to original)
        """
        # Call existing method with all parameters preserved
        result = self.reasoning_engine.reason_and_select_answer(
            question=question,
            case_description=case_description,
            options=options,
            retrieved_contexts=retrieved_contexts,
            correct_answer=None
        )
        
        return {
            'selected_answer': result.selected_answer,
            'confidence': result.confidence_score,
            'reasoning': result.reasoning_steps,
            'answer_scores': result.answer_scores if hasattr(result, 'answer_scores') else {}
        }


def create_langchain_components(
    embedding_model,
    faiss_store,
    multi_stage_retriever,
    reasoning_engine,
    guideline_reranker=None,
    context_pruner=None
):
    """
    Create LangChain wrappers for all existing components.
    
    Args:
        embedding_model: Your existing EmbeddingModel
        faiss_store: Your existing FAISSVectorStore
        multi_stage_retriever: Your existing MultiStageRetriever
        reasoning_engine: Your existing MedicalReasoningEngine
        guideline_reranker: Your existing GuidelineReranker (optional)
        context_pruner: Your existing ContextPruner (optional)
        
    Returns:
        Dictionary of LangChain-wrapped components
    """
    return {
        'embeddings': MedicalEmbeddingWrapper(embedding_model),
        'vectorstore': MedicalVectorStoreWrapper(faiss_store, embedding_model),
        'retriever': MedicalRetrieverWrapper(multi_stage_retriever),
        'reranker': MedicalRerankerWrapper(guideline_reranker, context_pruner),
        'llm': MedicalLLMWrapper(reasoning_engine)
    }
