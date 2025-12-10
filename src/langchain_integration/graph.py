"""
LangGraph State Machine for Medical QA Pipeline

Orchestrates existing components using LangGraph without modifying their behavior.
Every node executes your exact existing logic.
"""

from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langchain.schema import Document
import operator


class MedicalQAState(TypedDict):
    """
    State object passed between graph nodes.
    
    Preserves all information from your existing pipeline.
    """
    # Input
    question: str
    options: Dict[str, str]
    
    # Retrieval stage
    retrieved_documents: Annotated[List[Document], operator.add]
    retrieval_metadata: Dict[str, Any]
    
    # Reranking stage
    reranked_documents: List[Document]
    reranking_metadata: Dict[str, Any]
    
    # Reasoning stage
    context: str
    reasoning_steps: List[str]
    
    # Output
    selected_answer: str
    confidence: float
    answer_scores: Dict[str, float]
    
    # Metadata
    execution_metadata: Dict[str, Any]


class MedicalQAGraph:
    """
    LangGraph orchestration of your existing medical QA pipeline.
    
    Each node is a thin wrapper that calls your exact existing functions.
    NO changes to logic, parameters, or behavior.
    """
    
    def __init__(
        self,
        retriever,
        reranker,
        llm,
        top_k: int = 25
    ):
        """
        Args:
            retriever: MedicalRetrieverWrapper instance
            reranker: MedicalRerankerWrapper instance
            llm: MedicalLLMWrapper instance
            top_k: Number of documents to retrieve (preserves existing default)
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Construct LangGraph state machine.
        
        Flow:
            input → retrieve → rerank → reason → output
        """
        workflow = StateGraph(MedicalQAState)
        
        # Add nodes (each calls existing component)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("reason", self._reason_node)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "reason")
        workflow.add_edge("reason", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: MedicalQAState) -> Dict[str, Any]:
        """
        Execute existing retrieval logic.
        
        Calls your MedicalRetrieverWrapper which uses your exact MultiStageRetriever.
        """
        question = state["question"]
        
        # Call existing retriever (identical behavior)
        documents = self.retriever._get_relevant_documents(question)
        
        # Limit to top_k (preserving existing parameter)
        documents = documents[:self.top_k]
        
        return {
            "retrieved_documents": documents,
            "retrieval_metadata": {
                "num_retrieved": len(documents),
                "retriever_type": "multi_stage"
            }
        }
    
    def _rerank_node(self, state: MedicalQAState) -> Dict[str, Any]:
        """
        Execute existing reranking logic.
        
        Calls your MedicalRerankerWrapper which uses your exact reranking components.
        """
        question = state["question"]
        documents = state["retrieved_documents"]
        
        # Call existing reranker (identical behavior)
        reranked_documents = self.reranker.rerank(
            query=question,
            documents=documents,
            top_k=self.top_k
        )
        
        return {
            "reranked_documents": reranked_documents,
            "reranking_metadata": {
                "num_after_rerank": len(reranked_documents)
            }
        }
    
    def _reason_node(self, state: MedicalQAState) -> Dict[str, Any]:
        """
        Execute existing reasoning and answer selection logic.
        
        Calls your MedicalLLMWrapper which uses your exact reasoning engine.
        """
        question = state["question"]
        options_dict = state["options"]
        documents = state["reranked_documents"]
        
        # Convert LangChain Documents back to your Document format
        from src.retrieval.document_processor import Document as InternalDocument
        internal_docs = []
        for doc in documents:
            internal_docs.append(InternalDocument(
                content=doc.page_content,
                metadata=doc.metadata
            ))
        
        # Extract case description from question (for now, use full question)
        case_description = question
        
        # Call existing reasoning engine (identical behavior)
        result = self.llm.reason_and_select(
            question=question,
            case_description=case_description,
            options=options_dict,
            retrieved_contexts=internal_docs
        )
        
        # Build context string for metadata
        context = self._build_context(documents)
        
        return {
            "context": context,
            "reasoning_steps": result["reasoning"],
            "selected_answer": result["selected_answer"],
            "confidence": result["confidence"],
            "answer_scores": result["answer_scores"],
            "execution_metadata": {
                "num_context_docs": len(documents),
                "context_length": len(context)
            }
        }
    
    def _build_context(self, documents: List[Document]) -> str:
        """
        Build context string from documents.
        
        Uses identical format to your existing pipeline.
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            guideline_id = doc.metadata.get("guideline_id", "unknown")
            context_parts.append(
                f"[Guideline {guideline_id} - Context {i}]\n{doc.page_content}"
            )
        return "\n\n".join(context_parts)
    
    def invoke(
        self,
        question: str,
        options: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Execute full pipeline on a question.
        
        Args:
            question: Medical question
            options: Dict of answer options (e.g., {'A': 'text', 'B': 'text', ...})
            
        Returns:
            Complete result dictionary with answer, confidence, reasoning
        """
        # Initialize state
        initial_state = {
            "question": question,
            "options": options,
            "retrieved_documents": [],
            "retrieval_metadata": {},
            "reranked_documents": [],
            "reranking_metadata": {},
            "context": "",
            "reasoning_steps": [],
            "selected_answer": "",
            "confidence": 0.0,
            "answer_scores": {},
            "execution_metadata": {}
        }
        
        # Execute graph (runs your existing pipeline)
        final_state = self.graph.invoke(initial_state)
        
        return {
            "selected_answer": final_state["selected_answer"],
            "confidence": final_state["confidence"],
            "reasoning_steps": final_state["reasoning_steps"],
            "answer_scores": final_state["answer_scores"],
            "retrieved_documents": len(final_state["retrieved_documents"]),
            "reranked_documents": len(final_state["reranked_documents"]),
            "metadata": final_state["execution_metadata"]
        }
    
    async def ainvoke(
        self,
        question: str,
        options: List[str]
    ) -> Dict[str, Any]:
        """Async version - calls sync method."""
        return self.invoke(question, options)


def create_medical_qa_graph(
    embedding_model,
    faiss_store,
    multi_stage_retriever,
    reasoning_engine,
    guideline_reranker=None,
    context_pruner=None,
    top_k: int = 25
) -> MedicalQAGraph:
    """
    Create complete LangGraph pipeline using your existing components.
    
    This is the main entry point for plug-and-play integration.
    
    Args:
        embedding_model: Your existing EmbeddingModel
        faiss_store: Your existing FAISSVectorStore
        multi_stage_retriever: Your existing MultiStageRetriever
        reasoning_engine: Your existing MedicalReasoningEngine
        guideline_reranker: Your existing GuidelineReranker (optional)
        context_pruner: Your existing ContextPruner (optional)
        top_k: Number of documents to retrieve (default: 25)
        
    Returns:
        MedicalQAGraph instance ready to use
        
    Example:
        >>> from langchain_integration import create_medical_qa_graph
        >>> from reasoning.rag_pipeline import load_pipeline
        >>> 
        >>> # Load your existing pipeline
        >>> pipeline = load_pipeline()
        >>> 
        >>> # Create LangGraph wrapper
        >>> graph = create_medical_qa_graph(
        ...     embedding_model=pipeline.embedding_model,
        ...     faiss_store=pipeline.faiss_store,
        ...     multi_stage_retriever=pipeline.retriever,
        ...     reasoning_engine=pipeline.reasoning_engine
        ... )
        >>> 
        >>> # Use it (produces identical results)
        >>> result = graph.invoke(
        ...     question="What is the first-line treatment for STEMI?",
        ...     options=["A", "B", "C", "D"]
        ... )
        >>> print(f"Answer: {result['selected_answer']}")
        >>> print(f"Confidence: {result['confidence']}")
    """
    from langchain_integration.wrappers import (
        MedicalRetrieverWrapper,
        MedicalRerankerWrapper,
        MedicalLLMWrapper
    )
    
    # Create wrappers
    retriever = MedicalRetrieverWrapper(multi_stage_retriever)
    reranker = MedicalRerankerWrapper(guideline_reranker, context_pruner)
    llm = MedicalLLMWrapper(reasoning_engine)
    
    # Create and return graph
    return MedicalQAGraph(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        top_k=top_k
    )
