"""
Plug-and-Play Example: Using LangChain/LangGraph with Your Existing Pipeline

This demonstrates how to use the LangChain/LangGraph wrappers without changing
any existing code or behavior.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.rag_pipeline import load_pipeline
from src.langchain_integration import create_medical_qa_graph


def main():
    """
    Example: Run your pipeline through LangGraph orchestration.
    
    Results are IDENTICAL to calling pipeline.process() directly.
    """
    
    # Load your existing pipeline (unchanged)
    print("Loading existing pipeline...")
    pipeline = load_pipeline()
    
    # Wrap it with LangGraph (no behavioral changes)
    print("Creating LangGraph wrapper...")
    graph = create_medical_qa_graph(
        embedding_model=pipeline.embedding_model,
        faiss_store=pipeline.faiss_store,
        multi_stage_retriever=pipeline.retriever,
        reasoning_engine=pipeline.reasoning_engine,
        top_k=25  # Preserves your existing default
    )
    
    # Example question
    question = """
    A 65-year-old man presents to the emergency department with severe chest pain
    radiating to his left arm. ECG shows ST-segment elevation in leads V2-V4.
    What is the most appropriate immediate management?
    """
    
    options_list = ["A", "B", "C", "D"]
    options_dict = {
        "A": "Aspirin 300mg PO",
        "B": "Morphine 5mg IV",
        "C": "Clopidogrel 600mg PO",
        "D": "Primary PCI"
    }
    
    print("\n" + "="*80)
    print("LANGGRAPH EXECUTION")
    print("="*80)
    
    # Execute through LangGraph (identical results to direct pipeline call)
    result = graph.invoke(
        question=question,
        options=options_dict
    )
    
    print(f"\nQuestion: {question[:100]}...")
    print(f"\nSelected Answer: {result['selected_answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nReasoning Steps:")
    for i, step in enumerate(result['reasoning_steps'], 1):
        print(f"  {i}. {step}")
    print(f"\nAnswer Scores:")
    for option, score in result['answer_scores'].items():
        print(f"  {option}: {score:.4f}")
    print(f"\nMetadata:")
    print(f"  Retrieved Documents: {result['retrieved_documents']}")
    print(f"  After Reranking: {result['reranked_documents']}")
    print(f"  Context Length: {result['metadata'].get('context_length', 0)} chars")
    
    print("\n" + "="*80)
    print("DIRECT PIPELINE EXECUTION (for comparison)")
    print("="*80)
    
    # Execute directly (should produce identical results)
    direct_result = pipeline.answer_question(
        question_id="example_1",
        case_description=question,
        question="What is the most appropriate immediate management?",
        options=options_dict
    )
    
    print(f"\nSelected Answer: {direct_result.selected_answer}")
    print(f"Confidence: {direct_result.confidence_score:.2%}")
    
    # Verify identical results
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    answers_match = result['selected_answer'] == direct_result.selected_answer
    confidence_match = abs(result['confidence'] - direct_result.confidence_score) < 0.01
    
    print(f"✓ Answers match: {answers_match}")
    print(f"✓ Confidence matches: {confidence_match}")
    print(f"✓ Behavior preserved: {answers_match and confidence_match}")
    
    if not (answers_match and confidence_match):
        print("\n⚠️  WARNING: Results differ! Check wrapper implementation.")
    else:
        print("\n✅ Perfect match! LangGraph integration preserves exact behavior.")


if __name__ == "__main__":
    main()
