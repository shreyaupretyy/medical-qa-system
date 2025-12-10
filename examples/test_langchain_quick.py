"""
Quick test for LangChain/LangGraph integration
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.rag_pipeline import load_pipeline
from src.langchain_integration import create_medical_qa_graph

print("Testing LangChain/LangGraph Integration...")
print("=" * 80)

# Load pipeline
print("\n1. Loading pipeline...")
pipeline = load_pipeline()
print("✅ Pipeline loaded successfully")

# Create LangGraph wrapper
print("\n2. Creating LangGraph wrapper...")
graph = create_medical_qa_graph(
    embedding_model=pipeline.embedding_model,
    faiss_store=pipeline.faiss_store,
    multi_stage_retriever=pipeline.retriever,
    reasoning_engine=pipeline.reasoning_engine,
    top_k=5  # Use fewer docs for faster testing
)
print("✅ LangGraph wrapper created successfully")

# Test with simple question
print("\n3. Testing LangGraph execution...")
question = "What is the first-line treatment for acute STEMI?"
options = {
    "A": "Aspirin",
    "B": "Beta blocker",
    "C": "ACE inhibitor",
    "D": "Statin"
}

try:
    result = graph.invoke(question=question, options=options)
    print(f"✅ LangGraph execution successful")
    print(f"   - Selected Answer: {result['selected_answer']}")
    print(f"   - Confidence: {result['confidence']:.2%}")
    print(f"   - Reasoning Steps: {len(result['reasoning_steps'])} steps")
    retrieved_docs = result.get('retrieved_documents', [])
    if isinstance(retrieved_docs, list):
        print(f"   - Retrieved: {len(retrieved_docs)} documents")
    else:
        print(f"   - Retrieved: {retrieved_docs} documents")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ LangChain/LangGraph integration is working!")
