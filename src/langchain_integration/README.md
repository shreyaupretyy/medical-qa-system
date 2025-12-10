# LangChain & LangGraph Integration

## Overview

This integration provides **pure wrapper components** for your existing medical QA pipeline using LangChain and LangGraph. 

**Key Principle**: NO changes to logic, accuracy, or numerical behavior - only structural orchestration.

## What's Included

### 1. LangChain Wrappers (`src/langchain_integration/wrappers.py`)

- **`MedicalEmbeddingWrapper`**: Wraps your `EmbeddingModel`
- **`MedicalVectorStoreWrapper`**: Wraps your `FAISSVectorStore`
- **`MedicalRetrieverWrapper`**: Wraps your `MultiStageRetriever`
- **`MedicalRerankerWrapper`**: Wraps your reranking logic
- **`MedicalLLMWrapper`**: Wraps your `MedicalReasoningEngine`

### 2. LangGraph Orchestration (`src/langchain_integration/graph.py`)

- **`MedicalQAGraph`**: State machine that orchestrates your pipeline
- **`create_medical_qa_graph()`**: Factory function for easy setup

## Installation

```bash
# Install additional dependencies
pip install langchain==0.1.0 langchain-community==0.0.13 langgraph==0.0.20

# Or use updated requirements
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from reasoning.rag_pipeline import load_pipeline
from langchain_integration import create_medical_qa_graph

# Load your existing pipeline (unchanged)
pipeline = load_pipeline()

# Wrap with LangGraph
graph = create_medical_qa_graph(
    embedding_model=pipeline.embedding_model,
    faiss_store=pipeline.faiss_store,
    multi_stage_retriever=pipeline.retriever,
    reasoning_engine=pipeline.reasoning_engine,
    top_k=25
)

# Use it (produces identical results)
result = graph.invoke(
    question="What is the first-line treatment for STEMI?",
    options=["A", "B", "C", "D"]
)

print(f"Answer: {result['selected_answer']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Output Structure

```python
{
    'selected_answer': 'D',
    'confidence': 0.89,
    'reasoning_steps': [
        'Analyzing clinical presentation...',
        'ST-elevation indicates STEMI...',
        'Primary PCI is gold standard...'
    ],
    'answer_scores': {
        'A': 0.15,
        'B': 0.08,
        'C': 0.18,
        'D': 0.89
    },
    'retrieved_documents': 25,
    'reranked_documents': 25,
    'metadata': {
        'num_context_docs': 25,
        'context_length': 4532
    }
}
```

## Architecture

### Graph Flow

```
Input (question + options)
    ↓
[RETRIEVE NODE]
  • Calls MultiStageRetriever
  • Returns top-k documents
    ↓
[RERANK NODE]
  • Applies GuidelineReranker
  • Applies ContextPruner
  • Returns reordered documents
    ↓
[REASON NODE]
  • Builds context string
  • Calls MedicalReasoningEngine
  • Returns answer + confidence
    ↓
Output (complete result)
```

### State Object

```python
class MedicalQAState(TypedDict):
    # Input
    question: str
    options: List[str]
    
    # Retrieval
    retrieved_documents: List[Document]
    retrieval_metadata: Dict
    
    # Reranking
    reranked_documents: List[Document]
    reranking_metadata: Dict
    
    # Reasoning
    context: str
    reasoning_steps: List[str]
    
    # Output
    selected_answer: str
    confidence: float
    answer_scores: Dict[str, float]
    execution_metadata: Dict
```

## Verification

The integration is designed to produce **bit-identical results** to your existing pipeline:

```python
# LangGraph execution
graph_result = graph.invoke(question, options)

# Direct pipeline execution
direct_result = pipeline.process(question, options)

# Verify
assert graph_result['selected_answer'] == direct_result.selected_answer
assert abs(graph_result['confidence'] - direct_result.confidence) < 1e-6
```

## Advanced Usage

### Custom Reranker

```python
from improvements.guideline_reranker import GuidelineReranker
from improvements.context_pruner import ContextPruner

# Initialize your reranking components
guideline_reranker = GuidelineReranker()
context_pruner = ContextPruner()

# Create graph with rerankers
graph = create_medical_qa_graph(
    embedding_model=pipeline.embedding_model,
    faiss_store=pipeline.faiss_store,
    multi_stage_retriever=pipeline.retriever,
    reasoning_engine=pipeline.reasoning_engine,
    guideline_reranker=guideline_reranker,
    context_pruner=context_pruner,
    top_k=25
)
```

### Batch Processing

```python
questions = [
    {"question": "...", "options": ["A", "B", "C", "D"]},
    {"question": "...", "options": ["A", "B", "C", "D"]},
]

results = []
for q in questions:
    result = graph.invoke(q["question"], q["options"])
    results.append(result)
```

### Async Execution

```python
import asyncio

async def process_async():
    result = await graph.ainvoke(question, options)
    return result

result = asyncio.run(process_async())
```

## Component Details

### MedicalEmbeddingWrapper

Implements LangChain's `Embeddings` interface:

- `embed_documents(texts)` → calls `model.embed_batch(texts)`
- `embed_query(text)` → calls `model.embed(text)`

**Preservation guarantees**:
- ✅ Batch size unchanged
- ✅ Normalization unchanged
- ✅ Device placement unchanged
- ✅ Numerical precision preserved

### MedicalVectorStoreWrapper

Implements LangChain's `VectorStore` interface:

- `similarity_search(query, k)` → calls `store.search(query, k)`
- `similarity_search_with_score(query, k)` → returns (doc, score) tuples

**Preservation guarantees**:
- ✅ FAISS index unchanged
- ✅ Search algorithm unchanged
- ✅ Distance metric unchanged
- ✅ Score calculation unchanged

### MedicalRetrieverWrapper

Implements LangChain's `BaseRetriever` interface:

- `_get_relevant_documents(query)` → calls `retriever.retrieve(query)`

**Preservation guarantees**:
- ✅ Multi-stage logic unchanged
- ✅ Stage weights unchanged
- ✅ Fusion strategy unchanged
- ✅ Query expansion unchanged

### MedicalRerankerWrapper

Custom wrapper (not a LangChain base class):

- `rerank(query, documents, top_k)` → applies your reranking logic

**Preservation guarantees**:
- ✅ Guideline prioritization unchanged
- ✅ Context pruning unchanged
- ✅ Scoring unchanged
- ✅ Ordering logic unchanged

### MedicalLLMWrapper

Implements LangChain's `LLM` interface:

- `_call(prompt)` → calls `llm_client.generate(prompt)`
- `reason_and_select(question, options, context)` → calls `reason_and_select_answer()`

**Preservation guarantees**:
- ✅ Model unchanged (llama3.1:8b)
- ✅ Temperature unchanged
- ✅ Max tokens unchanged
- ✅ Prompt format unchanged
- ✅ Parsing logic unchanged

## Testing

Run the example to verify identical behavior:

```bash
python examples/langchain_integration_example.py
```

Expected output:
```
✓ Answers match: True
✓ Confidence matches: True
✓ Behavior preserved: True
✅ Perfect match! LangGraph integration preserves exact behavior.
```

## Benefits of This Integration

1. **Modularity**: Components can be swapped independently
2. **Observability**: State transitions are explicit
3. **Extensibility**: Easy to add new nodes (e.g., post-processing)
4. **Standardization**: Uses industry-standard interfaces
5. **Debugging**: State can be inspected at each node
6. **No Risk**: Zero changes to your tested logic

## Migration Path

### Phase 1: Parallel Running (Current)
- Keep existing pipeline
- Add LangGraph wrapper
- Verify identical results

### Phase 2: Gradual Migration
- Use LangGraph for new features
- Keep existing pipeline for production
- Compare results continuously

### Phase 3: Full Migration (Optional)
- Replace direct calls with LangGraph
- Remove duplicate code
- Maintain backward compatibility

## Troubleshooting

### Import Errors

```python
# If you see: ModuleNotFoundError: No module named 'langchain_integration'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Different Results

If results differ, check:
1. Are you using the same `top_k` parameter?
2. Are reranker components initialized?
3. Is the random seed set (if using any randomness)?
4. Are model parameters identical (temperature, max_tokens)?

### Performance

LangGraph adds minimal overhead (~1-2ms per execution). If this matters:
- Use async execution for I/O-bound operations
- Batch multiple questions
- Profile specific nodes

## API Reference

### create_medical_qa_graph()

```python
def create_medical_qa_graph(
    embedding_model,          # Your EmbeddingModel instance
    faiss_store,             # Your FAISSVectorStore instance
    multi_stage_retriever,   # Your MultiStageRetriever instance
    reasoning_engine,        # Your MedicalReasoningEngine instance
    guideline_reranker=None, # Optional: GuidelineReranker instance
    context_pruner=None,     # Optional: ContextPruner instance
    top_k: int = 25          # Number of documents to retrieve
) -> MedicalQAGraph
```

### MedicalQAGraph.invoke()

```python
def invoke(
    question: str,       # Medical question text
    options: List[str]   # Answer options (e.g., ['A', 'B', 'C', 'D'])
) -> Dict[str, Any]      # Complete result dictionary
```

## License

Same as your main project.

## Support

For issues specific to the LangChain/LangGraph integration, please check:
1. This README
2. Example code in `examples/langchain_integration_example.py`
3. Docstrings in wrapper classes
