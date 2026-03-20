# Retrieval Layer Usage Examples

## Quick Start

### 1. Initialize Components

```python
from src.retrieval import (
    HyDEExpander,
    HierarchicalChunker,
    HybridSearcher,
    CrossEncoderReranker,
)
from src.llm.router import ModelRouter

# Initialize
router = ModelRouter()
expander = HyDEExpander(model_router=router)
chunker = HierarchicalChunker()
reranker = CrossEncoderReranker()
```

### 2. Chunk Documents

```python
# Single document
chunks, stats = await chunker.chunk_document(
    document_id="arxiv_2023_001",
    title="Attention Is All You Need",
    content="""Transformer architecture... [full paper content]"""
)

print(f"Created {len(chunks)} chunks")
print(f"Total tokens: {stats.total_tokens}")
print(f"Embedding cost: {stats.embedding_cost_tokens} tokens")

# Multiple documents in parallel
documents = [
    {"document_id": "doc1", "title": "Paper 1", "content": "..."},
    {"document_id": "doc2", "title": "Paper 2", "content": "..."},
]

all_chunks = {}
doc_chunks = await chunker.batch_chunk_documents(documents)
for doc_id, (chunks, stats) in doc_chunks.items():
    all_chunks[doc_id] = chunks
```

### 3. Build Search Index

```python
# Flatten all chunks from all documents
indexed_chunks = []
for doc_chunks in all_chunks.values():
    indexed_chunks.extend(doc_chunks)

# Create searcher and add chunks
searcher = HybridSearcher()
searcher.add_chunks(indexed_chunks)

print(f"Indexed {len(indexed_chunks)} chunks")
stats = searcher.get_stats()  # Get index statistics
```

### 4. Expand Query with HyDE

```python
query = "What are the key innovations in the Transformer architecture?"

hyde_result = await expander.expand(query)

print(f"Original query: {query}")
print(f"Hypotheticals:")
for i, hyp in enumerate(hyde_result.hypotheticals, 1):
    print(f"  {i}. {hyp}")

# hyde_result.averaged_embedding → [768 floats]
# Use this for search
```

### 5. Hybrid Search

```python
# Search using averaged HyDE embedding
results = searcher.search(
    query=query,
    query_embedding=hyde_result.averaged_embedding,
    top_k=20
)

print(f"Found {len(results)} results")
for result in results[:5]:
    print(f"  Rank {result.rank}: {result.chunk.chunk_id}")
    print(f"    Score: {result.score:.3f}")
    print(f"    BM25 rank: {result.bm25_rank}, Vector rank: {result.vector_rank}")
```

### 6. Rerank Results

```python
# Rerank to top-5
final_results = reranker.rerank(
    query=query,
    results=results,
    top_k=5
)

print(f"\nFinal reranked results:")
for result in final_results:
    print(f"  Rank {result.rank}: {result.chunk.chunk_id}")
    print(f"    RRF score: {result.score:.3f}")
    print(f"    Reranker score: {result.reranker_score:.3f}")
```

## Complete Pipeline Example

```python
import asyncio
from src.retrieval import (
    HyDEExpander,
    HierarchicalChunker,
    HybridSearcher,
    CrossEncoderReranker,
)
from src.llm.router import ModelRouter

async def retrieve_documents(
    query: str,
    document_chunks: list,
) -> list:
    """
    Complete retrieval pipeline: query → top-5 chunks.
    """
    
    # Initialize components
    router = ModelRouter()
    expander = HyDEExpander(model_router=router)
    searcher = HybridSearcher(chunks=document_chunks)
    reranker = CrossEncoderReranker()
    
    # Step 1: HyDE expansion
    print(f"Expanding query: '{query}'")
    hyde_result = await expander.expand(query)
    print(f"Generated {len(hyde_result.hypotheticals)} hypothetical documents")
    
    # Step 2: Hybrid search
    print("Performing hybrid search...")
    results = searcher.search(
        query=query,
        query_embedding=hyde_result.averaged_embedding,
        top_k=20
    )
    print(f"Retrieved {len(results)} results")
    
    # Step 3: Cross-encoder reranking
    print("Reranking results...")
    final_results = reranker.rerank(
        query=query,
        results=results,
        top_k=5
    )
    print(f"Final top-{len(final_results)} results:")
    
    for i, result in enumerate(final_results, 1):
        print(f"\n{i}. {result.chunk.chunk_id}")
        print(f"   Text: {result.chunk.text[:100]}...")
        print(f"   Score: {result.score:.3f}")
        print(f"   Reranker: {result.reranker_score:.3f}")
    
    return final_results

# Run the pipeline
if __name__ == "__main__":
    # Assume document_chunks are already loaded
    results = asyncio.run(retrieve_documents(
        "What is machine learning?",
        document_chunks
    ))
```

## Advanced Examples

### Custom Configuration

```python
from src.retrieval import (
    HybridSearcher,
    HybridSearchConfig,
    CrossEncoderReranker,
    RerankerConfig,
)

# Customize hybrid search (prefer BM25)
hybrid_config = HybridSearchConfig(
    alpha=0.7,  # 70% BM25, 30% vector
    top_k=30,
    bm25_k1=2.0,  # Increase term frequency boost
    rrf_constant=80,
)
searcher = HybridSearcher(chunks=chunks, config=hybrid_config)

# Customize reranker (stricter)
reranker_config = RerankerConfig(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size=64,
    top_k=3,
    threshold=0.5,  # Only return scores >= 0.5
)
reranker = CrossEncoderReranker(config=reranker_config)
```

### Batch Chunking with Error Handling

```python
async def safe_chunk_documents(documents):
    """Chunk documents with error handling."""
    chunker = HierarchicalChunker()
    results = {}
    errors = {}
    
    try:
        chunked = await chunker.batch_chunk_documents(documents)
        for doc_id, (chunks, stats) in chunked.items():
            results[doc_id] = chunks
    except Exception as e:
        for doc in documents:
            doc_id = doc["document_id"]
            try:
                chunks, stats = await chunker.chunk_document(
                    doc_id, doc["title"], doc["content"]
                )
                results[doc_id] = chunks
            except Exception as chunk_error:
                errors[doc_id] = str(chunk_error)
    
    return results, errors
```

### Incremental Indexing

```python
# Start with empty searcher
searcher = HybridSearcher()

# Add chunks incrementally
for document in document_stream:
    chunks, _ = await chunker.chunk_document(
        document["id"],
        document["title"],
        document["content"],
    )
    searcher.add_chunks(chunks)
    print(f"Added {len(chunks)} chunks from {document['id']}")

# Query at any time
results = searcher.search(query, query_embedding, top_k=20)
```

### Async Reranking

```python
# Use async wrapper to avoid blocking
final_results = await reranker.rerank_async(
    query="...",
    results=results,
    top_k=5
)
# Non-blocking: runs in thread pool
```

### Getting Statistics

```python
from src.retrieval import HierarchicalChunker

chunker = HierarchicalChunker()

# After chunking
chunks, stats = await chunker.chunk_document(...)

print(f"Document statistics:")
print(f"  Total tokens: {stats.total_tokens}")
print(f"  Chunk tokens: {stats.chunk_tokens}")
print(f"  Embedding cost: {stats.embedding_cost_tokens}")

# Detailed chunk stats
chunk_stats = chunker.get_stats(chunks)
print(f"\nChunk details:")
print(f"  Total chunks: {chunk_stats['total_chunks']}")
print(f"  Parent chunks: {chunk_stats['parent_chunks']}")
print(f"  Child chunks: {chunk_stats['child_chunks']}")
print(f"  Average tokens per chunk: {chunk_stats['avg_chunk_tokens']}")
print(f"  Embedded chunks: {chunk_stats['embedded_chunks']}")
```

### Memory-Efficient Large-Scale Search

```python
# For very large document collections:
# 1. Process documents in batches
# 2. Create separate searchers per batch
# 3. Combine results with deduplication

async def search_large_collection(
    query: str,
    document_batches: list[list],
):
    """Search across large document collection."""
    
    expander = HyDEExpander(model_router=router)
    chunker = HierarchicalChunker()
    reranker = CrossEncoderReranker()
    
    # Expand query once
    hyde_result = await expander.expand(query)
    
    all_results = []
    
    # Process each batch
    for batch_idx, batch_docs in enumerate(document_batches):
        print(f"Processing batch {batch_idx + 1}")
        
        # Chunk documents
        doc_chunks = await chunker.batch_chunk_documents(batch_docs)
        chunks = []
        for doc_id, (doc_chunks_list, _) in doc_chunks.items():
            chunks.extend(doc_chunks_list)
        
        # Search batch
        searcher = HybridSearcher(chunks=chunks)
        batch_results = searcher.search(
            query=query,
            query_embedding=hyde_result.averaged_embedding,
            top_k=20
        )
        all_results.extend(batch_results)
    
    # Sort all results and rerank top-20
    all_results.sort(key=lambda x: x.score, reverse=True)
    final = reranker.rerank(query, all_results[:20], top_k=5)
    
    return final
```

## Integration with LanceDB (Future)

```python
# Once LanceDB integration is added:
import lancedb

async def search_with_lancedb(
    query: str,
    db_path: str = "./data/research_memory.lancedb",
):
    """Search using persistent vector store."""
    
    # Connect to LanceDB
    db = lancedb.connect(db_path)
    table = db.open_table("research_chunks")
    
    # Expand query
    router = ModelRouter()
    expander = HyDEExpander(model_router=router)
    hyde_result = await expander.expand(query)
    
    # Vector search in LanceDB
    db_results = table.search(
        hyde_result.averaged_embedding
    ).limit(20).to_list()
    
    # Convert to SearchResult objects
    chunks = [row["chunk"] for row in db_results]
    results = [SearchResult(chunk=c, score=s) for c, s in zip(chunks, ...)]
    
    # Rerank
    reranker = CrossEncoderReranker()
    final = reranker.rerank(query, results, top_k=5)
    
    return final
```

## Logging & Debugging

```python
import structlog

# Enable debug logging
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ],
)

# Now all operations are logged as JSON events
# Search in logs:
#   "event": "hyde_expand_start"
#   "event": "bm25_search_done"
#   "event": "rerank_complete"

# Example: check retrieval latency
hyde_result = await expander.expand(query)
print(f"HyDE generation latency: {hyde_result.generation_latency_ms}ms")
```

## Error Handling Best Practices

```python
async def robust_retrieve(query, chunks):
    """Retrieve with graceful error handling."""
    
    router = ModelRouter()
    
    try:
        # HyDE expansion (graceful fallback)
        try:
            expander = HyDEExpander(model_router=router)
            hyde_result = await expander.expand(query)
            embedding = hyde_result.averaged_embedding
        except Exception as e:
            logger.warning("HyDE failed, using basic embedding", error=str(e))
            # Fallback: use pre-computed query embedding
            embedding = await get_basic_embedding(query)
        
        # Hybrid search
        searcher = HybridSearcher(chunks=chunks)
        results = searcher.search(query, embedding, top_k=20)
        
        # Reranking (optional, skip on error)
        try:
            reranker = CrossEncoderReranker()
            final = reranker.rerank(query, results, top_k=5)
        except Exception as e:
            logger.warning("Reranking failed, using hybrid results", error=str(e))
            final = results[:5]
        
        return final
    
    except Exception as e:
        logger.error("Retrieval failed completely", error=str(e))
        raise
```

## Testing

```python
# Run tests
pytest tests/test_retrieval.py -v

# Test specific component
pytest tests/test_retrieval.py::TestHybridSearcher -v

# Run with coverage
pytest tests/test_retrieval.py --cov=src.retrieval
```
