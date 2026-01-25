# Testing Strategy for Gmail Takeout RAG

## Overview

This document outlines the testing strategy for the RAG system, including unit tests, integration tests, and RAG-specific evaluation metrics.

## Current State: Unit Tests

### What We Have (97% Coverage)

**Unit tests** test individual components in isolation with mocked dependencies:

```python
# Example: Testing chunker without real data
def test_chunk_email_size_limits(chunker, sample_email):
    chunks = chunker.chunk_email(sample_email)
    for chunk in chunks:
        assert chunk["token_count"] <= chunker.chunk_size
```

**Pros:**
- Fast execution (no API calls)
- Predictable (mocked responses)
- Great for testing edge cases
- Cheap (no API costs)

**Cons:**
- Don't test real API integration
- Don't verify end-to-end workflows
- Don't test real data quality

---

## Integration Tests

### What Are Integration Tests?

**Integration tests** verify that multiple components work together correctly with real dependencies (APIs, databases, file systems).

### Examples for Your RAG System

#### 1. End-to-End Pipeline Test

Tests the complete flow from HTML to indexed vectors:

```python
def test_complete_indexing_pipeline():
    """Test: HTML → Extract → Chunk → Embed → Index"""

    # Setup
    preprocessor = EmailPreprocessor()
    chunker = TokenChunker()
    rag = NewsletterRAG()

    # Real HTML file
    email_data = preprocessor.extract_from_html("test_data/sample_email.html")

    # Chunk it
    chunks = chunker.chunk_email(email_data)

    # Index in Qdrant (real call)
    for i, chunk in enumerate(chunks):
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk["text"]
        ).data[0].embedding

        qdrant_client.upsert(
            collection_name="test_collection",
            points=[PointStruct(id=i, vector=embedding, payload=chunk)]
        )

    # Verify retrieval works
    results = rag.retrieve_similar_chunks("test query", top_k=3)
    assert len(results) == 3
    assert all(r["score"] > 0 for r in results)
```

#### 2. Qdrant Integration Test

Tests real vector database operations:

```python
def test_qdrant_operations():
    """Test real Qdrant CRUD operations"""

    client = QdrantClient(host="localhost", port=6333)
    test_collection = "integration_test_collection"

    # Create collection
    client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    # Insert test vectors
    test_vectors = [
        PointStruct(
            id=i,
            vector=[0.1] * 1536,
            payload={"text": f"test {i}"}
        )
        for i in range(10)
    ]
    client.upsert(collection_name=test_collection, points=test_vectors)

    # Query
    results = client.query_points(
        collection_name=test_collection,
        query=[0.1] * 1536,
        limit=5
    )

    assert len(results.points) == 5

    # Cleanup
    client.delete_collection(test_collection)
```

#### 3. OpenAI API Integration Test

Tests real embedding and chat completions:

```python
@pytest.mark.integration
@pytest.mark.slow
def test_openai_embedding_api():
    """Test real OpenAI embedding calls"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Test embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="test email content"
    )

    assert len(response.data) == 1
    assert len(response.data[0].embedding) == 1536

    # Test chat completion
    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )

    assert len(chat_response.choices) > 0
    assert chat_response.choices[0].message.content
```

#### 4. Batch Upload Integration Test

Tests large-scale operations:

```python
def test_batch_upload_large_dataset():
    """Test batch upload with realistic data volume"""

    # Generate 1000 test chunks
    chunks = generate_test_chunks(count=1000)

    # Create embeddings in batches
    batch_size = 100
    points = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        embeddings = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        ).data

        for j, chunk in enumerate(batch):
            points.append(PointStruct(
                id=i + j,
                vector=embeddings[j].embedding,
                payload=chunk
            ))

    # Upload to Qdrant in batches
    qdrant_client.upsert(
        collection_name="test_batch",
        points=points
    )

    # Verify all uploaded
    collection_info = qdrant_client.get_collection("test_batch")
    assert collection_info.points_count == 1000
```

### When to Run Integration Tests

**Not on every commit** (they're slow and cost money):
- Run in CI/CD on PR merges
- Run before releases
- Run manually when testing API changes
- Use pytest markers:

```bash
# Run only fast tests (unit tests)
pytest -m "not integration and not slow"

# Run all tests including integration
pytest -m integration

# Run specific test file
pytest tests/integration/test_qdrant.py
```

### Integration Test Configuration

```python
# pytest.ini
[pytest]
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    slow: marks tests as slow (deselect with '-m "not slow"')
    requires_api: marks tests requiring API keys
```

---

## RAG Evaluation Metrics

### Why Evaluate RAG Systems?

RAG systems have two stages:
1. **Retrieval**: Finding relevant chunks
2. **Generation**: Creating answers from chunks

You need metrics for both.

### Retrieval Metrics

#### 1. Precision@K

**What it measures**: Of the K chunks retrieved, how many are actually relevant?

```python
def precision_at_k(retrieved_docs, relevant_docs, k):
    """
    Precision@K = (# relevant docs in top K) / K

    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        relevant_docs: Set of ground truth relevant document IDs
        k: Number of top results to consider

    Returns:
        float: Precision score between 0 and 1
    """
    top_k = retrieved_docs[:k]
    relevant_retrieved = len(set(top_k) & set(relevant_docs))
    return relevant_retrieved / k

# Example
retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = {"doc1", "doc3", "doc7"}
precision_3 = precision_at_k(retrieved, relevant, k=3)  # 2/3 = 0.67
```

#### 2. Recall@K

**What it measures**: Of all relevant chunks, what fraction did we retrieve in top K?

```python
def recall_at_k(retrieved_docs, relevant_docs, k):
    """
    Recall@K = (# relevant docs in top K) / (total # relevant docs)
    """
    top_k = retrieved_docs[:k]
    relevant_retrieved = len(set(top_k) & set(relevant_docs))
    return relevant_retrieved / len(relevant_docs)

# Example
recall_3 = recall_at_k(retrieved, relevant, k=3)  # 2/3 = 0.67
```

#### 3. Mean Reciprocal Rank (MRR)

**What it measures**: How high is the first relevant result?

```python
def mean_reciprocal_rank(queries_results):
    """
    MRR = average of (1 / rank of first relevant doc) across queries

    Higher is better. 1.0 = first result always relevant
    """
    reciprocal_ranks = []

    for retrieved, relevant in queries_results:
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)  # No relevant doc found

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# Example
queries = [
    (["doc1", "doc2", "doc3"], {"doc2"}),  # Rank 2 → 1/2
    (["doc5", "doc6", "doc7"], {"doc5"}),  # Rank 1 → 1/1
    (["doc8", "doc9", "doc10"], {"doc11"}), # Not found → 0
]
mrr = mean_reciprocal_rank(queries)  # (0.5 + 1.0 + 0.0) / 3 = 0.5
```

#### 4. Normalized Discounted Cumulative Gain (NDCG)

**What it measures**: Ranking quality (considers both relevance and position)

```python
import numpy as np

def dcg_at_k(relevances, k):
    """
    DCG@K = sum(rel_i / log2(i + 1)) for i in 1..k
    """
    relevances = np.array(relevances)[:k]
    gains = 2**relevances - 1
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(gains / discounts)

def ndcg_at_k(relevances, k):
    """
    NDCG@K = DCG@K / IDCG@K

    Args:
        relevances: List of relevance scores (0-5 scale) for retrieved docs
        k: Number of results to consider

    Returns:
        float: NDCG score between 0 and 1
    """
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0

# Example: 5 retrieved docs with relevance scores
relevances = [3, 5, 0, 2, 4]  # 5=highly relevant, 0=not relevant
ndcg_5 = ndcg_at_k(relevances, k=5)
```

### Generation Metrics

#### 1. Faithfulness

**What it measures**: Is the answer grounded in the retrieved context?

```python
from openai import OpenAI

def faithfulness_score(question, context, answer):
    """
    Use LLM to judge if answer is faithful to context.

    Returns:
        float: Score between 0 and 1
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""You are evaluating if an AI answer is faithful to the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Is the answer faithful to the context? Does it make claims not supported by the context?
Rate faithfulness from 0-10 (0=completely unfaithful, 10=perfectly faithful).

Respond with just the number."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    score = int(response.choices[0].message.content.strip())
    return score / 10
```

#### 2. Answer Relevance

**What it measures**: Does the answer actually address the question?

```python
def answer_relevance_score(question, answer):
    """
    Use LLM to judge if answer addresses the question.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""Rate how well this answer addresses the question.

QUESTION: {question}

ANSWER: {answer}

Rate from 0-10 (0=completely irrelevant, 10=perfectly addresses question).
Respond with just the number."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    score = int(response.choices[0].message.content.strip())
    return score / 10
```

#### 3. Context Relevance

**What it measures**: Are the retrieved chunks relevant to the question?

```python
def context_relevance_score(question, chunks):
    """
    Average relevance of retrieved chunks to the question.
    """
    scores = []

    for chunk in chunks:
        prompt = f"""Rate how relevant this text is to the question.

QUESTION: {question}

TEXT: {chunk['text']}

Rate from 0-10 (0=not relevant, 10=highly relevant).
Respond with just the number."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        score = int(response.choices[0].message.content.strip())
        scores.append(score / 10)

    return sum(scores) / len(scores)
```

### End-to-End RAG Metrics

Combine retrieval + generation metrics:

```python
class RAGEvaluator:
    """Comprehensive RAG evaluation"""

    def __init__(self, rag_system):
        self.rag = rag_system
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def evaluate(self, test_cases):
        """
        Evaluate RAG on test cases.

        Args:
            test_cases: List of (question, expected_answer, relevant_doc_ids)

        Returns:
            dict: Metrics summary
        """
        metrics = {
            "precision_at_3": [],
            "recall_at_3": [],
            "mrr": [],
            "faithfulness": [],
            "answer_relevance": [],
            "context_relevance": []
        }

        for question, expected_answer, relevant_docs in test_cases:
            # Get RAG response
            result = self.rag.query(question, top_k=5)

            # Retrieved doc IDs
            retrieved = [chunk["chunk_id"] for chunk in result["sources"]]

            # Retrieval metrics
            metrics["precision_at_3"].append(
                precision_at_k(retrieved, relevant_docs, k=3)
            )
            metrics["recall_at_3"].append(
                recall_at_k(retrieved, relevant_docs, k=3)
            )

            # Generation metrics
            context = "\n".join([c["text"] for c in result["sources"]])

            metrics["faithfulness"].append(
                faithfulness_score(question, context, result["answer"])
            )
            metrics["answer_relevance"].append(
                answer_relevance_score(question, result["answer"])
            )
            metrics["context_relevance"].append(
                context_relevance_score(question, result["sources"])
            )

        # Compute averages
        return {
            metric: sum(values) / len(values)
            for metric, values in metrics.items()
        }
```

### Creating Test Cases

For your Gmail RAG, create a test set:

```python
# tests/fixtures/eval_test_cases.json
[
    {
        "question": "Did I get any Amazon order confirmations?",
        "expected_answer_keywords": ["amazon", "order", "confirmation"],
        "relevant_chunk_ids": [
            "emails/emails_to_html/1234_Amazon_Order_Confirmation.html_chunk_0",
            "emails/emails_to_html/1235_Amazon_Shipment.html_chunk_0"
        ]
    },
    {
        "question": "What newsletters did I get from Instagram?",
        "expected_answer_keywords": ["instagram", "newsletter"],
        "relevant_chunk_ids": [
            "emails/emails_to_html/5678_Instagram_Weekly.html_chunk_0"
        ]
    }
]
```

---

## Implementation Plan

### Phase 1: Basic Integration Tests (Week 1)

1. Create `tests/integration/` directory
2. Add Qdrant integration tests
3. Add OpenAI API tests (with rate limiting)
4. Add end-to-end pipeline test

### Phase 2: RAG Evaluation Framework (Week 2)

1. Create test case dataset (20-50 questions)
2. Implement retrieval metrics
3. Implement generation metrics
4. Create evaluation report generator

### Phase 3: Continuous Evaluation (Week 3)

1. Set up CI/CD integration
2. Benchmark different configurations
3. Track metrics over time
4. Create dashboard for metrics

---

## Best Practices

### 1. Separate Test Environments

```python
# Use different collections for testing
if os.getenv("ENV") == "test":
    COLLECTION_NAME = "newsletter_chunks_test"
else:
    COLLECTION_NAME = "newsletter_chunks"
```

### 2. Cost Management

```python
# Limit API calls in tests
@pytest.fixture(scope="session")
def cached_embeddings():
    """Cache embeddings to reduce API calls"""
    cache = {}

    def get_embedding(text):
        if text not in cache:
            cache[text] = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding
        return cache[text]

    return get_embedding
```

### 3. Test Data Management

```bash
tests/
├── unit/              # Fast, mocked tests
├── integration/       # Real API tests
├── fixtures/          # Test data
│   ├── sample_emails/
│   ├── eval_cases.json
│   └── ground_truth.json
└── conftest.py        # Shared fixtures
```

### 4. Monitoring in Production

```python
# Log metrics for each query
def query_with_metrics(question, top_k=5):
    start_time = time.time()

    result = rag.query(question, top_k)

    # Log metrics
    metrics = {
        "latency": time.time() - start_time,
        "num_chunks_retrieved": len(result["sources"]),
        "avg_similarity_score": np.mean([s["score"] for s in result["sources"]]),
        "answer_length": len(result["answer"])
    }

    logger.info(f"Query metrics: {metrics}")

    return result
```

---

## Resources

- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation framework
- [TruLens](https://www.trulens.org/) - LLM app evaluation
- [LangSmith](https://www.langchain.com/langsmith) - LangChain evaluation tools
- [Phoenix](https://github.com/Arize-ai/phoenix) - ML observability
