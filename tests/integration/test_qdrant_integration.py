"""
Integration tests for Qdrant vector database

Run with: pytest tests/integration/test_qdrant_integration.py -v
Mark as slow: pytest -m "not slow" (to skip these tests)
"""

import pytest
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


@pytest.fixture(scope="module")
def qdrant_client():
    """Connect to real Qdrant instance."""
    client = QdrantClient(host="localhost", port=6333)
    yield client


@pytest.fixture
def test_collection_name():
    """Generate unique test collection name."""
    return f"test_collection_{os.getpid()}"


@pytest.fixture
def cleanup_collection(qdrant_client, test_collection_name):
    """Cleanup test collection after test."""
    yield
    try:
        qdrant_client.delete_collection(test_collection_name)
    except Exception:
        pass  # Collection might not exist


@pytest.mark.integration
@pytest.mark.slow
def test_create_and_delete_collection(qdrant_client, test_collection_name, cleanup_collection):
    """Test creating and deleting a collection."""
    # Create collection
    qdrant_client.create_collection(
        collection_name=test_collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    # Verify it exists
    collection_info = qdrant_client.get_collection(test_collection_name)
    assert collection_info.status == "green"
    assert collection_info.points_count == 0

    # Delete collection
    qdrant_client.delete_collection(test_collection_name)


@pytest.mark.integration
@pytest.mark.slow
def test_insert_and_query_vectors(qdrant_client, test_collection_name, cleanup_collection):
    """Test inserting vectors and querying them."""
    # Create collection
    qdrant_client.create_collection(
        collection_name=test_collection_name,
        vectors_config=VectorParams(size=128, distance=Distance.COSINE)
    )

    # Insert test vectors
    test_vectors = [
        PointStruct(
            id=i,
            vector=[0.1 * i] * 128,
            payload={"text": f"test document {i}", "index": i}
        )
        for i in range(10)
    ]

    qdrant_client.upsert(
        collection_name=test_collection_name,
        points=test_vectors
    )

    # Verify count
    collection_info = qdrant_client.get_collection(test_collection_name)
    assert collection_info.points_count == 10

    # Query similar vectors
    query_vector = [0.1 * 5] * 128  # Should be closest to vector 5
    results = qdrant_client.query_points(
        collection_name=test_collection_name,
        query=query_vector,
        limit=3
    )

    assert len(results.points) == 3
    # First result should be closest
    assert results.points[0].payload["index"] == 5


@pytest.mark.integration
@pytest.mark.slow
def test_batch_upload(qdrant_client, test_collection_name, cleanup_collection):
    """Test batch uploading vectors (simulates large dataset)."""
    # Create collection
    qdrant_client.create_collection(
        collection_name=test_collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    # Create 1000 test vectors
    total_vectors = 1000
    batch_size = 100

    for batch_start in range(0, total_vectors, batch_size):
        batch_points = [
            PointStruct(
                id=i,
                vector=[0.01] * 1536,
                payload={"text": f"batch vector {i}"}
            )
            for i in range(batch_start, min(batch_start + batch_size, total_vectors))
        ]

        qdrant_client.upsert(
            collection_name=test_collection_name,
            points=batch_points
        )

    # Verify all uploaded
    collection_info = qdrant_client.get_collection(test_collection_name)
    assert collection_info.points_count == total_vectors


@pytest.mark.integration
@pytest.mark.slow
def test_filter_by_payload(qdrant_client, test_collection_name, cleanup_collection):
    """Test filtering search results by payload metadata."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # Create collection
    qdrant_client.create_collection(
        collection_name=test_collection_name,
        vectors_config=VectorParams(size=128, distance=Distance.COSINE)
    )

    # Insert vectors with different categories
    vectors = []
    for i in range(20):
        vectors.append(PointStruct(
            id=i,
            vector=[0.1] * 128,
            payload={
                "text": f"document {i}",
                "category": "A" if i < 10 else "B"
            }
        ))

    qdrant_client.upsert(collection_name=test_collection_name, points=vectors)

    # Query with filter for category A
    results = qdrant_client.query_points(
        collection_name=test_collection_name,
        query=[0.1] * 128,
        query_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="A"))]
        ),
        limit=15
    )

    # Should only get category A results
    assert len(results.points) == 10
    assert all(p.payload["category"] == "A" for p in results.points)
