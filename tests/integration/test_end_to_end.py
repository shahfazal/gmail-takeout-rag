"""
End-to-end integration tests for the complete RAG pipeline

These tests verify the entire flow works together.
"""

import pytest
import os
from pathlib import Path
from src.preprocessor import EmailPreprocessor
from src.chunker import TokenChunker
from src.rag import NewsletterRAG


@pytest.fixture
def sample_email_html(tmp_path):
    """Create a sample HTML email for testing."""
    html_content = """
    <html>
    <body>
        <div class="header">
            <div><span class="label">Subject:</span> Weekly Newsletter - AI Updates</div>
            <div><span class="label">From:</span> newsletter@example.com</div>
            <div><span class="label">Date:</span> Mon, 1 Jan 2024 12:00:00 +0000</div>
        </div>
        <div class="content">
            <h1>This Week in AI</h1>
            <p>OpenAI released GPT-4 Turbo with improved performance.</p>
            <p>Google announced Gemini Ultra, their most capable model.</p>
            <p>Meta open-sourced Llama 3 with 70B parameters.</p>
        </div>
    </body>
    </html>
    """

    email_file = tmp_path / "test_email.html"
    email_file.write_text(html_content)
    return str(email_file)


@pytest.mark.integration
@pytest.mark.slow
def test_preprocessing_and_chunking(sample_email_html):
    """Test: HTML → Extract → Chunk"""
    preprocessor = EmailPreprocessor()
    chunker = TokenChunker(chunk_size=100, overlap=20)

    # Extract
    email_data = preprocessor.extract_from_html(sample_email_html)

    assert email_data["subject"] == "Weekly Newsletter - AI Updates"
    assert "OpenAI" in email_data["body_text"]
    assert "Gemini" in email_data["body_text"]

    # Chunk
    chunks = chunker.chunk_email(email_data)

    assert len(chunks) > 0
    assert all(c["subject"] == email_data["subject"] for c in chunks)
    assert all(c["token_count"] <= 100 for c in chunks)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api
def test_complete_rag_pipeline_with_test_collection():
    """Test complete RAG pipeline with a temporary test collection."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from openai import OpenAI

    # Setup
    test_collection = f"test_e2e_{os.getpid()}"
    qdrant_client = QdrantClient(host="localhost", port=6333)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # Create test collection
        qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

        # Create test documents
        test_docs = [
            {"text": "I received an email about machine learning from OpenAI.", "id": 0},
            {"text": "Amazon sent me an order confirmation for my purchase.", "id": 1},
            {"text": "Instagram notified me about new followers.", "id": 2},
        ]

        # Create embeddings and index
        texts = [doc["text"] for doc in test_docs]
        embeddings_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        points = [
            PointStruct(
                id=doc["id"],
                vector=embeddings_response.data[i].embedding,
                payload={
                    "text": doc["text"],
                    "subject": f"Email {doc['id']}",
                    "from": "test@example.com",
                    "date": "2024-01-01"
                }
            )
            for i, doc in enumerate(test_docs)
        ]

        qdrant_client.upsert(collection_name=test_collection, points=points)

        # Initialize RAG
        rag = NewsletterRAG(collection_name=test_collection)

        # Test retrieval
        result = rag.query("Tell me about machine learning emails", top_k=2)

        # Assertions
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 2

        # Should retrieve the ML-related document
        sources_text = " ".join([s["text_preview"] for s in result["sources"]])
        assert "machine learning" in sources_text.lower() or "openai" in sources_text.lower()

    finally:
        # Cleanup
        try:
            qdrant_client.delete_collection(test_collection)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api
def test_retrieval_accuracy():
    """Test that retrieval returns relevant documents."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from openai import OpenAI

    test_collection = f"test_accuracy_{os.getpid()}"
    qdrant_client = QdrantClient(host="localhost", port=6333)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # Create collection
        qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

        # Create diverse test documents
        test_docs = [
            "Apple announced new iPhone features including improved camera and battery life.",
            "Google's latest AI research shows improvements in natural language processing.",
            "Amazon is offering discounts on electronics this week including laptops and tablets.",
            "Tesla released a software update for their autonomous driving system.",
            "Microsoft unveiled new productivity tools for remote work and collaboration."
        ]

        # Index documents
        embeddings_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=test_docs
        )

        points = [
            PointStruct(
                id=i,
                vector=embeddings_response.data[i].embedding,
                payload={"text": doc, "id": i}
            )
            for i, doc in enumerate(test_docs)
        ]

        qdrant_client.upsert(collection_name=test_collection, points=points)

        # Test query: should retrieve AI-related document
        query = "Tell me about artificial intelligence research"
        query_embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        results = qdrant_client.query_points(
            collection_name=test_collection,
            query=query_embedding,
            limit=1
        )

        # Top result should be the AI/NLP document (index 1)
        top_result = results.points[0]
        assert "AI" in top_result.payload["text"] or "natural language" in top_result.payload["text"]

    finally:
        try:
            qdrant_client.delete_collection(test_collection)
        except Exception:
            pass
