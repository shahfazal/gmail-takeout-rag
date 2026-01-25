"""
Integration tests for OpenAI API

Run with: pytest tests/integration/test_openai_integration.py -v
Requires: OPENAI_API_KEY environment variable

IMPORTANT: These tests make real API calls and cost money!
"""

import pytest
import os
from openai import OpenAI


@pytest.fixture(scope="module")
def openai_client():
    """Create OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api
def test_embedding_api(openai_client):
    """Test OpenAI embedding API."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input="This is a test email about machine learning."
    )

    assert len(response.data) == 1
    embedding = response.data[0].embedding
    assert len(embedding) == 1536  # text-embedding-3-small dimension
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api
def test_batch_embeddings(openai_client):
    """Test batch embedding creation."""
    texts = [
        "First test email",
        "Second test email",
        "Third test email"
    ]

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    assert len(response.data) == 3
    for i, embedding_obj in enumerate(response.data):
        assert len(embedding_obj.embedding) == 1536
        assert embedding_obj.index == i


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api
def test_chat_completion_api(openai_client):
    """Test OpenAI chat completion API."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'test successful' and nothing else."}
        ],
        max_tokens=10,
        temperature=0
    )

    assert len(response.choices) > 0
    content = response.choices[0].message.content
    assert content is not None
    assert "test successful" in content.lower()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api
def test_rag_prompt_with_context(openai_client):
    """Test RAG-style prompt with context."""
    context = """
    Email 1: Subject: Meeting Tomorrow
    Content: Don't forget about our team meeting at 3pm.

    Email 2: Subject: Project Update
    Content: The new feature is 80% complete.
    """

    question = "When is the meeting?"

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0,
        max_tokens=50
    )

    answer = response.choices[0].message.content.lower()
    assert "3pm" in answer or "3 pm" in answer or "tomorrow" in answer


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api
def test_embedding_similarity(openai_client):
    """Test that similar texts have similar embeddings."""
    import numpy as np

    texts = [
        "I love machine learning and AI",
        "Machine learning and artificial intelligence are great",
        "I enjoy cooking pasta and pizza"
    ]

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [np.array(e.embedding) for e in response.data]

    # Calculate cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])  # Similar topics
    sim_0_2 = cosine_similarity(embeddings[0], embeddings[2])  # Different topics

    # Similar texts should have higher similarity
    assert sim_0_1 > sim_0_2
    assert sim_0_1 > 0.8  # Should be quite similar
