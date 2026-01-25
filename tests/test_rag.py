"""
Tests for NewsletterRAG

Note: These tests require mocking OpenAI and Qdrant to avoid API calls.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.rag import NewsletterRAG


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    with patch('src.rag.OpenAI') as mock:
        client = Mock()

        # Mock embedding response
        embedding_response = Mock()
        embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        client.embeddings.create.return_value = embedding_response

        # Mock chat completion response
        chat_response = Mock()
        chat_response.choices = [Mock(message=Mock(content="Test answer"))]
        client.chat.completions.create.return_value = chat_response

        mock.return_value = client
        yield client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    with patch('src.rag.QdrantClient') as mock:
        client = Mock()

        # Mock collection info
        collection_info = Mock()
        collection_info.points_count = 100
        client.get_collection.return_value = collection_info

        # Mock search results
        search_result = Mock()
        search_result.points = [
            Mock(
                payload={
                    "text": "Test chunk text",
                    "subject": "Test Subject",
                    "from": "test@example.com",
                    "date": "Mon, 1 Jan 2024",
                    "source": "/path/to/email.html",
                    "chunk_id": "test_chunk_0",
                    "chunk_index": 0,
                    "token_count": 50
                },
                score=0.95
            )
        ]
        client.query_points.return_value = search_result

        mock.return_value = client
        yield client


def test_rag_initialization(mock_openai_client, mock_qdrant_client):
    """Test RAG system initialization."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        rag = NewsletterRAG()

        assert rag.collection_name == "newsletter_chunks"
        assert rag.embedding_model == "text-embedding-3-small"
        assert rag.chat_model == "gpt-4o-mini"


def test_rag_initialization_no_api_key():
    """Test that initialization fails without API key."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key"):
            NewsletterRAG()


def test_retrieve_similar_chunks(mock_openai_client, mock_qdrant_client):
    """Test retrieval of similar chunks."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        rag = NewsletterRAG()

        results = rag.retrieve_similar_chunks("test query", top_k=5)

        assert isinstance(results, list)
        assert len(results) > 0

        # Check result structure
        first_result = results[0]
        assert "text" in first_result
        assert "subject" in first_result
        assert "from" in first_result
        assert "date" in first_result
        assert "score" in first_result
        assert "source" in first_result


def test_generate_answer(mock_openai_client, mock_qdrant_client):
    """Test answer generation."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        rag = NewsletterRAG()

        chunks = [
            {
                "text": "Test chunk",
                "subject": "Test Subject",
                "from": "test@example.com",
                "date": "Mon, 1 Jan 2024",
                "score": 0.95
            }
        ]

        answer = rag.generate_answer("test query", chunks)

        assert isinstance(answer, str)
        assert len(answer) > 0


def test_query_full_pipeline(mock_openai_client, mock_qdrant_client):
    """Test complete RAG query pipeline."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        rag = NewsletterRAG()

        result = rag.query("test query", top_k=3)

        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "num_sources" in result

        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert isinstance(result["num_sources"], int)


def test_query_with_custom_parameters(mock_openai_client, mock_qdrant_client):
    """Test query with custom temperature and max_tokens."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        rag = NewsletterRAG()

        result = rag.query(
            "test query",
            top_k=5,
            temperature=0.5,
            max_tokens=100
        )

        assert isinstance(result, dict)
        assert "answer" in result


def test_sources_formatting(mock_openai_client, mock_qdrant_client):
    """Test that sources are properly formatted."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        rag = NewsletterRAG()

        result = rag.query("test query", top_k=3)

        for source in result["sources"]:
            assert "subject" in source
            assert "from" in source
            assert "date" in source
            assert "score" in source
            assert "text_preview" in source

            # Text preview should be truncated
            assert len(source["text_preview"]) <= 203  # 200 + "..."
