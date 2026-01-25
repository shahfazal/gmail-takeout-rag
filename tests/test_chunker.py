"""
Tests for TokenChunker
"""

import pytest
from src.chunker import TokenChunker


@pytest.fixture
def chunker():
    """Fixture to create chunker instance."""
    return TokenChunker(chunk_size=50, overlap=10)


@pytest.fixture
def sample_email():
    """Sample email data for testing."""
    return {
        "source": "/path/to/email.html",
        "subject": "Test Email",
        "from": "test@example.com",
        "date": "Mon, 1 Jan 2024 12:00:00 +0000",
        "body_text": "This is a short test email body. " * 20  # Repeat to ensure multiple chunks
    }


def test_chunker_initialization():
    """Test chunker initialization with custom parameters."""
    chunker = TokenChunker(chunk_size=100, overlap=20)

    assert chunker.chunk_size == 100
    assert chunker.overlap == 20
    assert chunker.encoder is not None


def test_chunker_default_parameters():
    """Test chunker initialization with default parameters."""
    chunker = TokenChunker()

    assert chunker.chunk_size == 300
    assert chunker.overlap == 50


def test_chunk_email_structure(chunker, sample_email):
    """Test that chunk_email returns correct structure."""
    chunks = chunker.chunk_email(sample_email)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    # Check first chunk structure
    first_chunk = chunks[0]
    assert "text" in first_chunk
    assert "source" in first_chunk
    assert "subject" in first_chunk
    assert "from" in first_chunk
    assert "date" in first_chunk
    assert "chunk_id" in first_chunk
    assert "chunk_index" in first_chunk
    assert "token_count" in first_chunk


def test_chunk_email_metadata_preserved(chunker, sample_email):
    """Test that metadata is preserved in chunks."""
    chunks = chunker.chunk_email(sample_email)

    for chunk in chunks:
        assert chunk["source"] == sample_email["source"]
        assert chunk["subject"] == sample_email["subject"]
        assert chunk["from"] == sample_email["from"]
        assert chunk["date"] == sample_email["date"]


def test_chunk_email_indices(chunker, sample_email):
    """Test that chunk indices are sequential."""
    chunks = chunker.chunk_email(sample_email)

    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_chunk_email_size_limits(chunker, sample_email):
    """Test that chunks respect size limits."""
    chunks = chunker.chunk_email(sample_email)

    for chunk in chunks:
        # Each chunk should be <= chunk_size
        assert chunk["token_count"] <= chunker.chunk_size
        # Last chunk might be smaller
        assert chunk["token_count"] > 0


def test_chunk_email_unique_ids(chunker, sample_email):
    """Test that chunk IDs are unique."""
    chunks = chunker.chunk_email(sample_email)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))  # All unique


def test_chunk_email_short_text():
    """Test chunking text shorter than chunk size."""
    chunker = TokenChunker(chunk_size=100, overlap=10)

    short_email = {
        "source": "/path/to/email.html",
        "subject": "Short Email",
        "from": "test@example.com",
        "date": "Mon, 1 Jan 2024 12:00:00 +0000",
        "body_text": "Short message."
    }

    chunks = chunker.chunk_email(short_email)

    # Should still create at least one chunk
    assert len(chunks) >= 1
    assert chunks[0]["text"] == "Short message."


def test_chunk_multiple_emails(chunker):
    """Test chunking multiple emails at once."""
    emails = [
        {
            "source": f"/path/to/email{i}.html",
            "subject": f"Email {i}",
            "from": "test@example.com",
            "date": "Mon, 1 Jan 2024 12:00:00 +0000",
            "body_text": f"This is email number {i}. " * 10
        }
        for i in range(3)
    ]

    all_chunks = chunker.chunk_multiple_emails(emails)

    assert isinstance(all_chunks, list)
    assert len(all_chunks) > 0

    # Should have chunks from all emails
    sources = set(chunk["source"] for chunk in all_chunks)
    assert len(sources) == 3


def test_chunk_empty_text(chunker):
    """Test handling of empty email body."""
    empty_email = {
        "source": "/path/to/email.html",
        "subject": "Empty Email",
        "from": "test@example.com",
        "date": "Mon, 1 Jan 2024 12:00:00 +0000",
        "body_text": ""
    }

    chunks = chunker.chunk_email(empty_email)

    # Should handle gracefully
    assert isinstance(chunks, list)
