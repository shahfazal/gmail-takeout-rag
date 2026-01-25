"""
Tests for EmailPreprocessor
"""

import pytest
from pathlib import Path
from src.preprocessor import EmailPreprocessor


@pytest.fixture
def preprocessor():
    """Fixture to create preprocessor instance."""
    return EmailPreprocessor()


@pytest.fixture
def sample_html_content():
    """Sample HTML email content for testing."""
    return """
    <html>
    <body>
        <div class="header">
            <div><span class="label">Subject:</span> Test Email Subject</div>
            <div><span class="label">From:</span> sender@example.com</div>
            <div><span class="label">Date:</span> Mon, 1 Jan 2024 12:00:00 +0000</div>
        </div>
        <div class="content">
            <p>This is a test email body with some content.</p>
            <p>Multiple paragraphs and    extra    whitespace.</p>
        </div>
    </body>
    </html>
    """


def test_clean_text(preprocessor):
    """Test text cleaning functionality."""
    dirty_text = "Multiple    spaces\n\n\nand   newlines"
    clean = preprocessor.clean_text(dirty_text)

    assert clean == "Multiple spaces and newlines"
    assert "  " not in clean  # No double spaces


def test_clean_text_empty(preprocessor):
    """Test cleaning empty text."""
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""


def test_clean_text_html_entities(preprocessor):
    """Test removal of HTML entities."""
    text_with_entities = "Hello&nbsp;world&amp;test"
    clean = preprocessor.clean_text(text_with_entities)

    # Should remove HTML entities
    assert "&nbsp;" not in clean
    assert "&amp;" not in clean


def test_extract_from_html_structure(preprocessor, tmp_path, sample_html_content):
    """Test that extract_from_html returns correct structure."""
    # Write sample HTML to temp file
    test_file = tmp_path / "test_email.html"
    test_file.write_text(sample_html_content)

    result = preprocessor.extract_from_html(str(test_file))

    # Check structure
    assert "source" in result
    assert "subject" in result
    assert "from" in result
    assert "date" in result
    assert "body_text" in result

    # Check types
    assert isinstance(result["source"], str)
    assert isinstance(result["subject"], str)
    assert isinstance(result["from"], str)
    assert isinstance(result["date"], str)
    assert isinstance(result["body_text"], str)


def test_extract_from_html_content(preprocessor, tmp_path, sample_html_content):
    """Test that extract_from_html extracts correct content."""
    test_file = tmp_path / "test_email.html"
    test_file.write_text(sample_html_content)

    result = preprocessor.extract_from_html(str(test_file))

    assert result["subject"] == "Test Email Subject"
    assert result["from"] == "sender@example.com"
    assert result["date"] == "Mon, 1 Jan 2024 12:00:00 +0000"
    assert "test email body" in result["body_text"].lower()


def test_extract_from_html_missing_fields(preprocessor, tmp_path):
    """Test extraction with missing header fields."""
    html_content = """
    <html>
    <body>
        <div class="content">
            <p>Just body content, no headers.</p>
        </div>
    </body>
    </html>
    """

    test_file = tmp_path / "test_email.html"
    test_file.write_text(html_content)

    result = preprocessor.extract_from_html(str(test_file))

    # Should return empty strings for missing fields
    assert result["subject"] == ""
    assert result["from"] == ""
    assert result["date"] == ""
    # But body should still be extracted
    assert "Just body content" in result["body_text"]


def test_extract_from_html_file_not_found(preprocessor):
    """Test handling of non-existent file."""
    with pytest.raises(FileNotFoundError):
        preprocessor.extract_from_html("/nonexistent/file.html")
