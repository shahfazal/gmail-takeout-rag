"""
Email Preprocessor
Handles extraction and cleaning of text from HTML email files
"""

from bs4 import BeautifulSoup
from typing import Dict, Any
import re


class EmailPreprocessor:
    """Extracts structured data from HTML email files."""
    
    def extract_from_html(self, html_path: str) -> Dict[str, Any]:
        """
        Extracts email metadata and body content from HTML file.
        
        Args:
            html_path: Path to HTML email file
            
        Returns:
            Dictionary with structure:
            {
                "source": "path/to/file.html",
                "subject": "Email subject line",
                "from": "sender@example.com",
                "date": "Fri, 31 May 2024...",
                "body_text": "Cleaned email body text"
            }
        """
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract metadata from header section
        subject = self._extract_field(soup, "Subject:")
        sender = self._extract_field(soup, "From:")
        date = self._extract_field(soup, "Date:")
        
        # Extract body content
        content_div = soup.find('div', class_='content')
        if content_div:
            body_text = content_div.get_text(separator=' ', strip=True)
        else:
            body_text = soup.get_text(separator=' ', strip=True)
        
        # Clean the text
        body_text = self.clean_text(body_text)
        
        return {
            "source": html_path,
            "subject": subject,
            "from": sender,
            "date": date,
            "body_text": body_text
        }
    
    def _extract_field(self, soup: BeautifulSoup, label: str) -> str:
        """
        Helper to extract a specific field from the email header.
        
        Args:
            soup: BeautifulSoup parsed HTML
            label: Field label to find (e.g., "Subject:", "From:")
            
        Returns:
            Extracted field value or empty string
        """
        # Find the span with class="label" that contains our label text
        label_span = soup.find('span', class_='label', string=label)
        if label_span:
            # Get the parent div and extract text after the label
            parent = label_span.parent
            if parent:
                full_text = parent.get_text()
                # Remove the label part to get just the value
                value = full_text.replace(label, '').strip()
                return value
        return ""
    
    def clean_text(self, text: str) -> str:
        """
        Normalizes whitespace and removes artifacts.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text with normalized whitespace
            
        What this does:
        - Collapses multiple spaces/newlines into single spaces
        - Removes extra whitespace
        - Fixes common encoding issues
        """
        if not text:
            return ""
        
        # Collapse multiple whitespace into single spaces
        text = " ".join(text.split())
        
        # Remove any remaining HTML entities that might have slipped through
        text = re.sub(r'&[a-z]+;', ' ', text)
        
        # Remove URLs (optional - you might want to keep them)
        # text = re.sub(r'https?://\S+', '[URL]', text)
        
        return text.strip()


# Example usage (for testing):
if __name__ == "__main__":
    preprocessor = EmailPreprocessor()
    
    # Test with one email
    # Update this path to match your local setup
    test_file = "emails/emails_to_html/0001_We_re_updating_our_Privacy_Policy_as_we_expand_AI_.html"
    
    email_data = preprocessor.extract_from_html(test_file)
    
    print("="*60)
    print("EMAIL EXTRACTION TEST")
    print("="*60)
    print(f"Source: {email_data['source']}")
    print(f"Subject: {email_data['subject']}")
    print(f"From: {email_data['from']}")
    print(f"Date: {email_data['date']}")
    print("\nBody (first 500 chars):")
    print(email_data['body_text'][:500])
    print(f"\n\nTotal body length: {len(email_data['body_text'])} characters")
