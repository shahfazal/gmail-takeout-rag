"""
Text Chunker
Splits long documents into smaller, overlapping chunks for efficient embedding
"""

import tiktoken
from typing import List, Dict, Any


class TokenChunker:
    """
    Splits documents into fixed-size token chunks with overlap.
    
    Why chunking is necessary:
    - Embedding models have token limits (e.g., 8192 tokens)
    - Smaller chunks = more precise retrieval
    - Each chunk becomes a separate searchable unit
    
    Why overlap:
    - Preserves context across chunk boundaries
    - If important info spans boundary, overlap ensures it's captured
    """
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk (default 300)
            overlap: How many tokens to overlap between chunks (default 50)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize tiktoken encoder
        # cl100k_base is the encoding used by GPT-4 and text-embedding models
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def chunk_email(self, email_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits an email into chunks.
        
        Args:
            email_data: Dictionary from EmailPreprocessor with:
                - source: file path
                - subject: email subject
                - from: sender
                - date: date sent
                - body_text: cleaned email body
        
        Returns:
            List of chunk dictionaries, each containing:
                - text: the chunk text
                - source: original email file
                - subject: email subject
                - from: sender
                - date: date
                - chunk_id: unique identifier for this chunk
                - chunk_index: which chunk this is (0, 1, 2, ...)
        """
        chunks = []
        
        # Get the body text
        text = email_data["body_text"]
        
        # Convert text to tokens
        tokens = self.encoder.encode(text)
        
        # Calculate how many chunks we'll create
        # We step by (chunk_size - overlap) to create overlapping chunks
        step_size = self.chunk_size - self.overlap
        
        chunk_index = 0
        
        # Sliding window over tokens
        for i in range(0, len(tokens), step_size):
            # Get tokens for this chunk
            chunk_tokens = tokens[i : i + self.chunk_size]
            
            # Decode back to text
            chunk_text = self.encoder.decode(chunk_tokens)
            
            # Create chunk dictionary with metadata
            chunks.append({
                "text": chunk_text,
                "source": email_data["source"],
                "subject": email_data["subject"],
                "from": email_data["from"],
                "date": email_data["date"],
                "chunk_id": f"{email_data['source']}_chunk_{chunk_index}",
                "chunk_index": chunk_index,
                "token_count": len(chunk_tokens)
            })
            
            chunk_index += 1
        
        return chunks
    
    def chunk_multiple_emails(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convenience method to chunk multiple emails at once.
        
        Args:
            emails: List of email dictionaries from EmailPreprocessor
            
        Returns:
            Flat list of all chunks from all emails
        """
        all_chunks = []
        
        for email in emails:
            email_chunks = self.chunk_email(email)
            all_chunks.extend(email_chunks)
        
        return all_chunks


# Example usage (for testing):
if __name__ == "__main__":
    from preprocessor import EmailPreprocessor
    from pathlib import Path
    
    # Initialize preprocessor and chunker
    preprocessor = EmailPreprocessor()
    chunker = TokenChunker(chunk_size=300, overlap=50)
    
    # Test with one email
    # Update this path to match your local setup
    test_file = "emails/emails_to_html/0001_We_re_updating_our_Privacy_Policy_as_we_expand_AI_.html"
    
    email_data = preprocessor.extract_from_html(test_file)
    chunks = chunker.chunk_email(email_data)
    
    print("="*70)
    print("CHUNKING TEST")
    print("="*70)
    print(f"Original email: {Path(test_file).name}")
    print(f"Body length: {len(email_data['body_text'])} characters")
    print(f"Number of chunks created: {len(chunks)}")
    print("\nChunk details:")
    
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i}:")
        print(f"    Token count: {chunk['token_count']}")
        print(f"    Text preview: {chunk['text'][:100]}...")
        print(f"    Chunk ID: {chunk['chunk_id']}")
