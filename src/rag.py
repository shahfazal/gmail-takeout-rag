"""
Newsletter RAG System

Handles retrieval and generation for newsletter queries.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from qdrant_client import QdrantClient
import os


class NewsletterRAG:
    """RAG system for querying newsletter emails."""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "newsletter_chunks",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini"
    ):
        """
        Initialize the RAG system.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
            openai_api_key: OpenAI API key (defaults to env var)
            embedding_model: OpenAI embedding model name
            chat_model: OpenAI chat model name
        """
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
        self.openai_client = OpenAI(api_key=api_key)
        
        # Model configurations
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        
        # Verify Qdrant connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to Qdrant and collection exists."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"✅ Connected to Qdrant collection '{self.collection_name}'")
            print(f"   Vectors: {collection_info.points_count}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant collection: {e}")
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar chunks for a query.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of chunk dictionaries with text, metadata, and scores
        """
        # Create embedding for the query
        query_embedding = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query
        ).data[0].embedding
        
        # Search Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        ).points
        
        # Extract and format results
        results = []
        for hit in search_results:
            results.append({
                "text": hit.payload["text"],
                "subject": hit.payload["subject"],
                "from": hit.payload["from"],
                "date": hit.payload["date"],
                "score": hit.score,
                "source": hit.payload["source"],
                "chunk_id": hit.payload.get("chunk_id", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "token_count": hit.payload.get("token_count", 0)
            })
        
        return results
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate an answer using retrieved chunks as context.
        
        Args:
            query: User's question
            retrieved_chunks: List of relevant chunks from retrieval
            temperature: LLM temperature
            max_tokens: Max tokens in response
            
        Returns:
            Generated answer string
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i}] {chunk['subject']} (from {chunk['from']}, {chunk['date']})\n"
                f"{chunk['text']}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""You are a helpful assistant that answers questions about newsletter emails.

Use the following context from the user's newsletters to answer their question.
If the context doesn't contain relevant information, say so honestly.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        # Call OpenAI
        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about newsletter emails based on provided context."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
            max_tokens: Max tokens in response
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        # Retrieve relevant chunks
        chunks = self.retrieve_similar_chunks(question, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(question, chunks, temperature, max_tokens)
        
        # Format sources for output
        sources = [
            {
                "subject": chunk["subject"],
                "from": chunk["from"],
                "date": chunk["date"],
                "score": chunk["score"],
                "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            }
            for chunk in chunks
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }
