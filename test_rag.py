#!/usr/bin/env python3
"""
Test script for NewsletterRAG module
"""

import sys
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import NewsletterRAG

# Load environment
load_dotenv()

# Initialize RAG
print("Initializing Newsletter RAG...")
rag = NewsletterRAG()

# Test query
print("\n" + "="*80)
print("Testing query: 'What newsletters did I get about YouTube or videos?'")
print("="*80 + "\n")

result = rag.query(
    "What newsletters did I get about YouTube or videos?",
    top_k=3
)

print(f"**Answer:**\n{result['answer']}\n")
print(f"**Sources ({result['num_sources']} found):**\n")

for i, source in enumerate(result['sources'], 1):
    print(f"{i}. {source['subject']}")
    print(f"   From: {source['from']}")
    print(f"   Date: {source['date']}")
    print(f"   Score: {source['score']:.3f}")
    print(f"   Preview: {source['text_preview']}\n")

print("="*80)
print("✅ RAG module test complete!")
