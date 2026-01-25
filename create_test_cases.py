#!/usr/bin/env python3
"""
Interactive script to help create evaluation test cases

This script helps you:
1. Query your RAG system
2. See which chunks were retrieved
3. Mark which ones are actually relevant
4. Build a test dataset
"""

import json
import os
from pathlib import Path
from src.rag import NewsletterRAG


def show_query_results(rag, question, top_k=5):
    """Show results for a query."""
    print(f"\n{'='*80}")
    print(f"QUERY: {question}")
    print('='*80)

    result = rag.query(question, top_k=top_k)

    print(f"\nANSWER:\n{result['answer']}\n")
    print(f"SOURCES ({len(result['sources'])} retrieved):\n")

    sources_info = []
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. Score: {source['score']:.3f}")
        print(f"   From: {source['from']}")
        print(f"   Subject: {source['subject'][:70]}")
        print(f"   Preview: {source['text_preview'][:100]}...")
        print()

        # Store for later
        sources_info.append({
            "index": i,
            "score": source['score'],
            "text_preview": source['text_preview']
        })

    return result['sources'], sources_info


def interactive_test_case_creator():
    """Interactive test case creator."""
    print("="*80)
    print("INTERACTIVE TEST CASE CREATOR")
    print("="*80)
    print("\nThis will help you create evaluation test cases.")
    print("For each question, you'll see retrieved results and mark which are relevant.\n")

    # Initialize RAG
    print("Initializing RAG system...")
    try:
        rag = NewsletterRAG()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("  1. Qdrant is running: docker ps | grep qdrant")
        print("  2. Collection exists: 'newsletter_chunks'")
        print("  3. OPENAI_API_KEY is set in .env")
        return

    print("RAG system ready!\n")

    # Example questions based on the emails we saw
    example_questions = [
        "Did I get any emails from Instagram?",
        "What emails did I receive from Experian?",
        "Show me newsletters about travel",
        "Any emails about loan offers or credit?",
        "Did I get notifications from social media?",
    ]

    print("Example questions you could test:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")

    print("\nYou can use these or create your own.\n")

    test_cases = []

    while True:
        print("\n" + "="*80)
        question = input("\nEnter a test question (or 'done' to finish): ").strip()

        if question.lower() == 'done':
            break

        if not question:
            continue

        # Query the system
        sources, sources_info = show_query_results(rag, question, top_k=5)

        # Ask which are relevant
        print("\nWhich results are ACTUALLY relevant to the question?")
        print("Enter numbers separated by spaces (e.g., '1 3 5')")
        print("Or 'none' if no results are relevant")
        print("Or 'all' if all results are relevant")

        relevant_input = input("Relevant results: ").strip().lower()

        if relevant_input == 'none':
            relevant_indices = []
        elif relevant_input == 'all':
            relevant_indices = list(range(1, len(sources) + 1))
        else:
            try:
                relevant_indices = [int(x) for x in relevant_input.split()]
            except ValueError:
                print("Invalid input, skipping this question.")
                continue

        # Collect chunk IDs for relevant sources
        # Note: We need to get chunk_id from the original query
        print("\nFetching chunk IDs...")

        # Re-query to get full metadata
        from qdrant_client import QdrantClient
        openai_client = rag.openai_client

        query_embedding = openai_client.embeddings.create(
            model=rag.embedding_model,
            input=question
        ).data[0].embedding

        search_results = rag.qdrant_client.query_points(
            collection_name=rag.collection_name,
            query=query_embedding,
            limit=5
        ).points

        relevant_chunk_ids = []
        for idx in relevant_indices:
            if 1 <= idx <= len(search_results):
                chunk_id = search_results[idx - 1].payload.get("chunk_id", "")
                if chunk_id:
                    relevant_chunk_ids.append(chunk_id)

        # Add to test cases
        test_case = {
            "question": question,
            "relevant_chunk_ids": relevant_chunk_ids,
            "notes": f"Created from interactive session. {len(relevant_chunk_ids)} relevant chunks."
        }

        test_cases.append(test_case)
        print(f"\n✓ Test case added! ({len(test_cases)} total)")

    # Save test cases
    if test_cases:
        output_file = "evaluation/test_cases.json"
        os.makedirs("evaluation", exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(test_cases, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✓ Saved {len(test_cases)} test cases to {output_file}")
        print('='*80)
        print("\nNext steps:")
        print("  1. Review the test cases in evaluation/test_cases.json")
        print("  2. Run evaluation: python run_evaluation.py")
        print("  3. See baseline metrics and iterate!")
    else:
        print("\nNo test cases created.")


def quick_start_with_examples():
    """Create test cases from example questions without interaction."""
    print("Creating example test cases based on your emails...\n")

    rag = NewsletterRAG()

    # Example questions
    example_queries = [
        "Did I get any emails from Instagram?",
        "Show me emails from Experian about loans",
        "Any travel newsletters?",
    ]

    test_cases = []

    for question in example_queries:
        print(f"Testing: {question}")

        # Get results
        result = rag.query(question, top_k=3)

        # For now, assume top 2 are relevant (you'd manually verify this)
        # In practice, you'd review and adjust
        from qdrant_client import QdrantClient
        openai_client = rag.openai_client

        query_embedding = openai_client.embeddings.create(
            model=rag.embedding_model,
            input=question
        ).data[0].embedding

        search_results = rag.qdrant_client.query_points(
            collection_name=rag.collection_name,
            query=query_embedding,
            limit=3
        ).points

        # Take top 2 as relevant (you should manually verify)
        relevant_chunk_ids = [
            search_results[i].payload.get("chunk_id", "")
            for i in range(min(2, len(search_results)))
        ]

        test_cases.append({
            "question": question,
            "relevant_chunk_ids": relevant_chunk_ids,
            "notes": "Auto-generated example - VERIFY MANUALLY"
        })

        print(f"  Found {len(relevant_chunk_ids)} chunks\n")

    # Save
    output_file = "evaluation/example_test_cases.json"
    os.makedirs("evaluation", exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(test_cases, f, indent=2)

    print(f"✓ Saved {len(test_cases)} example test cases to {output_file}")
    print("\n⚠️  IMPORTANT: These are auto-generated. You should:")
    print("   1. Review each test case")
    print("   2. Verify the relevant chunks are actually relevant")
    print("   3. Add more test cases for better evaluation")


if __name__ == "__main__":
    print("Choose mode:")
    print("  1. Interactive mode (recommended)")
    print("  2. Quick start with examples")

    choice = input("\nChoice (1 or 2): ").strip()

    if choice == "1":
        interactive_test_case_creator()
    elif choice == "2":
        quick_start_with_examples()
    else:
        print("Invalid choice")
