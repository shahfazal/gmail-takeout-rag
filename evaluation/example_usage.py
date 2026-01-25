"""
Example usage of RAG evaluation metrics

This script demonstrates how to evaluate your RAG system.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.rag import NewsletterRAG
from evaluation.metrics import RAGEvaluator

# Load environment
load_dotenv()


def create_test_cases():
    """
    Create test cases for evaluation.

    Each test case is a tuple of:
    (question, set_of_relevant_chunk_ids)

    You would build this by manually reviewing your emails and
    identifying which chunks should be retrieved for each question.
    """
    test_cases = [
        (
            "Did I get any emails from Amazon?",
            {
                # These would be actual chunk IDs from your dataset
                "emails/emails_to_html/1234_Amazon_Order.html_chunk_0",
                "emails/emails_to_html/5678_Amazon_Shipment.html_chunk_0",
            }
        ),
        (
            "What newsletters did I get about AI?",
            {
                "emails/emails_to_html/9012_OpenAI_Newsletter.html_chunk_0",
                "emails/emails_to_html/3456_Google_AI_Update.html_chunk_0",
            }
        ),
        # Add more test cases...
    ]

    return test_cases


def run_evaluation():
    """Run complete evaluation on test dataset."""

    print("Initializing RAG system...")
    rag = NewsletterRAG(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="newsletter_chunks"
    )

    print("Initializing evaluator...")
    evaluator = RAGEvaluator(rag)

    print("Loading test cases...")
    test_cases = create_test_cases()

    print(f"\nRunning evaluation on {len(test_cases)} test cases...\n")
    metrics = evaluator.evaluate_dataset(test_cases, k=5)

    print("\n" + evaluator.generate_report(metrics))

    return metrics


def run_single_query_evaluation():
    """Example: Evaluate a single query."""

    rag = NewsletterRAG()
    evaluator = RAGEvaluator(rag)

    question = "What emails did I get from Instagram?"
    relevant_ids = {
        "emails/emails_to_html/1234_Instagram.html_chunk_0",
        "emails/emails_to_html/5678_Instagram_Weekly.html_chunk_0",
    }

    print(f"Evaluating: {question}\n")
    metrics = evaluator.evaluate_single_query(question, relevant_ids, k=5)

    print("Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    # Choose which example to run
    print("RAG Evaluation Example\n")
    print("1. Single query evaluation")
    print("2. Full dataset evaluation")

    choice = input("\nChoose (1 or 2): ").strip()

    if choice == "1":
        run_single_query_evaluation()
    elif choice == "2":
        run_evaluation()
    else:
        print("Invalid choice")
