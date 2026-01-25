#!/usr/bin/env python3
"""
Run evaluation on your RAG system

This reads test cases and evaluates your RAG system.
"""

import json
import os
from pathlib import Path
from src.rag import NewsletterRAG
from evaluation.metrics import RAGEvaluator


def load_test_cases(filepath="evaluation/test_cases.json"):
    """Load test cases from JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        print("\nCreate test cases first:")
        print("  python create_test_cases.py")
        return None

    with open(filepath, 'r') as f:
        test_cases = json.load(f)

    return test_cases


def run_evaluation(test_cases_file="evaluation/test_cases.json"):
    """Run complete evaluation."""

    print("="*80)
    print("RAG SYSTEM EVALUATION")
    print("="*80)

    # Load test cases
    print("\n1. Loading test cases...")
    test_cases_data = load_test_cases(test_cases_file)

    if not test_cases_data:
        return

    # Convert to format expected by evaluator
    test_cases = [
        (tc["question"], set(tc["relevant_chunk_ids"]))
        for tc in test_cases_data
    ]

    print(f"   Loaded {len(test_cases)} test cases")

    # Initialize RAG and evaluator
    print("\n2. Initializing RAG system...")
    try:
        rag = NewsletterRAG()
        print("   ✓ RAG system initialized")
    except Exception as e:
        print(f"   Error: {e}")
        return

    print("\n3. Initializing evaluator...")
    evaluator = RAGEvaluator(rag)
    print("   ✓ Evaluator initialized")

    # Run evaluation
    print(f"\n4. Running evaluation on {len(test_cases)} test cases...")
    print("   (This may take a few minutes and will cost ~$0.20-0.30)\n")

    try:
        metrics = evaluator.evaluate_dataset(test_cases, k=5)
    except Exception as e:
        print(f"\n   Error during evaluation: {e}")
        return

    # Show results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    report = evaluator.generate_report(metrics)
    print(report)

    # Save results
    output_file = "evaluation/baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if metrics.get('precision_at_k', 0) < 0.5:
        print("⚠️  Low Precision: System retrieving too many irrelevant chunks")
        print("   Try: Reduce chunk size, improve query preprocessing")

    if metrics.get('recall_at_k', 0) < 0.5:
        print("⚠️  Low Recall: System missing relevant chunks")
        print("   Try: Increase K, improve embeddings, add query expansion")

    if metrics.get('faithfulness', 0) < 0.7:
        print("⚠️  Low Faithfulness: Model making up information")
        print("   Try: Improve system prompt, add constraints, use smaller max_tokens")

    if metrics.get('answer_relevance', 0) < 0.7:
        print("⚠️  Low Answer Relevance: Answers going off-topic")
        print("   Try: Improve system prompt, focus on question in prompt")

    if metrics.get('context_relevance', 0) < 0.6:
        print("⚠️  Low Context Relevance: Retrieved chunks not relevant")
        print("   Try: Different embedding model, tune chunk size, filter results")

    if (metrics.get('precision_at_k', 0) >= 0.7 and
        metrics.get('faithfulness', 0) >= 0.8):
        print("✓ Good performance! System is working well.")
        print("  Consider: Adding more test cases, testing edge cases")


def quick_evaluation():
    """Run a quick test with just a few questions."""
    print("Running quick evaluation with 3 test questions...\n")

    rag = NewsletterRAG()
    evaluator = RAGEvaluator(rag)

    # Quick test cases
    test_cases = [
        (
            "Did I get emails from Instagram?",
            {"emails/emails_to_html/0002_instagram__prillylatuconsina96_and_graoficial_have.html_chunk_0"}
        ),
        (
            "Any emails about credit or loans?",
            {"emails/emails_to_html/0003_Nagma__you_have_new_loan_matches_to_view___.html_chunk_0"}
        ),
        (
            "Show me travel related emails",
            {"emails/emails_to_html/4560_Win_1_of_25_Duluth_Pack_and_Hydro_Flask_hiking_bun.html_chunk_0"}
        )
    ]

    print("Evaluating...")
    metrics = evaluator.evaluate_dataset(test_cases, k=5)

    report = evaluator.generate_report(metrics)
    print("\n" + report)

    print("\n⚠️  Note: This is a quick test with made-up chunk IDs.")
    print("For accurate results, create real test cases:")
    print("  python create_test_cases.py")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_evaluation()
    else:
        run_evaluation()
