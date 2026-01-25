"""
RAG Evaluation Metrics

This module provides metrics for evaluating both retrieval and generation quality.
"""

from typing import List, Set, Dict, Tuple
import numpy as np
from openai import OpenAI
import os


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Precision@K: Fraction of retrieved docs that are relevant.

        Args:
            retrieved_docs: List of retrieved document IDs (ordered by relevance)
            relevant_docs: Set of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            float: Precision score between 0 and 1
        """
        top_k = retrieved_docs[:k]
        relevant_retrieved = len(set(top_k) & relevant_docs)
        return relevant_retrieved / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Recall@K: Fraction of relevant docs that were retrieved.

        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            float: Recall score between 0 and 1
        """
        if len(relevant_docs) == 0:
            return 0.0

        top_k = retrieved_docs[:k]
        relevant_retrieved = len(set(top_k) & relevant_docs)
        return relevant_retrieved / len(relevant_docs)

    @staticmethod
    def mean_reciprocal_rank(queries_results: List[Tuple[List[str], Set[str]]]) -> float:
        """
        MRR: Average of (1 / rank of first relevant doc).

        Args:
            queries_results: List of (retrieved_docs, relevant_docs) tuples

        Returns:
            float: MRR score (higher is better, max 1.0)
        """
        reciprocal_ranks = []

        for retrieved, relevant in queries_results:
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    reciprocal_ranks.append(1 / rank)
                    break
            else:
                reciprocal_ranks.append(0)  # No relevant doc found

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def ndcg_at_k(relevances: List[float], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain@K.

        Args:
            relevances: List of relevance scores (e.g., 0-5) for retrieved docs
            k: Number of results to consider

        Returns:
            float: NDCG score between 0 and 1
        """
        def dcg_at_k(relevances, k):
            relevances = np.array(relevances)[:k]
            gains = 2**relevances - 1
            discounts = np.log2(np.arange(len(relevances)) + 2)
            return np.sum(gains / discounts)

        dcg = dcg_at_k(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = dcg_at_k(ideal_relevances, k)

        return dcg / idcg if idcg > 0 else 0.0


class GenerationMetrics:
    """Metrics for evaluating generation quality using LLM-as-judge."""

    def __init__(self, api_key: str = None):
        """
        Initialize with OpenAI client for LLM-based evaluation.

        Args:
            api_key: OpenAI API key (defaults to env var)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def faithfulness(self, question: str, context: str, answer: str) -> float:
        """
        Evaluate if the answer is faithful to the provided context.

        Args:
            question: The user's question
            context: Retrieved context passages
            answer: Generated answer

        Returns:
            float: Faithfulness score between 0 and 1
        """
        prompt = f"""You are evaluating if an AI-generated answer is faithful to the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Evaluate:
1. Does the answer make claims not supported by the context?
2. Does the answer contradict the context?
3. Is the answer grounded in the context?

Rate faithfulness from 0-10:
- 0: Completely unfaithful (makes up information)
- 5: Partially faithful (some claims not in context)
- 10: Perfectly faithful (all claims supported by context)

Respond with ONLY the number (0-10)."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )

            score = int(response.choices[0].message.content.strip())
            return score / 10.0
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")
            return 0.0

    def answer_relevance(self, question: str, answer: str) -> float:
        """
        Evaluate if the answer addresses the question.

        Args:
            question: The user's question
            answer: Generated answer

        Returns:
            float: Relevance score between 0 and 1
        """
        prompt = f"""Rate how well this answer addresses the question.

QUESTION: {question}

ANSWER: {answer}

Rate from 0-10:
- 0: Completely irrelevant or doesn't answer the question
- 5: Partially answers the question
- 10: Perfectly addresses the question

Respond with ONLY the number (0-10)."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )

            score = int(response.choices[0].message.content.strip())
            return score / 10.0
        except Exception as e:
            print(f"Error in answer relevance evaluation: {e}")
            return 0.0

    def context_relevance(self, question: str, chunks: List[Dict]) -> float:
        """
        Evaluate average relevance of retrieved chunks to the question.

        Args:
            question: The user's question
            chunks: List of retrieved chunk dictionaries with 'text' field

        Returns:
            float: Average context relevance score between 0 and 1
        """
        if not chunks:
            return 0.0

        scores = []

        for chunk in chunks:
            prompt = f"""Rate how relevant this text chunk is to the question.

QUESTION: {question}

TEXT CHUNK:
{chunk['text'][:500]}...

Rate from 0-10:
- 0: Not relevant at all
- 5: Somewhat relevant
- 10: Highly relevant and directly addresses the question

Respond with ONLY the number (0-10)."""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=5
                )

                score = int(response.choices[0].message.content.strip())
                scores.append(score / 10.0)
            except Exception as e:
                print(f"Error in context relevance evaluation: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores)


class RAGEvaluator:
    """Complete RAG system evaluator."""

    def __init__(self, rag_system, api_key: str = None):
        """
        Initialize evaluator.

        Args:
            rag_system: NewsletterRAG instance
            api_key: OpenAI API key for LLM-based metrics
        """
        self.rag = rag_system
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics(api_key)

    def evaluate_single_query(
        self,
        question: str,
        relevant_doc_ids: Set[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate a single query across all metrics.

        Args:
            question: User's question
            relevant_doc_ids: Ground truth relevant document IDs
            k: Number of results to retrieve

        Returns:
            dict: Dictionary of metric scores
        """
        # Get RAG response
        result = self.rag.query(question, top_k=k)

        # Extract retrieved doc IDs
        retrieved_ids = [chunk["chunk_id"] for chunk in result["sources"]]

        # Retrieval metrics
        precision = self.retrieval_metrics.precision_at_k(retrieved_ids, relevant_doc_ids, k)
        recall = self.retrieval_metrics.recall_at_k(retrieved_ids, relevant_doc_ids, k)

        # Generation metrics
        context = "\n\n".join([chunk["text"] for chunk in result["sources"]])
        faithfulness = self.generation_metrics.faithfulness(question, context, result["answer"])
        answer_rel = self.generation_metrics.answer_relevance(question, result["answer"])
        context_rel = self.generation_metrics.context_relevance(question, result["sources"])

        return {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "faithfulness": faithfulness,
            "answer_relevance": answer_rel,
            "context_relevance": context_rel,
            "retrieved_count": len(retrieved_ids),
            "answer_length": len(result["answer"])
        }

    def evaluate_dataset(
        self,
        test_cases: List[Tuple[str, Set[str]]],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate RAG system on a test dataset.

        Args:
            test_cases: List of (question, relevant_doc_ids) tuples
            k: Number of results to retrieve

        Returns:
            dict: Average metrics across all test cases
        """
        all_metrics = []

        print(f"Evaluating {len(test_cases)} test cases...")

        for i, (question, relevant_ids) in enumerate(test_cases):
            print(f"  [{i+1}/{len(test_cases)}] {question[:60]}...")
            metrics = self.evaluate_single_query(question, relevant_ids, k)
            all_metrics.append(metrics)

        # Compute averages
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = sum(values) / len(values)
            avg_metrics[f"{key}_std"] = np.std(values)

        return avg_metrics

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """
        Generate a formatted evaluation report.

        Args:
            metrics: Dictionary of metric scores

        Returns:
            str: Formatted report
        """
        report = []
        report.append("=" * 60)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("RETRIEVAL METRICS:")
        report.append(f"  Precision@K:        {metrics.get('precision_at_k', 0):.3f} ± {metrics.get('precision_at_k_std', 0):.3f}")
        report.append(f"  Recall@K:           {metrics.get('recall_at_k', 0):.3f} ± {metrics.get('recall_at_k_std', 0):.3f}")
        report.append("")
        report.append("GENERATION METRICS:")
        report.append(f"  Faithfulness:       {metrics.get('faithfulness', 0):.3f} ± {metrics.get('faithfulness_std', 0):.3f}")
        report.append(f"  Answer Relevance:   {metrics.get('answer_relevance', 0):.3f} ± {metrics.get('answer_relevance_std', 0):.3f}")
        report.append(f"  Context Relevance:  {metrics.get('context_relevance', 0):.3f} ± {metrics.get('context_relevance_std', 0):.3f}")
        report.append("")
        report.append("STATISTICS:")
        report.append(f"  Avg Retrieved:      {metrics.get('retrieved_count', 0):.1f}")
        report.append(f"  Avg Answer Length:  {metrics.get('answer_length', 0):.0f} chars")
        report.append("=" * 60)

        return "\n".join(report)
