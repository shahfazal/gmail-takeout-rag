# RAG Evaluation Quick Start

## Running Integration Tests

### 1. Unit Tests Only (Fast, No API Calls)

```bash
# Run all unit tests (current 24 tests, 97% coverage)
pytest tests/ -m "not integration and not slow"

# With coverage report
pytest tests/ -m "not integration" --cov=src --cov-report=html
```

### 2. Integration Tests (Requires Qdrant + APIs)

```bash
# Make sure Qdrant is running
docker ps | grep qdrant

# Run integration tests (costs money - OpenAI API calls)
pytest tests/integration/ -v

# Run only Qdrant tests (no API costs)
pytest tests/integration/test_qdrant_integration.py -v

# Run only OpenAI tests
pytest tests/integration/test_openai_integration.py -v -m requires_api
```

### 3. End-to-End Tests

```bash
# Complete pipeline test (preprocessor → chunker → embeddings → qdrant → query)
pytest tests/integration/test_end_to_end.py -v
```

---

## Evaluating RAG Quality

### Quick Single Query Evaluation

```python
from src.rag import NewsletterRAG
from evaluation.metrics import RAGEvaluator

# Initialize
rag = NewsletterRAG()
evaluator = RAGEvaluator(rag)

# Evaluate one query
question = "Did I get emails from Amazon?"
relevant_ids = {"path/to/amazon_email.html_chunk_0"}  # Ground truth

metrics = evaluator.evaluate_single_query(question, relevant_ids, k=5)

print(f"Precision: {metrics['precision_at_k']:.3f}")
print(f"Faithfulness: {metrics['faithfulness']:.3f}")
```

### Full Dataset Evaluation

```python
# Create test dataset
test_cases = [
    ("Question 1?", {"relevant_chunk_id_1", "relevant_chunk_id_2"}),
    ("Question 2?", {"relevant_chunk_id_3"}),
    # ... more test cases
]

# Evaluate
metrics = evaluator.evaluate_dataset(test_cases, k=5)
report = evaluator.generate_report(metrics)
print(report)
```

---

## Understanding the Metrics

### Retrieval Metrics

| Metric | What It Measures | Good Score | Bad Score |
|--------|------------------|------------|-----------|
| **Precision@K** | Of K results, how many are relevant? | >0.7 | <0.3 |
| **Recall@K** | Of all relevant docs, what % retrieved? | >0.6 | <0.2 |
| **MRR** | How high is the first relevant result? | >0.8 | <0.4 |

### Generation Metrics (LLM-as-Judge)

| Metric | What It Measures | Good Score | Bad Score |
|--------|------------------|------------|-----------|
| **Faithfulness** | Answer grounded in context? | >0.8 | <0.5 |
| **Answer Relevance** | Answer addresses question? | >0.8 | <0.5 |
| **Context Relevance** | Retrieved chunks relevant? | >0.7 | <0.4 |

---

## Example Workflow

### Step 1: Create Ground Truth Dataset

Manually review 20-50 questions and identify relevant chunks:

```json
// evaluation/test_cases.json
[
  {
    "question": "What Amazon orders did I receive?",
    "relevant_chunks": [
      "emails/emails_to_html/1234_Amazon_Order_Confirmation.html_chunk_0",
      "emails/emails_to_html/5678_Amazon_Shipment_Notification.html_chunk_0"
    ]
  }
]
```

### Step 2: Run Evaluation

```bash
python evaluation/example_usage.py
```

### Step 3: Analyze Results

```
RAG SYSTEM EVALUATION REPORT
============================================================

RETRIEVAL METRICS:
  Precision@5:        0.720 ± 0.123
  Recall@5:           0.650 ± 0.145

GENERATION METRICS:
  Faithfulness:       0.850 ± 0.092
  Answer Relevance:   0.780 ± 0.110
  Context Relevance:  0.690 ± 0.134

STATISTICS:
  Avg Retrieved:      5.0
  Avg Answer Length:  247 chars
============================================================
```

### Step 4: Iterate and Improve

Low scores indicate areas to improve:

- **Low Precision**: Retrieval is noisy → tune chunk size, improve embeddings
- **Low Recall**: Missing relevant docs → increase K, improve query expansion
- **Low Faithfulness**: Model hallucinates → improve prompt, add constraints
- **Low Answer Relevance**: Model goes off-topic → improve system prompt
- **Low Context Relevance**: Bad retrieval → tune embeddings, add filters

---

## Cost Estimates

### Integration Tests

- OpenAI embedding tests: ~$0.001 per test
- Chat completion tests: ~$0.002 per test
- **Total per run**: ~$0.05

### Evaluation (LLM-as-Judge)

Each test case requires:
- 1 embedding call (query): $0.0001
- 1 chat completion (answer generation): $0.001
- 3-5 chat completions (metric evaluation): $0.003-$0.005

**Cost per test case**: ~$0.004-$0.006
**Cost for 50 test cases**: ~$0.20-$0.30

---

## Continuous Evaluation

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: |
          pip install -r requirements.txt
          pytest tests/ -m "not integration"

  integration-tests:
    runs-on: ubuntu-latest
    # Only on main branch (to save API costs)
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Start Qdrant
        run: docker run -d -p 6333:6333 qdrant/qdrant
      - name: Run integration tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pip install -r requirements.txt
          pytest tests/integration/
```

---

## Tools for RAG Evaluation

### Open Source

1. **RAGAS**: https://github.com/explodinggradients/ragas
   - Pre-built metrics for RAG
   - Integrates with LangChain

2. **Phoenix by Arize**: https://github.com/Arize-ai/phoenix
   - ML observability
   - Embedding visualization
   - Drift detection

3. **TruLens**: https://www.trulens.org/
   - LLM app evaluation
   - Feedback tracking
   - Guardrails

### Commercial

1. **LangSmith** (LangChain)
   - Production monitoring
   - Test case management
   - A/B testing

2. **Weights & Biases**
   - Experiment tracking
   - Model comparison
   - Hyperparameter tuning

---

## Next Steps

1. **Create test dataset**: Start with 10-20 questions
2. **Run baseline evaluation**: See current performance
3. **Experiment**: Try different chunk sizes, models, prompts
4. **Track metrics over time**: Build a dashboard
5. **Set up CI/CD**: Automate testing on PRs

## Further Reading

See `docs/TESTING_STRATEGY.md` for detailed explanation of metrics and strategies.
