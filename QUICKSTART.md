# Gmail Takeout RAG - Quick Start

## TL;DR

Query your Gmail exported mails from Cursor using natural language!

## Prerequisites

- Google Takeout Gmail export (.mbox files)
- Docker installed
- Python 3.11+
- OpenAI API key

---

## Part 1: Data Processing (One-Time Setup)

### Step 1: Install Dependencies

```bash
cd gmail-takeout-rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install packages
python3 pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Step 2: Convert .mbox to HTML

```bash
cd emails
python3 mbox_to_html.py /path/to/your/Takeout/Mail/your_emails.mbox
```

This creates HTML files in `emails/emails_to_html/`

### Step 3: Start Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  --name qdrant-newsletter \
  qdrant/qdrant:latest
```

### Step 4: Process & Index Emails

Open `gmail-takeout-rag.ipynb` and run all cells:

```bash
jupyter notebook gmail-takeout-rag.ipynb
```

This will:
- Extract and clean emails
- Chunk into 300-token pieces
- Create embeddings (costs ~$0.001)
- Store in Qdrant

**Verify**: http://localhost:6333/dashboard should show your collection with vectors!

---

## Part 2: MCP Setup for Cursor (5 minutes)

### Step 1: Test RAG Module

```bash
cd gmail-takeout-rag
source .venv/bin/activate
python3 test_rag.py
```

You should see a query about YouTube with sources returned.

### Step 2: Configure Cursor MCP

Find your Cursor MCP config and add:

```json
{
  "mcpServers": {
    "gmail-takeout-rag": {
      "command": "/absolute/path/to/gmail-takeout-rag/.venv/bin/python3",
      "args": [
        "/absolute/path/to/gmail-takeout-rag/mcp_server.py"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-your_key_here"
      }
    }
  }
}
```

**Important**: Use absolute paths! Replace `/absolute/path/to/` with your actual path.

### Step 3: Restart Cursor

### Step 4: Test It!

In Cursor chat, ask:
```
"What newsletters did I receive about YouTube?"
"Show me newsletter collection stats"
```

Cursor will use the `user-gmail-takeout-rag` MCP tool (note the "user-" prefix).

---

## Available MCP Tools

### 1. `query_newsletters`

Search newsletters using natural language.

**Parameters:**
- `question` (required): Your question
- `top_k` (optional, default 5): Number of results

**Example:**
```
"What emails did I get about AI?"
```

### 2. `list_newsletter_stats`

Get collection statistics (number of chunks, status, etc.)

**Example:**
```
"Show me stats"
```

---

## Usage Examples

### In Cursor

```
"What newsletters about machine learning?"
"Show me emails from Instagram"
"Did I get anything about Python in 2020?"
```

### In Jupyter Notebook

```python
# Run cells in newsletter_rag.ipynb
ask_newsletter_expert(
    "What newsletters about Python?",
    top_k=5
)
```

---

## Troubleshooting

### MCP server not showing in Cursor

- Check MCP settings panel for errors
- Verify python path: `which python3` (use the venv path!)
- Try absolute paths everywhere
- Restart Cursor

### "Failed to connect to Qdrant"

```bash
# Check if running
docker ps | grep qdrant

# Restart if needed
docker restart qdrant-newsletter

# Check dashboard
open http://localhost:6333/dashboard
```

### "OpenAI API key not found"

- Check `.env` has `OPENAI_API_KEY=sk-...`
- Verify it's also in Cursor MCP config env section

### "No vectors found"

- Re-run notebook indexing cells
- Check Qdrant dashboard shows vectors
- Verify collection name is `newsletter_chunks`

---

## Architecture

```
User Query (Cursor)
    ↓
MCP Server (mcp_server.py)
    ↓
RAG Module (src/rag.py)
    ↓
Qdrant Docker ← 193+ email chunks
    ↓
OpenAI (embeddings + GPT-4)
    ↓
Answer + Sources
```

---

## Cost

- **Indexing**: ~$0.001 per 100 emails (one-time)
- **Per Query**: ~$0.001 (embedding + LLM)
- Very affordable!

---

## Next Steps

- Add more newsletters (reprocess → chunk → embed → index)
- Try different queries
- Check [README.md](README.md) for full documentation
- See [SUMMARY.md](SUMMARY.md) for architecture details

---

**You're all set!** Query your newsletters from Cursor like chatting with an assistant!
