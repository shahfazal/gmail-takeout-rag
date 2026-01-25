# Gmail Takeout RAG

> **Query your Gmail using natural language, powered by RAG (Retrieval-Augmented Generation)**

A RAG POC for processing Google Takeout Gmail exports and making them searchable via natural language queries. Works both as a Jupyter notebook for exploration and as an MCP server for Cursor IDE integration.

**Note**: This was implmemented as an exercise in building a RAG system. Hence the Gmail exports - since the LLMs out there do not have this context (hopefully) and you're really providing never-before-seen content to the LLMs, to really test your RAG is working. 

## Features

- Process **Google Takeout Gmail exports** (.mbox format)
- Convert emails to searchable vector embeddings
- Query via **natural language** ("What newsletters did I get about AI?")
- Interactive exploration via **Jupyter notebook**
- **MCP server** for Cursor IDE integration
- Persistent storage using **Qdrant** vector database
- Fast retrieval with semantic search

## What This Does

```
Google Takeout exported .mbox files
    ↓
Convert to HTML (mbox_to_html.py)
    ↓
Extract & Clean (preprocessor.py)
    ↓
Chunk into tokens (chunker.py)
    ↓
Create embeddings (OpenAI)
    ↓
Store in Qdrant vector DB
    ↓
Query via natural language!
```

## Quick Start

### Prerequisites

1. **Google Takeout Gmail export** (.mbox files)
   - Go to [Google Takeout](https://takeout.google.com/)
   - Select "Mail" only
   - Choose "Export once"
   - Download and extract

2. **System Requirements**
   - Python 3.11+
   - Docker (for Qdrant)
   - OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd gmail-takeout-rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
python3 -m pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Data Processing Pipeline

#### Step 1: Convert .mbox to HTML

```bash
cd emails
python3 mbox_to_html.py path/to/your/Takeout/Mail/your_emails.mbox
```

This creates HTML files in `emails/emails_to_html/`

#### Step 2: Process & Index Emails

Open `gmail-takeout-rag.ipynb` in Jupyter (or Cursor) and run through the cells:

```bash
jupyter notebook gmail-takeout-rag.ipynb
```

The notebook will:
- Extract metadata and clean text
- Chunk into 300-token pieces with 50-token overlap
- Create embeddings via OpenAI
- Store in Qdrant vector database

#### Step 3: Query Your Newsletters

**Via Notebook:**
```python
ask_newsletter_expert("What newsletters did I get about Python?", top_k=5)
```

**Via MCP Server (Cursor):**

See [QUICKSTART.md](QUICKSTART.md) for MCP setup instructions.

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[SUMMARY.md](SUMMARY.md)** - Complete project overview

## Architecture

```
gmail-takeout-rag/
├── src/
│   ├── preprocessor.py     # Extract & clean email HTML
│   ├── chunker.py           # Token-based text chunking
│   └── rag.py               # RAG pipeline (retrieve + generate)
│
├── emails/
│   ├── mbox_to_html.py      # Convert .mbox to HTML
│   └── emails_to_html/      # Output directory (gitignored)
│
├── mcp_server.py            # MCP server for Cursor
├── test_rag.py              # Test script
└── gmail-takeout-rag.ipynb  # Interactive notebook
```

## Technology Stack

- **Vector Database**: [Qdrant](https://qdrant.tech/) (Docker)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **LLM**: OpenAI `gpt-4o-mini` (answer generation)
- **Chunking**: `tiktoken` (`cl100k_base` encoding)
- **MCP**: Model Context Protocol for Cursor integration

## Usage Examples

### Notebook Queries

```python
# Find emails about specific topics
ask_newsletter_expert("Did I get any emails about machine learning?")

# Search by sender
ask_newsletter_expert("Show me newsletters from YouTube")

# Time-based queries
ask_newsletter_expert("What newsletters did I receive in 2020?")
```

### MCP Queries (in Cursor)

```
"What newsletters did I get about AI?"
"Show me newsletter collection statistics"
"Did I receive emails from Instagram?"
```

## Privacy & Security

- All data stays **local** (except OpenAI API calls)
- No email data committed to git (`.gitignore` configured)
- API keys stored in `.env` (never committed)
- Qdrant data directory excluded from git

## Performance

- **Initial Indexing**: ~$0.001 per 100 emails (one-time cost)
- **Query Cost**: ~$0.001 per query (embedding + LLM)
- **Retrieval Speed**: ~500ms per query
- **Generation Speed**: ~2-3s per query

## Contributing

This is a personal project, but feel free to:
- Report issues
- Suggest enhancements
- Fork and adapt for your needs

## License

MIT License - see [LICENSE](LICENSE) file for details

