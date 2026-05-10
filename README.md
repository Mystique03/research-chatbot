---
title: Research Paper Summarizer
emoji: 📚
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
---

# Research Paper Summarizer

A RAG-powered research assistant that lets you upload academic papers (PDF), get instant summaries, and ask questions about their content. Falls back to ArXiv, PubMed, and web search when answers aren't found in the uploaded papers.

**Try It:** https://huggingface.co/spaces/Mystique03/research-paper-summarizer

---

## Features

- **PDF Ingestion** : upload one or more research papers; text is extracted, chunked, embedded, and stored in Pinecone
- **Hybrid Search** : combines dense vector search (Pinecone) with sparse keyword search (BM25) for better retrieval
- **Summarization** : generates structured summaries of ingested papers using Qwen LLM
- **Chat Interface** : ask natural language questions; answers include source citations with page numbers
- **External Fallback** : when the answer isn't in the paper, automatically queries ArXiv, PubMed, or the web via Tavily
- **Multi-paper support** : ingest multiple papers and query across all of them

---

**RAG Pipeline:**
1. PDF → text extraction (PyMuPDF) → chunking (512 tokens, 64 overlap)
2. Chunks → embeddings (FastEmbed `BAAI/bge-small-en-v1.5`, 384-dim) → Pinecone upsert
3. Chunks → BM25 index saved locally as `.pkl`
4. Query → hybrid retrieval (Pinecone dense + BM25 sparse) → reranking → LLM answer
5. If confidence low → LangGraph agent queries ArXiv / PubMed / Tavily

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| LLM | Groq (Qwen 2.5) |
| Embeddings | FastEmbed — `BAAI/bge-small-en-v1.5` |
| Vector DB | Pinecone |
| Keyword Search | rank-bm25 |
| PDF Parsing | PyMuPDF |
| Web Search | Tavily |
| Research DBs | ArXiv, PubMed (Biopython) |

---

## Running Locally

**Prerequisites:** Python 3.11, API keys for Pinecone, Groq, and Tavily

```bash
git clone https://github.com/Mystique03/research-chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```env
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=research-papers
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```

Start backend:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Start frontend (new terminal):
```bash
streamlit run app.py
```

Visit `http://localhost:8501`

---

## Deployment (Hugging Face Spaces)

Uses Docker SDK. Both FastAPI and Streamlit run in the same container via `start.sh`.

Set the following in **Space Settings → Variables and secrets:**

| Secret | Value |
|---|---|
| `PINECONE_API_KEY` | your Pinecone key |
| `PINECONE_INDEX_NAME` | `research-papers` |
| `GROQ_API_KEY` | your Groq key |
| `TAVILY_API_KEY` | your Tavily key |

---

## Project Structure

```
research-paper-summarizer/
├── api/
│   └── main.py          # FastAPI endpoints (ingest, summarize, query)
├── rag_pipeline/
│   ├── ingestion.py     # PDF parsing, chunking, embedding, Pinecone upsert
│   ├── retrieval.py     # Hybrid search (dense + BM25)
│   ├── chains.py        # LangChain summarization chain
│   ├── agents.py        # LangGraph orchestration + external fallback
│   └── evaluation.py   # RAGAS evaluation
├── app.py               # Streamlit frontend
├── Dockerfile           # HF Spaces Docker config
├── start.sh             # Starts FastAPI + Streamlit
└── requirements.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check, list loaded papers |
| `POST` | `/ingest` | Upload and process a PDF |
| `POST` | `/summarize/{paper_id}` | Generate summary for a paper |
| `POST` | `/query` | Ask a question across ingested papers |
| `GET` | `/papers` | List all ingested papers with metadata |
