# Credit Scoring RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot API that answers credit-scoring
questions — including summarizing raw customer score breakdowns — grounded in
your own reference documents. It indexes documents into a local FAISS vector
store and generates answers with a locally-hosted Llama 3.2 model via Ollama.

## Features

- **Retrieval-augmented answers** — loads `.txt` / `.docx` documents from a
  `./data` folder, chunks them, and indexes them in a FAISS vector store so
  responses are grounded in your own material rather than the model's raw
  knowledge.
- **Local LLM inference** — runs entirely against a self-hosted Ollama
  instance (`llama3.2`), so no data leaves your infrastructure.
- **Streaming responses** — a `/chat/stream` endpoint streams tokens back as
  they're generated, in addition to a standard synchronous `/chat` endpoint.
- **Per-user conversation memory** — keeps a short rolling history per
  `user_id` so follow-up questions retain context.
- **Performance metrics** — each response includes timing metrics for
  retrieval and generation, useful for profiling.

## Architecture

```
Client
  │  POST /chat or /chat/stream
  ▼
FastAPI server (server.py)
  │
  ▼
RAG pipeline (rag_chatbot.py)
  ├─ Document loader (TextLoader / Docx2txtLoader)
  ├─ Text splitter → chunks
  ├─ HuggingFace embeddings (all-MiniLM-L6-v2)
  ├─ FAISS vector store (similarity search)
  └─ Ollama (llama3.2) → generated answer
```

## Tech Stack

- **API:** FastAPI, Uvicorn
- **RAG / orchestration:** LangChain
- **Vector store:** FAISS
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **LLM:** Llama 3.2, served locally via [Ollama](https://ollama.com)

## Getting Started

### 1. Environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama

Make sure [Ollama](https://ollama.com) is installed and running locally with
the `llama3.2` model pulled:

```bash
ollama pull llama3.2
```

By default the app expects Ollama at `http://10.10.0.147:11434` — update
`OLLAMA_BASE_URL` in `rag_chatbot.py` to point at your own Ollama host
(e.g. `http://localhost:11434`).

### 4. Add your knowledge base

Place `.txt` or `.docx` reference documents in the `./data` directory. These
are what the chatbot retrieves from when answering questions.

### 5. Run the API

```bash
python server.py
```

The API is available at `http://localhost:8088`.

## API Reference

### `POST /chat`

```json
{
  "query": "Summarize this customer's credit score breakdown...",
  "user_id": "default"
}
```

Returns:

```json
{
  "response": "...",
  "metrics": { "retrieval_time": 0.12, "generation_time": 1.8 }
}
```

### `POST /chat/stream`

Same request body as `/chat`, but returns a `text/plain` streaming response
as tokens are generated.

## Project Structure

```
.
├── server.py          # FastAPI app and route definitions
├── rag_chatbot.py      # RAG pipeline: loading, embeddings, retrieval, generation
├── test_query.py       # Example script hitting /chat
├── test_stream.py      # Example script hitting /chat/stream
├── requirements.txt
└── data/                # Your knowledge base documents (not tracked in git)
```
