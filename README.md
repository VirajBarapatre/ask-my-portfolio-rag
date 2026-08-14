# Ask My Portfolio — RAG Q&A Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about my resume,
project READMEs, and certificates, grounded in retrieved chunks from those documents.

## Status

- [x] Phase 0: Setup
- [x] Phase 1: Document ingestion & chunking
- [x] Phase 2: Embedding & vector store
- [ ] Phase 3: Retrieval + generation pipeline
- [ ] Phase 4: Streamlit UI
- [ ] Phase 5: Azure deployment (optional)
- [ ] Phase 6: NLP preprocessing (optional)

## Project Structure

```
ask-my-portfolio-rag/
├── data/
│   ├── resume/           # drop your resume PDF(s) here
│   ├── readmes/          # drop project README .md files here
│   └── certificates/     # drop certificate summaries as .txt files here
├── src/
│   ├── config.py         # paths + tunable settings (chunk size, model names, etc.)
│   ├── ingest.py         # Phase 1: load + chunk documents
│   └── embed_store.py    # Phase 2: embed chunks + build/query FAISS index
├── faiss_index/          # generated — the persisted vector store (git-ignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Add your documents

- Put your resume PDF(s) in `data/resume/`
- Put your project README `.md` files in `data/readmes/`
- Put certificate summaries as `.txt` files in `data/certificates/`

(These are git-ignored by default since they're personal documents — see `.gitignore`.)

## Run Phase 1 — sanity-check chunking

This loads every document and prints a few sample chunks so you can visually confirm
nothing is being split mid-sentence in a way that destroys meaning.

```bash
python -m src.ingest
```

## Run Phase 2 — build the vector store and test retrieval

This embeds all chunks with `all-MiniLM-L6-v2`, builds a FAISS index, persists it to
`faiss_index/`, then runs a few test queries so you can eyeball whether retrieval is
actually pulling relevant chunks.

```bash
python -m src.embed_store
```

Edit the `TEST_QUERIES` list at the bottom of `src/embed_store.py` to try your own
questions, e.g. "What ML models has Viraj used?" or "What was the RMSE on the pricing
project?"

## Design notes (for interview explanations)

- **Chunking**: `RecursiveCharacterTextSplitter` with ~400 token chunks and 50 token
  overlap. Overlap prevents context from being lost when an idea straddles a chunk
  boundary. Chunk size is deliberately smaller than a typical RAG default (~1000)
  because the source documents (resume, READMEs) are short and dense — smaller
  chunks give more precise retrieval for this kind of content.
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers) — free, local, fast,
  384-dim. No API cost, no network dependency at inference time.
- **Vector store**: FAISS, flat L2 index, persisted to disk with `save_local` so
  re-running the app doesn't require re-embedding every time.
- **Metadata**: every chunk keeps its source filename and document type, so answers
  can eventually cite which document they came from (used in Phase 3/4).

## Next steps (Phase 3+)

Phase 3 will add a `generate.py` that takes the retrieved chunks from
`embed_store.py`, builds a grounded prompt, and calls an LLM (OpenAI/Anthropic API
or a local Ollama model) with instructions to answer only from context and say
"I don't know" when the answer isn't retrievable.
