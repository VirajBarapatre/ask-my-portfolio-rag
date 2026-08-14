"""
Phase 2: Embedding & Vector Store

Embeds each chunk with a local sentence-transformers model, stores the
embeddings + chunk text in a FAISS index, and persists that index to disk so
it doesn't need to be rebuilt on every run.

Run directly to build the index from data/ and test retrieval with a few
sample queries:

    python -m src.embed_store
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.config import EMBEDDING_MODEL_NAME, ENABLE_NLP_ENRICHMENT, FAISS_INDEX_DIR, TOP_K
from src.ingest import load_documents, chunk_documents


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    all-MiniLM-L6-v2: 384-dim, ~80MB, runs on CPU in well under a second per
    chunk. Good default for a demo project — no API key, no cost, no network
    dependency once the model weights are cached locally.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(chunks: list[Document]) -> FAISS:
    """Embed all chunks and build a FAISS index from scratch."""
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS) -> None:
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_INDEX_DIR))
    print(f"Vector store persisted to {FAISS_INDEX_DIR}")


def load_vector_store() -> FAISS:
    """
    Load a previously persisted FAISS index instead of re-embedding everything.

    allow_dangerous_deserialization=True is required because FAISS.load_local
    unpickles a local file. It's only "dangerous" if you don't trust the
    source of the index file — since this is one you built yourself, it's
    safe here.
    """
    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve(vector_store: FAISS, query: str, k: int = TOP_K) -> list[Document]:
    """Embed a query and return the top-k most similar chunks."""
    return vector_store.similarity_search(query, k=k)


def _print_retrieval_results(query: str, results: list[Document]) -> None:
    print(f"\nQuery: {query!r}")
    print("-" * 60)
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("doc_type", "unknown")
        print(f"  [{i}] source={source} doc_type={doc_type}")
        preview = doc.page_content.replace("\n", " ")[:200]
        print(f"      {preview}...")
        skills = doc.metadata.get("skills")
        if skills:
            print(f"      skills: {skills}")
    print()


TEST_QUERIES = [
    "What ML models has this person worked with?",
    "What was the accuracy or performance of the pricing project?",
    "What certifications does this person hold?",
    "What programming languages does this person know?",
]


if __name__ == "__main__":
    print("Loading and chunking documents...")
    raw_docs = load_documents()
    chunks = chunk_documents(raw_docs)
    print(f"Produced {len(chunks)} chunks from {len(raw_docs)} document(s).")
    if ENABLE_NLP_ENRICHMENT:
        from src.nlp_enrich import enrich_chunks  # imported lazily so spaCy
        # (a heavier dependency) is only loaded when enrichment is actually on
        print("\nEnriching chunks with spaCy NER + skill matching...")
        chunks = enrich_chunks(chunks)

    print("\nBuilding vector store (embedding all chunks)...")
    store = build_vector_store(chunks)
    save_vector_store(store)

    print("\nRunning test retrieval queries...")
    for query in TEST_QUERIES:
        results = retrieve(store, query)
        _print_retrieval_results(query, results)

    print(
        "Sanity check: for each query above, do the retrieved chunks actually "
        "contain the answer? If not, note it — that's exactly the kind of "
        "failure case Phase 3 asks you to iterate on (try adjusting CHUNK_SIZE, "
        "CHUNK_OVERLAP, or TOP_K in src/config.py)."
    )
