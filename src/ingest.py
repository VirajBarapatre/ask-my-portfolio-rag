"""
Phase 1: Document Ingestion & Chunking

Loads every source document (resume PDF, project README markdown files,
certificate text summaries), tags each with metadata about where it came from,
and splits everything into overlapping chunks small enough to embed and
retrieve precisely.

Run directly to sanity-check the chunks:

    python -m src.ingest
"""

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

from src.config import (
    RESUME_DIR,
    READMES_DIR,
    CERTIFICATES_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents() -> list[Document]:
    """
    Load every document from data/resume, data/readmes, and data/certificates.

    Each loaded Document gets a `doc_type` metadata field so that later, in the
    Streamlit UI (Phase 4), we can show the user which *kind* of source an
    answer was grounded in, not just the filename.
    """
    documents: list[Document] = []

    # Resume: PDF files
    for pdf_path in sorted(RESUME_DIR.glob("*.pdf")):
        loaded = PyPDFLoader(str(pdf_path)).load()
        for doc in loaded:
            doc.metadata["source"] = pdf_path.name
            doc.metadata["doc_type"] = "resume"
        documents.extend(loaded)

    # Project READMEs: markdown files
    for md_path in sorted(READMES_DIR.glob("*.md")):
        loaded = TextLoader(str(md_path), encoding="utf-8").load()
        for doc in loaded:
            doc.metadata["source"] = md_path.name
            doc.metadata["doc_type"] = "project_readme"
        documents.extend(loaded)

    # Certificates: plain text summaries
    for txt_path in sorted(CERTIFICATES_DIR.glob("*.txt")):
        loaded = TextLoader(str(txt_path), encoding="utf-8").load()
        for doc in loaded:
            doc.metadata["source"] = txt_path.name
            doc.metadata["doc_type"] = "certificate"
        documents.extend(loaded)

    if not documents:
        print(
            "No documents found. Add files to data/resume/, data/readmes/, "
            "or data/certificates/ before running this."
        )

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split loaded documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words — falling back to hard character cuts only as
    a last resort. That's why it's the standard default over a naive
    fixed-length splitter: it minimizes the chance of cutting a sentence in
    a way that loses meaning.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _print_sample_chunks(chunks: list[Document], n: int = 5) -> None:
    print(f"\nTotal chunks produced: {len(chunks)}\n")
    print(f"--- Showing first {min(n, len(chunks))} chunks ---\n")
    for i, chunk in enumerate(chunks[:n]):
        print(f"[Chunk {i}] source={chunk.metadata.get('source')} "
              f"doc_type={chunk.metadata.get('doc_type')}")
        print(chunk.page_content)
        print("-" * 60)


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} raw document(s) before chunking.")

    chunks = chunk_documents(docs)
    _print_sample_chunks(chunks)
