"""
Phase 3: Retrieval + Generation Pipeline

Ties Phase 2's retriever to a locally-running LLM (via Ollama) with a prompt
engineered to answer strictly from the retrieved context — and to say
"I don't know" rather than hallucinate when the context doesn't contain the
answer.

Prerequisites:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull llama3.1:8b
    3. Ollama runs a local server automatically after install (default
       http://localhost:11434). If it's not running, start it with:
       ollama serve

Run directly to ask a few real test questions end-to-end:

    python -m src.generate
"""

import ollama

from langchain.schema import Document

from src.config import OLLAMA_MODEL, OLLAMA_HOST
from src.embed_store import load_vector_store, retrieve


SYSTEM_PROMPT = """You are a Q&A assistant that answers questions about Viraj \
Barapatre's resume, projects, and certificates, using ONLY the context provided \
below.

Rules:
1. Answer strictly using the information in the context. Do not use any \
outside knowledge, and do not make assumptions beyond what is stated. \
Specifically: do not infer related skills, tools, or technologies that are \
not explicitly named in the context, even if they seem like a reasonable or \
common pairing (e.g. do not assume someone knows "Bash" just because they \
used Git).
2. If the context does not contain enough information to answer the \
question, respond exactly with: "I don't know based on the available \
documents." Do not guess or fabricate an answer.
3. Preserve names, proper nouns, and technical terms exactly as they appear \
in the context. If a name appears in ALL CAPS in the source (e.g. resume \
headers), convert it to normal title case for readability, but do not split, \
merge, or alter any part of the name itself (e.g. "RAVINDRA" must become \
"Ravindra", never "R Avindra").
4. Be concise and direct. Do not repeat the question back.
5. When helpful, mention which document(s) the answer came from (e.g. \
"According to the resume..." or "Per the AML-Sentinel-AI project README...").
"""


def build_prompt(query: str, context_docs: list[Document]) -> str:
    """
    Assemble the retrieved chunks into a single context block and combine it
    with the user's question into the final prompt sent to the LLM.

    Each chunk is labeled with its source filename so the model can (and is
    instructed to) cite where an answer came from — this is the "show your
    grounding" behavior that matters most in interviews.
    """
    context_blocks = []
    for i, doc in enumerate(context_docs):
        source = doc.metadata.get("source", "unknown")
        context_blocks.append(f"[Source {i + 1}: {source}]\n{doc.page_content}")

    context_text = "\n\n---\n\n".join(context_blocks)

    return f"""Context:
{context_text}

---

Question: {query}

Answer the question using only the context above."""


def call_ollama(prompt: str) -> str:
    """Send the prompt to a locally running Ollama model and return the reply."""
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


def answer_question(query: str, k: int | None = None) -> dict:
    """
    Full RAG pipeline for one question: retrieve relevant chunks, build the
    grounded prompt, call the LLM, and return both the answer and the source
    chunks used — so the caller (e.g. the Streamlit UI in Phase 4) can show
    users exactly what the answer was grounded in.
    """
    store = load_vector_store()
    retrieve_kwargs = {"k": k} if k is not None else {}
    context_docs = retrieve(store, query, **retrieve_kwargs)

    prompt = build_prompt(query, context_docs)
    answer = call_ollama(prompt)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "doc_type": doc.metadata.get("doc_type", "unknown"),
                "excerpt": doc.page_content[:150],
            }
            for doc in context_docs
        ],
    }


TEST_QUESTIONS = [
    "What ML models has Viraj worked with?",
    "What certifications does Viraj hold?",
    "What programming languages does Viraj know?",
    "What was Viraj's role in the CampusEYE project?",
    "Has Viraj worked with Kubernetes?",  # deliberately unanswerable - tests "I don't know"
]


def _print_result(result: dict) -> None:
    print(f"\nQ: {result['query']}")
    print("-" * 60)
    print(f"A: {result['answer']}")
    print("\nGrounded in:")
    for src in result["sources"]:
        print(f"  - {src['source']} ({src['doc_type']})")
    print()


if __name__ == "__main__":
    print("Running end-to-end RAG test questions (retrieval + Ollama generation)...")
    print("This requires Ollama running locally with the model already pulled.\n")

    for question in TEST_QUESTIONS:
        result = answer_question(question)
        _print_result(result)
