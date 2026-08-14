"""
Phase 6: NLP Preprocessing (enrichment, not text mutation)

Design decision worth explaining in an interview: this module does NOT
lowercase, strip stopwords, or stem the chunk text before embedding. Doing so
would actively hurt retrieval quality — sentence-transformer models like
all-MiniLM-L6-v2 are trained on natural, cased, punctuated text, and
destructive preprocessing throws away semantic signal the model relies on
(that style of preprocessing is a holdover from classic TF-IDF/BM25 pipelines,
not modern dense embeddings).

Instead, spaCy is used to *enrich* each chunk with structured metadata that
sits alongside the original, untouched text:
  - Named entities (organizations, dates, locations, etc.) via spaCy's NER
  - Skill/technology mentions via a PhraseMatcher against a curated keyword
    list pulled from the actual resume/project content

This metadata doesn't affect what gets embedded or how similarity search
works — it's additive, useful for things like faceted filtering ("show me
only chunks mentioning FAISS") in a future UI iteration.

Run directly to see enrichment applied to a few real chunks:

    python -m src.nlp_enrich
"""

import spacy
from spacy.matcher import PhraseMatcher

from langchain.schema import Document

from src.ingest import chunk_documents, load_documents


# Curated from the actual resume/project content processed by this pipeline.
# In a larger system this list might come from a skills taxonomy file instead
# of being hardcoded, but for a personal-portfolio-sized project a flat list
# is simple, explicit, and easy to defend in an interview.
SKILL_KEYWORDS = [
    "Python", "SQL", "PostgreSQL", "MySQL", "Snowflake", "BigQuery",
    "Power BI", "Tableau", "Looker Studio", "Apache Superset",
    "Pandas", "NumPy", "Scikit-Learn", "TensorFlow", "Keras", "PyTorch",
    "Random Forest", "Isolation Forest", "CNN", "YOLOv8", "SmolVLM",
    "Flask", "REST API", "FAISS", "LangChain", "Streamlit", "Ollama",
    "Apache Airflow", "Apache Beam", "Google Cloud Platform", "Azure",
    "Git", "GitHub", "Docker", "OpenCV",
]


def load_spacy_model():
    """
    Load once, reuse everywhere. If this errors with a model-not-found
    message, the model wasn't downloaded — run:
        python -m spacy download en_core_web_sm
    """
    return spacy.load("en_core_web_sm")


def build_skill_matcher(nlp) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in SKILL_KEYWORDS]
    matcher.add("SKILLS", patterns)
    return matcher

# Case-insensitive lookup back to the canonical spelling in SKILL_KEYWORDS.
# Needed because PhraseMatcher matches case-insensitively (attr="LOWER") but
# returns whatever casing happened to appear in that specific chunk — e.g. a
# code block with "git clone" would otherwise report the skill as lowercase
# "git" while the resume's skills section reports "Git", producing two
# different-looking entries for the same skill.
_SKILL_CANONICAL = {skill.lower(): skill for skill in SKILL_KEYWORDS}


def extract_entities(nlp, text: str) -> list[str]:
    """Return entities as 'text (LABEL)' strings, e.g. 'Citi (ORG)'."""
    doc = nlp(text)
    # Dedupe while preserving order — the same entity often appears twice in
    # a short chunk (e.g. a company name in a heading and again in a sentence).
    seen = set()
    entities = []
    for ent in doc.ents:
        formatted = f"{ent.text} ({ent.label_})"
        if formatted not in seen:
            seen.add(formatted)
            entities.append(formatted)
    return entities


def extract_skills(nlp, matcher: PhraseMatcher, text: str) -> list[str]:
    doc = nlp(text)
    matches = matcher(doc)
    seen = set()
    skills = []
    for match_id, start, end in matches:
        span_text = doc[start:end].text
        canonical = _SKILL_CANONICAL.get(span_text.lower(), span_text)
        if canonical and canonical.lower() not in seen:
            seen.add(canonical.lower())
            skills.append(canonical)
    return skills


def enrich_chunks(chunks: list[Document]) -> list[Document]:
    """
    Attach `entities` and `skills` metadata to each chunk, in place.

    Metadata values are stored as semicolon-joined strings rather than lists —
    FAISS/langchain metadata round-trips more reliably through simple string
    values than nested lists across save/load cycles, and it keeps the
    printed output in embed_store.py's test queries readable.
    """
    nlp = load_spacy_model()
    matcher = build_skill_matcher(nlp)

    for chunk in chunks:
        entities = extract_entities(nlp, chunk.page_content)
        skills = extract_skills(nlp, matcher, chunk.page_content)
        chunk.metadata["entities"] = "; ".join(entities) if entities else ""
        chunk.metadata["skills"] = "; ".join(skills) if skills else ""

    return chunks


def _print_sample_enrichment(chunks: list[Document], n: int = 5) -> None:
    shown = 0
    for chunk in chunks:
        if not chunk.metadata.get("entities") and not chunk.metadata.get("skills"):
            continue  # skip chunks with nothing interesting for the demo
        print(f"\n[{chunk.metadata.get('source')}]")
        print(chunk.page_content[:150] + "...")
        if chunk.metadata.get("entities"):
            print(f"  Entities: {chunk.metadata['entities']}")
        if chunk.metadata.get("skills"):
            print(f"  Skills:   {chunk.metadata['skills']}")
        shown += 1
        if shown >= n:
            break


if __name__ == "__main__":
    print("Loading and chunking documents...")
    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"Enriching {len(chunks)} chunks with spaCy NER + skill matching...")
    enriched = enrich_chunks(chunks)

    print(f"\n--- Showing {5} enriched chunks with hits ---")
    _print_sample_enrichment(enriched, n=5)