"""
Central place for paths and tunable settings.

Keeping these in one file means when you're iterating in Phase 3 (tuning chunk
size, top-k, etc. based on failure cases) you only need to change values here,
not hunt through multiple scripts.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESUME_DIR = DATA_DIR / "resume"
READMES_DIR = DATA_DIR / "readmes"
CERTIFICATES_DIR = DATA_DIR / "certificates"

FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"

# --- Chunking (Phase 1) -------------------------------------------------

# Measured in characters, not tokens, since RecursiveCharacterTextSplitter
# splits on characters.
#
# Bumped up from an initial 400/50 after real testing showed 400-char chunks
# were splitting the resume's dense, paragraph-free skills section mid-category
# (e.g. cutting "Machine Learning (Isolation Forest, Random Forest)" away from
# the chunk that actually got retrieved for "what ML models..."). 600/100 keeps
# more of each skills category intact in a single chunk without ballooning
# chunk count much (62 chunks -> roughly 40-45 expected).
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --- Embeddings (Phase 2) -----------------------------------------------

# Small, fast, free, runs locally on CPU. 384-dim output vectors.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval (Phase 2/3) -----------------------------------------------

TOP_K = 5

# --- Generation (Phase 3) -------------------------------------------------

# Model must be pulled locally first: `ollama pull phi4-mini`
# (Using phi4-mini since it's already available; swap to a larger model like
# "llama3.1:8b" if you want to compare quality later.)
OLLAMA_MODEL = "phi4-mini"
OLLAMA_HOST = "http://localhost:11434"

# --- NLP Enrichment (Phase 6, optional) -----------------------------------

# When True, embed_store.py runs each chunk through spaCy NER + skill
# matching before building the vector store, attaching `entities` and
# `skills` metadata (see src/nlp_enrich.py). This does NOT change the text
# that gets embedded — only adds metadata alongside it.
ENABLE_NLP_ENRICHMENT = True