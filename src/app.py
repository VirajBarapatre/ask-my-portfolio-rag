"""
Phase 4: Streamlit UI — "case file" design (v4 — enriched sidebar)

Run with:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import CERTIFICATES_DIR, GROQ_MODEL, LLM_PROVIDER, OLLAMA_MODEL, READMES_DIR, RESUME_DIR
from src.embed_store import load_vector_store, retrieve
from src.generate import build_prompt, call_llm


st.set_page_config(page_title="Ask My Portfolio · Case File", page_icon="🗂️", layout="wide")

DOC_TYPE_META = {
    "resume":          {"label": "RESUME",      "tilt": -2},
    "project_readme":  {"label": "PROJECT",     "tilt": 1},
    "certificate":     {"label": "CERTIFICATE", "tilt": -1},
}
CATEGORY_LABELS = {"resume": "Resume", "project_readme": "Project Files", "certificate": "Certificates"}

EXAMPLE_QUESTIONS = [
    "What ML models has Viraj worked with?",
    "What certifications does Viraj hold?",
    "What was Viraj's role in CampusEYE?",
    "What tech stack did AML Sentinel use?",
]

ABOUT_TEXT = (
    "Data & Quantitative Analyst turning complex datasets into decisions — "
    "from production ETL pipelines to retrieval-grounded AI systems like this one."
)

LINKS = [
    {"label": "GitHub", "icon": "🐙", "url": "https://github.com/VirajBarapatre"},
    {"label": "LinkedIn", "icon": "💼", "url": "https://linkedin.com/in/viraj-barapatre"},
    {"label": "Portfolio", "icon": "🌐", "url": "https://viraj-portfolio-three.vercel.app"},
    {"label": "Email", "icon": "✉️", "url": "mailto:virajbarapatre@outlook.com"},
]

TECH_STACK = [
    "Python", "LangChain", "FAISS", "Sentence-Transformers",
    "spaCy", "Ollama", "Groq", "Streamlit",
]

LIGHT = {
    "bg": "#EEF1F6", "card": "#FFFFFF", "border": "#C9D2E3", "ink": "#14192B",
    "ink2": "#5B6478", "answer_bg": "#FBFCFE", "answer_border": "#E1E6EF",
    "pill_bg": "#F6F8FB", "resume": "#2453D6", "project": "#6D3FC4", "cert": "#B5382E",
    "ready_bg": "#EBF7F0", "ready_ink": "#1E8F5F", "ready_border": "#BFE3D0",
}
DARK = {
    "bg": "#141822", "card": "#1B2130", "border": "#2C3446", "ink": "#ECEFF6",
    "ink2": "#98A2B8", "answer_bg": "#212739", "answer_border": "#323C52",
    "pill_bg": "#232A3C", "resume": "#6E9FFF", "project": "#B79BFF", "cert": "#FF8477",
    "ready_bg": "#16281F", "ready_ink": "#4ADE80", "ready_border": "#255239",
}


def render_css() -> None:
    t = DARK if st.session_state.get("theme", "light") == "dark" else LIGHT
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&family=IBM+Plex+Mono:wght@500;600&display=swap');

        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes slideInLeft {{
            from {{ opacity: 0; transform: translateX(-16px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        @keyframes pulseGently {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}

        html, body, [class*="css"] {{ font-family: 'Source Serif 4', serif; }}
        * {{ transition: background-color 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s cubic-bezier(0.4, 0, 0.2, 1), color 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1), transform 0.3s cubic-bezier(0.4, 0, 0.2, 1); }}

        .stApp {{ background-color: {t['bg']}; }}
        section[data-testid="stSidebar"] {{ background-color: {t['card']}; border-right: 1px solid {t['border']}; }}
        .block-container {{ padding-top: 48px; padding-bottom: 48px; max-width: 960px; padding-left: 20px; padding-right: 20px; }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: transparent; border: none !important;
            border-radius: 0 !important; box-shadow: none;
            padding: 0 !important;
        }}

        h1.dossier-title {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 48px;
            color: {t['ink']}; margin: 0 0 12px 0; letter-spacing: -1.2px; line-height: 1.1;
        }}
        p.dossier-sub {{
            font-family: 'Source Serif 4', serif; color: {t['ink2']}; font-size: 16px;
            margin: 0 0 40px 0; font-style: normal; line-height: 1.7; font-weight: 400;
            max-width: 600px;
        }}
        .folder-tab {{
            display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 10px;
            font-weight: 600; letter-spacing: 0.15em; color: {t['ink2']}; text-transform: uppercase;
            margin-bottom: 20px; margin-left: 0; opacity: 0.7;
        }}

        .pill-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 40px 0; align-items: center; }}
        .pill {{
            font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 500;
            padding: 9px 18px; border-radius: 28px; border: 1px solid {t['border']};
            color: {t['ink2']}; background: {t['pill_bg']};
            animation: fadeInUp 0.5s ease-out; height: fit-content;
            box-shadow: 0 1px 2px rgba(20, 25, 43, 0.04);
        }}
        .pill .n {{ color: {t['ink']}; font-weight: 700; }}
        .pill-ready {{ color: {t['ready_ink']}; border-color: {t['ready_border']}; background: {t['ready_bg']}; }}

        .card-label {{
            font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 700;
            letter-spacing: 0.15em; color: {t['ink2']}; text-transform: uppercase;
            margin: 40px 0 18px 0; opacity: 0.7;
        }}

        input[type="text"] {{
            font-family: 'Source Serif 4', serif !important; font-size: 15px !important;
            background: {t['card']} !important; color: {t['ink']} !important;
            border: 1.5px solid {t['border']} !important; border-radius: 8px !important;
            padding: 0 16px !important; line-height: 1.2; height: 44px !important;
            min-height: 44px !important; box-sizing: border-box !important;
            box-shadow: 0 2px 4px rgba(20, 25, 43, 0.03) !important;
        }}
        div[data-testid="stTextInput"],
        div[data-testid="stTextInput"] > div,
        div[data-testid="stTextInput"] div[data-baseweb="input"] {{
            height: 44px !important;
            min-height: 44px !important;
        }}
        div[data-testid="stTextInput"] div[data-baseweb="input"] {{
            border-radius: 8px !important;
        }}
        input[type="text"]::placeholder {{ color: {t['ink2']}bb !important; opacity: 0.5; }}
        input[type="text"]:focus {{
            border-color: {t['resume']} !important;
            box-shadow: 0 0 0 4px {t['resume']}15 !important;
            outline: none !important;
        }}

        div[data-testid="stFormSubmitButton"] > button {{
            background: {t['ink']}; color: {t['bg']}; border: none; border-radius: 8px;
            font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 14px;
            padding: 0 40px; letter-spacing: 0.5px;
            box-shadow: 0 4px 12px rgba(20, 25, 43, 0.15);
            cursor: pointer; position: relative; overflow: hidden; height: 44px !important;
            min-height: 44px !important; box-sizing: border-box;
            display: flex; align-items: center; justify-content: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        div[data-testid="stFormSubmitButton"] {{
            height: 44px !important;
            min-height: 44px !important;
            display: flex;
            align-items: stretch;
        }}
        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stFormSubmitButton"] > button p {{
            margin: 0;
        }}
        div[data-testid="stFormSubmitButton"] > button:hover {{
            background: {t['resume']}; color: white; box-shadow: 0 6px 20px {t['resume']}40;
            transform: translateY(-2px);
        }}
        div[data-testid="stFormSubmitButton"] > button:active {{
            transform: translateY(0);
            box-shadow: 0 2px 8px {t['resume']}30;
        }}

        div.stButton > button {{
            background: {t['card']}; color: {t['ink2']}; border: 1.5px solid {t['border']};
            border-radius: 8px; font-family: 'Source Serif 4', serif; font-size: 14px;
            font-weight: 500; padding: 11px 16px; cursor: pointer; position: relative;
            box-shadow: 0 2px 6px rgba(20, 25, 43, 0.05);
            height: 48px; display: flex; align-items: center; justify-content: center;
            min-height: 48px; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        div.stButton > button:hover {{
            border-color: {t['resume']};
            color: {t['resume']};
            box-shadow: 0 4px 12px {t['resume']}20;
            transform: translateY(-2px);
            background: {t['bg']};
        }}
        div.stButton > button:active {{ transform: translateY(0); }}

        .entry {{
            display: flex; margin-bottom: 40px; opacity: 0;
            animation: slideInLeft 0.6s ease-out forwards;
        }}
        .entry-tab {{
            width: 3px; border-radius: 2px; margin-right: 22px; flex-shrink: 0;
            box-shadow: 0 3px 8px rgba(20, 25, 43, 0.12);
        }}
        .entry-body {{ flex: 1; min-width: 0; }}
        .entry-q {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 17px;
            color: {t['ink']}; margin-bottom: 14px; letter-spacing: -0.4px; line-height: 1.3;
        }}
        .entry-answer {{
            background: {t['card']}; border: 1px solid {t['border']}; border-radius: 10px;
            padding: 24px 26px; color: {t['ink']}; font-size: 15.5px; line-height: 1.85;
            font-family: 'Source Serif 4', serif; word-wrap: break-word;
            box-shadow: 0 2px 8px rgba(20, 25, 43, 0.05);
        }}

        .stamp-row {{ margin-top: 18px; display: flex; flex-wrap: wrap; gap: 10px; }}
        .stamp {{
            font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; font-weight: 500;
            letter-spacing: 0.04em; padding: 6px 11px; border-radius: 4px;
            border: 1px dashed; display: inline-block; transition: all 0.3s ease;
        }}
        .stamp:hover {{ opacity: 0.8; transform: scale(1.02); }}

        /* Sidebar: identity block */
        .sb-name {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 18px;
            color: {t['ink']}; margin: 0 0 6px 0; letter-spacing: -0.3px;
        }}
        .sb-about {{
            color: {t['ink2']}; font-size: 12.5px; line-height: 1.6; margin: 0 0 18px 0;
            font-style: italic;
        }}

        .sb-toggle-row {{ display: flex; justify-content: flex-end; margin-bottom: 8px; }}

        /* Sidebar: links */
        .sb-links {{ display: flex; flex-direction: column; gap: 6px; margin-bottom: 22px; }}
        .sb-link {{
            display: flex; align-items: center; gap: 9px; padding: 8px 10px;
            border-radius: 7px; background: {t['pill_bg']}; border: 1px solid {t['border']};
            color: {t['ink']} !important; text-decoration: none !important;
            font-family: 'Source Serif 4', serif; font-size: 13px; font-weight: 500;
        }}
        .sb-link:hover {{ border-color: {t['resume']}; color: {t['resume']} !important; }}

        /* Sidebar: tech stack pills */
        .sb-tech-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 24px; }}
        .sb-tech-pill {{
            font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600;
            padding: 4px 9px; border-radius: 5px; background: {t['answer_bg']};
            border: 1px solid {t['border']}; color: {t['ink2']};
        }}

        .sb-section-title {{
            font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 700;
            letter-spacing: 0.1em; text-transform: uppercase; color: {t['ink2']};
            opacity: 0.75; margin: 4px 0 10px 0;
        }}
        .sb-divider {{ height: 1px; background: {t['border']}; margin: 4px 0 20px 0; }}

        /* Sidebar: document index */
        .sb-tab {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 14px;
            color: {t['ink']}; margin: 0 0 4px 0; letter-spacing: -0.3px;
        }}
        .sb-description {{
            color: {t['ink2']}; font-size: 11.5px; font-style: italic; margin: 0 0 18px 0;
            line-height: 1.6; opacity: 0.8;
        }}
        .sb-description code {{
            background: {t['answer_bg']}; color: {t['resume']}; padding: 3px 6px;
            border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
            font-style: normal; font-weight: 500;
        }}

        .cat-tab {{
            font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600;
            letter-spacing: 0.1em; text-transform: uppercase; padding: 8px 12px;
            border-radius: 6px 6px 0 0; display: block; margin: 16px 0 0 0;
            box-shadow: 0 1px 3px rgba(20, 25, 43, 0.05);
        }}
        .cat-body {{
            background: {t['answer_bg']}; border: 1px solid {t['border']}; border-radius: 0 6px 6px 6px;
            padding: 12px 12px; margin-bottom: 0; box-shadow: 0 1px 3px rgba(20, 25, 43, 0.04);
        }}
        .file-line {{
            font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: {t['ink2']};
            padding: 8px 0; border-bottom: 1px dotted {t['border']}80;
            word-break: break-all; overflow-wrap: anywhere; line-height: 1.5;
            transition: color 0.2s ease, padding 0.2s ease;
        }}
        .file-line:hover {{ color: {t['ink']}; padding-left: 2px; }}
        .file-line:last-child {{ border-bottom: none; }}

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
            width: 100%; background: {t['pill_bg']}; color: {t['ink2']}; border: 1px solid {t['border']};
            border-radius: 8px; font-family: 'Source Serif 4', serif; font-size: 13px;
            font-weight: 500; padding: 10px 16px; cursor: pointer; height: 44px;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 1px 3px rgba(20, 25, 43, 0.06);
            transition: all 0.25s ease;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{
            border-color: {t['resume']}; color: {t['resume']}; box-shadow: 0 2px 8px {t['resume']}15;
            transform: translateY(-1px); background: {t['answer_bg']};
        }}

        [data-testid="stSpinner"] {{ animation: pulseGently 2s ease-in-out infinite; }}

        div[data-testid="stForm"] {{ background: transparent; border: none; padding: 0 !important; }}
        div[data-testid="stForm"] > div {{ padding: 0 !important; }}

        div[data-testid="stHorizontalBlock"] {{ gap: 12px; }}
        div[data-testid="stColumn"] {{ gap: 12px; }}
        div[data-testid="stHorizontalBlock"] > div {{ min-width: 0; }}

        details {{
            border: 1px solid {t['border']}; border-radius: 8px; padding: 14px 16px;
            background: {t['card']}; margin-top: 14px;
            box-shadow: 0 1px 3px rgba(20, 25, 43, 0.04);
        }}
        summary {{
            font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
            color: {t['ink2']}; cursor: pointer;
        }}
        details[open] summary {{ color: {t['ink']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_vector_store():
    return load_vector_store()


def _index_count(store) -> int:
    try:
        return store.index.ntotal
    except Exception:  # noqa: BLE001
        return -1


def _list_indexed_documents() -> dict[str, list[str]]:
    return {
        "resume": sorted(p.name for p in RESUME_DIR.glob("*.pdf")),
        "project_readme": sorted(p.name for p in READMES_DIR.glob("*.md")),
        "certificate": sorted(p.name for p in CERTIFICATES_DIR.glob("*.txt")),
    }


def ask(query: str) -> dict:
    store = get_vector_store()
    context_docs = retrieve(store, query)
    prompt = build_prompt(query, context_docs)
    answer = call_llm(prompt)
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "doc_type": doc.metadata.get("doc_type", "unknown"),
                "excerpt": doc.page_content[:220],
                "skills": doc.metadata.get("skills", ""),
            }
            for doc in context_docs
        ],
    }


def render_sidebar() -> None:
    t = DARK if st.session_state.get("theme", "light") == "dark" else LIGHT

    # Theme toggle, top-right
    toggle_col = st.sidebar.columns([0.7, 0.3   ])[0]
    with toggle_col:
        is_dark = st.session_state.get("theme", "light") == "dark"
        if st.button("☀️" if is_dark else "🌙", key="theme_toggle", help="Toggle theme"):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()

    # Identity block
    st.sidebar.markdown(
        f"""
        <div class="sb-name">Viraj Barapatre</div>
        <div class="sb-about">{ABOUT_TEXT}</div>
        """,
        unsafe_allow_html=True,
    )

    # Links
    links_html = "".join(
        f'<a class="sb-link" href="{l["url"]}" target="_blank">{l["icon"]} {l["label"]}</a>'
        for l in LINKS
    )
    st.sidebar.markdown(f'<div class="sb-links">{links_html}</div>', unsafe_allow_html=True)

    # Tech stack
    st.sidebar.markdown('<div class="sb-section-title">Built with</div>', unsafe_allow_html=True)
    tech_html = "".join(f'<span class="sb-tech-pill">{tech}</span>' for tech in TECH_STACK)
    st.sidebar.markdown(f'<div class="sb-tech-row">{tech_html}</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # Document index
    st.sidebar.markdown('<div class="sb-tab">🗂️ Case Contents</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div class='sb-description'>"
        "Source documents indexed for retrieval. Edit <code>data/</code>, then run "
        "<code>python -m src.embed_store</code> to re-file.</div>",
        unsafe_allow_html=True,
    )

    documents = _list_indexed_documents()
    colors = {"resume": t["resume"], "project_readme": t["project"], "certificate": t["cert"]}

    for doc_type, files in documents.items():
        ink = colors[doc_type]
        st.sidebar.markdown(
            f"<div class='cat-tab' style='color:{ink}; background:{ink}18;'>"
            f"{CATEGORY_LABELS[doc_type]} ({len(files)})</div>",
            unsafe_allow_html=True,
        )
        body = "".join(f"<div class='file-line'>{f}</div>" for f in files) or (
            "<div class='file-line'>None found.</div>"
        )
        st.sidebar.markdown(f"<div class='cat-body'>{body}</div>", unsafe_allow_html=True)

    st.sidebar.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    if st.sidebar.button("Clear case log", use_container_width=True):
        st.session_state.history = []
        st.rerun()


def render_stamp(doc_type: str, label: str, t: dict) -> str:
    colors = {"resume": t["resume"], "project_readme": t["project"], "certificate": t["cert"]}
    meta = DOC_TYPE_META.get(doc_type, {"label": "SOURCE", "tilt": 0})
    ink = colors.get(doc_type, t["ink2"])
    return (
        f"<span class='stamp' style='color:{ink}; border-color:{ink}99; "
        f"transform:rotate({meta['tilt']}deg);'>{meta['label']} · {label}</span>"
    )


def render_qa(item: dict) -> None:
    t = DARK if st.session_state.get("theme", "light") == "dark" else LIGHT
    colors = {"resume": t["resume"], "project_readme": t["project"], "certificate": t["cert"]}
    top_type = item["sources"][0]["doc_type"] if item["sources"] else "resume"
    tab_color = colors.get(top_type, t["ink2"])

    stamps_html = "".join(render_stamp(s["doc_type"], s["source"], t) for s in item["sources"])

    st.markdown(
        f"""
        <div class="entry">
            <div class="entry-tab" style="background:{tab_color};"></div>
            <div class="entry-body">
                <div class="entry-q">{item['query']}</div>
                <div class="entry-answer">{item['answer']}
                    <div class="stamp-row">{stamps_html}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"View {len(item['sources'])} retrieved chunk(s)"):
        for i, src in enumerate(item["sources"]):
            ink = colors.get(src["doc_type"], t["ink2"])
            st.markdown(
                f"**{i + 1}. {src['source']}** "
                f"<span style='color:{ink}; font-family:IBM Plex Mono,monospace; "
                f"font-size:11px;'>[{src['doc_type']}]</span>",
                unsafe_allow_html=True,
            )
            st.caption(src["excerpt"] + "...")
            if src.get("skills"):
                st.caption(f"🏷️ Detected skills: {src['skills']}")


def render_main() -> None:
    st.markdown('<div class="folder-tab">✨ Premium Q&A</div>', unsafe_allow_html=True)

    st.markdown('<h1 class="dossier-title">Ask My Portfolio</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="dossier-sub">Retrieve answers backed by sources from my résumé, projects, and certificates. '
        "Powered by retrieval-augmented generation.</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading..."):
        store = get_vector_store()
    n_vectors = _index_count(store)
    doc_counts = _list_indexed_documents()

    st.markdown(
        f"""
        <div class="pill-row">
            <span class="pill"><span class="n">{n_vectors if n_vectors >= 0 else '?'}</span> chunks</span>
            <span class="pill"><span class="n">{len(doc_counts['resume'])}</span> resume</span>
            <span class="pill"><span class="n">{len(doc_counts['project_readme'])}</span> projects</span>
            <span class="pill"><span class="n">{len(doc_counts['certificate'])}</span> certs</span>
            <span class="pill">model: <span class="n">{GROQ_MODEL if LLM_PROVIDER == 'groq' else OLLAMA_MODEL}</span> ({LLM_PROVIDER})</span>
            <span class="pill pill-ready">● ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown('<div class="card-label">Quick questions</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLE_QUESTIONS), gap="small")
    clicked_question = None
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(question, key=f"chip_{question}", use_container_width=True):
                clicked_question = question

    st.markdown('<div class="card-label">New inquiry</div>', unsafe_allow_html=True)
    with st.form("ask_form", clear_on_submit=True):
        form_cols = st.columns([1, 0.50], gap="small")
        with form_cols[0]:
            query = st.text_input(
                "Query",
                placeholder="Ask a question...",
                label_visibility="collapsed",
            )
        with form_cols[1]:
            submitted = st.form_submit_button("Search", use_container_width=True)

    question_to_run = clicked_question or (query.strip() if submitted and query.strip() else None)

    if question_to_run:
        with st.spinner("Searching..."):
            try:
                result = ask(question_to_run)
                st.session_state.history.append(result)
            except Exception as exc:  # noqa: BLE001
                st.error(
                    f"Something went wrong: {exc}\n\n"
                    "Common causes: Ollama isn't running (`ollama serve`), the "
                    "model isn't pulled, or the vector store hasn't been built "
                    "yet (`python -m src.embed_store`)."
                )

    t = DARK if st.session_state.get("theme", "light") == "dark" else LIGHT
    if st.session_state.history:
        st.markdown(
            f"<div style='margin-top: 60px; padding-top: 40px; border-top: 1px solid {t['border']};'>"
            f"<p style='font-size: 12px; font-weight: 700; text-transform: uppercase; "
            f"letter-spacing: 0.15em; color: {t['ink2']}; margin: 0;'>Results</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        for item in reversed(st.session_state.history):
            render_qa(item)
    else:
        st.markdown(
            f"<div style='margin-top: 60px; text-align: center; color: {t['ink2']}; font-size: 14px;'>"
            f"<p>Ask a question or select from the quick inquiries above to get started.</p></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    render_css()
    render_sidebar()
    render_main()
