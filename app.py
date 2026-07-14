"""
DocuChat (Semantic) — Chat with your documents, powered by vector embeddings.

A retrieval-augmented (RAG) app using semantic search: upload a PDF (or use the
built-in sample), ask questions in natural language, and get answers grounded in
the document with the exact source passage shown. Unlike keyword search, semantic
embeddings match by *meaning* — so "vacation time" finds "annual leave" even
though the words differ.

Stack: Streamlit · Groq (LLM) · fastembed (ONNX embeddings, all-MiniLM-L6-v2) · pypdf.
"""

import os
import re
import time

import streamlit as st
import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader
from groq import Groq

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuChat — Chat with your documents",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

GROQ_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, ~90MB, fast on CPU
CHUNK_SIZE = 500          # characters per chunk — small enough to isolate a topic
CHUNK_OVERLAP = 80        # overlap so context isn't cut mid-thought
TOP_K = 3                 # passages retrieved per question
RATE_LIMIT_QUESTIONS = 15 # per session, protects the demo API key
MAX_PDF_MB = 10

# Optional Hugging Face token — avoids anonymous-request rate limiting when the
# embedding model is downloaded on a cold start (fresh container / after reboot).
try:
    _hf_token = st.secrets.get("HF_TOKEN", "")
except Exception:
    _hf_token = ""
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token

# ----------------------------------------------------------------------------
# Styling — lift Streamlit out of its default look
# ----------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main .block-container { max-width: 720px; padding-top: 2.5rem; padding-bottom: 5rem; }

  h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }

  /* Header */
  .dc-header { border-bottom: 1px solid #ececec; padding-bottom: 1.25rem; margin-bottom: 1.75rem; }
  .dc-title { font-family: 'Fraunces', serif; font-size: 2.1rem; font-weight: 600; color: #1a1a1a; margin: 0; line-height: 1.1; }
  .dc-sub { color: #6b6b6b; font-size: 0.95rem; margin-top: 0.4rem; }

  /* Answer card */
  .dc-answer { background: #faf9f6; border: 1px solid #eae7df; border-radius: 12px; padding: 1.25rem 1.4rem; margin: 0.5rem 0 1rem; font-size: 1.02rem; line-height: 1.6; color: #1f1f1f; }

  /* Source passage */
  .dc-source { background: #ffffff; border: 1px solid #ececec; border-left: 3px solid #c96442; border-radius: 8px; padding: 0.85rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; line-height: 1.55; color: #444; }
  .dc-source-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #c96442; font-weight: 600; margin-bottom: 0.3rem; }
  .dc-score { float: right; color: #999; font-weight: 500; }

  /* Status pill */
  .dc-pill { display: inline-block; background: #f0efe9; color: #555; border-radius: 20px; padding: 0.25rem 0.8rem; font-size: 0.8rem; margin-bottom: 1rem; }
  .dc-pill b { color: #c96442; }

  /* Footer note */
  .dc-note { color: #9a9a9a; font-size: 0.8rem; border-top: 1px solid #f0f0f0; margin-top: 2.5rem; padding-top: 1rem; }

  /* Trim Streamlit chrome */
  #MainMenu, header, footer { visibility: hidden; }
  .stDeployButton { display: none; }
  div[data-testid="stFileUploader"] label { font-size: 0.9rem; color: #555; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Built-in sample document (so the demo works with zero uploads)
# ----------------------------------------------------------------------------
SAMPLE_NAME = "Northwind Ltd — Employee Handbook (excerpt)"
SAMPLE_TEXT = """
Northwind Ltd Employee Handbook

1. Working Hours and Flexibility
Standard working hours are 9:00 AM to 5:00 PM, Monday through Friday, totalling 40 hours per week. Core hours during which all employees are expected to be available are 10:00 AM to 4:00 PM. Outside core hours, employees may adjust their start and end times with manager approval. Remote work is permitted up to three days per week for roles that do not require on-site presence.

2. Annual Leave
Full-time employees are entitled to 26 days of paid annual leave per calendar year, in addition to public holidays. Leave accrues monthly and unused leave of up to 5 days may be carried over into the following year, provided it is used before 31 March. Requests for leave should be submitted at least two weeks in advance through the HR portal. During peak business periods (November and December), leave requests are limited to a maximum of 5 consecutive days.

3. Sick Leave and Absence
Employees who are unable to attend work due to illness must notify their manager before 10:00 AM on the first day of absence. For absences of more than three consecutive days, a medical certificate is required. The company provides up to 10 days of fully paid sick leave per year; beyond this, statutory sick pay applies.

4. Expenses and Reimbursement
Business expenses are reimbursed provided they are pre-approved and supported by valid receipts. Travel by rail should be booked in standard class. Meal allowances when travelling are capped at £30 per day. Expense claims must be submitted within 30 days of the expense being incurred; claims submitted after 60 days will not be reimbursed.

5. Equipment and Remote Work
The company provides a laptop and necessary peripherals to all employees. Employees working remotely are responsible for ensuring a secure internet connection. The company contributes up to £150 towards home-office equipment in an employee's first year. All company equipment must be returned within five working days of an employee's last day.

6. Code of Conduct
Employees are expected to treat colleagues, clients, and partners with respect. Harassment or discrimination of any kind will not be tolerated and may result in disciplinary action up to and including termination. Confidential company information must not be shared outside the organisation without written authorisation.

7. Notice Periods
During the probationary period of three months, either party may terminate employment with one week's notice. After probation, the standard notice period is one month for employees and increases to three months for senior management roles.
"""

# ----------------------------------------------------------------------------
# Core RAG functions
# ----------------------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into topic-focused chunks.

    Strategy: break on blank lines and numbered section headers first (so a
    section like "4. Expenses" becomes its own unit), then size-cap any piece
    that's still too long. This keeps each chunk about one topic, which is what
    makes the retrieved 'source passage' actually match the question.
    """
    text = text.replace('\r\n', '\n')
    # Split on blank lines OR on numbered headers like "4. Something"
    raw_parts = re.split(r'\n\s*\n|(?=\n\s*\d+\.\s+[A-Z])', text)
    parts = []
    for part in raw_parts:
        part = re.sub(r'[ \t]+', ' ', part).strip()
        if len(part) < 40:
            continue
        if len(part) <= size:
            parts.append(part)
        else:
            # size-cap long parts with overlap, breaking near sentence ends
            start = 0
            while start < len(part):
                end = start + size
                if end < len(part):
                    window = part[end:end + 100]
                    m = re.search(r'[.!?]\s', window)
                    if m:
                        end = end + m.end()
                chunk = part[start:end].strip()
                if len(chunk) > 40:
                    parts.append(chunk)
                start = end - overlap
    return parts


def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


# --- RETRIEVAL: semantic vector embeddings ---------------------------------
# The embedding model turns each chunk into a 384-dim vector that captures
# meaning. Retrieval compares the question's vector to each chunk's vector by
# cosine similarity, so a question worded differently from the document ("how
# much vacation") still matches the right passage ("annual leave"). The model
# runs locally via ONNX (fastembed) — no API key, no data leaves the machine.

@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load the ONNX embedding model once and cache it across reruns."""
    return TextEmbedding(model_name=EMBED_MODEL)


def embed_texts(texts):
    """Return an (n, dim) numpy array of L2-normalized embeddings."""
    embedder = load_embedder()
    vecs = np.array(list(embedder.embed(list(texts))), dtype=np.float32)
    # L2-normalize so dot product == cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def build_index(chunks):
    """Embed all chunks up front; the 'index' is just their vectors."""
    matrix = embed_texts(chunks)
    return None, matrix  # (kept 2-tuple shape so the rest of the app is unchanged)


def retrieve(question, chunks, vectorizer, matrix, top_k=TOP_K):
    q_vec = embed_texts([question])[0]          # normalized query vector
    sims = matrix @ q_vec                        # cosine sim (both normalized)
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = [(chunks[i], float(sims[i])) for i in top_idx if sims[i] > 0.15]
    return results
# ---------------------------------------------------------------------------


def generate_answer(question, passages, client):
    context = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{p}" for i, (p, _) in enumerate(passages)
    )
    system = (
        "You are a precise document assistant. Answer the user's question using "
        "ONLY the provided passages from their document. If the answer isn't in "
        "the passages, say so plainly — do not invent information. Keep answers "
        "concise and factual. When helpful, refer to what the document states."
    )
    user = f"Passages from the document:\n\n{context}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
    st.session_state.chunks = None
    st.session_state.vectorizer = None
    st.session_state.matrix = None
    st.session_state.q_count = 0
    st.session_state.history = []


def load_document(name, text):
    chunks = chunk_text(text)
    if not chunks:
        st.error("Couldn't read any text from that file. If it's a scanned PDF, "
                 "it needs OCR first — which can be added for production use.")
        return
    with st.spinner("Embedding document (loading the model on first run may take a few seconds)…"):
        vectorizer, matrix = build_index(chunks)
    st.session_state.doc_name = name
    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.matrix = matrix
    st.session_state.history = []


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="dc-header">
  <div class="dc-title">📄 DocuChat</div>
  <div class="dc-sub">Upload a document and ask it anything. Semantic search finds the right passage by <em>meaning</em>, not just keywords — then answers are grounded in the source, with the exact passage shown.</div>
</div>
""", unsafe_allow_html=True)

# API key check — read from Streamlit secrets (cloud) or environment (local)
def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")

api_key = get_api_key()
if not api_key:
    st.warning("⚠️ No GROQ_API_KEY found. Set it in the app's Secrets to enable answers. "
               "Retrieval (finding the right passage) works without it.")
client = Groq(api_key=api_key) if api_key else None

# Document loader
col1, col2 = st.columns([3, 2])
with col1:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="visible")
with col2:
    st.write("")
    st.write("")
    use_sample = st.button("📋 Try the sample document", use_container_width=True)

if uploaded is not None:
    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        st.error(f"That file is {size_mb:.1f} MB — the demo caps at {MAX_PDF_MB} MB.")
    elif st.session_state.doc_name != uploaded.name:
        text = extract_pdf_text(uploaded)
        load_document(uploaded.name, text)

if use_sample and st.session_state.doc_name != SAMPLE_NAME:
    load_document(SAMPLE_NAME, SAMPLE_TEXT)

# Chat area
if st.session_state.doc_name:
    st.markdown(
        f'<div class="dc-pill">Loaded: <b>{st.session_state.doc_name}</b> · '
        f'{len(st.session_state.chunks)} passages · semantic index</div>',
        unsafe_allow_html=True,
    )

    # Suggested questions for the sample
    if st.session_state.doc_name == SAMPLE_NAME and not st.session_state.history:
        st.caption("Try asking:")
        sug = st.columns(2)
        suggestions = [
            "How many days of annual leave do I get?",
            "What's the notice period after probation?",
            "Can I carry over unused holiday?",
            "What's the daily meal allowance when travelling?",
        ]
        for i, s in enumerate(suggestions):
            if sug[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_q = s

    question = st.chat_input("Ask a question about the document…")
    if "pending_q" in st.session_state:
        question = st.session_state.pop("pending_q")

    if question:
        if st.session_state.q_count >= RATE_LIMIT_QUESTIONS:
            st.error(f"Demo limit reached ({RATE_LIMIT_QUESTIONS} questions per session). "
                     "Refresh to start over — production builds have no such cap.")
        elif client is None:
            st.error("Answers need a GROQ_API_KEY. Retrieval still works — see passages below.")
            passages = retrieve(question, st.session_state.chunks,
                                st.session_state.vectorizer, st.session_state.matrix)
            for i, (p, score) in enumerate(passages):
                st.markdown(f'<div class="dc-source"><div class="dc-source-label">'
                            f'Source passage {i+1}<span class="dc-score">match {score:.0%}</span></div>'
                            f'{p}</div>', unsafe_allow_html=True)
        else:
            st.session_state.q_count += 1
            passages = retrieve(question, st.session_state.chunks,
                                st.session_state.vectorizer, st.session_state.matrix)
            if not passages:
                answer = ("I couldn't find anything relevant to that in the document. "
                          "Try rephrasing, or ask about something the document covers.")
                passages = []
            else:
                with st.spinner("Thinking…"):
                    answer = generate_answer(question, passages, client)
            st.session_state.history.append((question, answer, passages))

    # Render history (newest last)
    for q, a, passages in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f'<div class="dc-answer">{a}</div>', unsafe_allow_html=True)
        if passages:
            with st.expander(f"See the {len(passages)} source passage(s) this answer is based on"):
                for i, (p, score) in enumerate(passages):
                    st.markdown(f'<div class="dc-source"><div class="dc-source-label">'
                                f'Source passage {i+1}<span class="dc-score">match {score:.0%}</span></div>'
                                f'{p}</div>', unsafe_allow_html=True)

else:
    st.info("👆 Upload a PDF or load the sample document to begin.")

# Footer
st.markdown("""
<div class="dc-note">
  DocuChat uses semantic vector search: each passage and question is converted to a
  384-dimensional embedding, and the app retrieves passages by meaning rather than
  matching keywords — so a question can be worded completely differently from the
  document and still find the right answer. Responses are generated only from the
  retrieved passages, keeping them grounded in your actual document. Embeddings run
  locally (all-MiniLM-L6-v2 via ONNX); answers use Groq. Extendable with
  multi-document libraries, OCR for scanned files, and persistent vector storage.
</div>
""", unsafe_allow_html=True)
