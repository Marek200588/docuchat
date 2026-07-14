# 📄 DocuChat (Semantic) — Chat with your documents, powered by vector embeddings

A **retrieval-augmented generation (RAG)** app using **semantic search**. Upload a PDF, ask questions in plain language, and get answers grounded in the document — with the exact source passage shown.

The difference from keyword search: this version matches by **meaning**. Ask *"how much vacation time do I get?"* and it finds the *"annual leave"* section even though the words are completely different. That's what semantic embeddings do that keyword matching can't.

**Live demo:** _[paste your Streamlit Cloud URL here after deploying]_

---

## How it works

```
PDF → text extraction → chunking → embed each chunk into a 384-dim vector
                                                    ↓
   your question → embed into a vector → cosine similarity vs every chunk
                                                    ↓
                  most similar passages → LLM answers from them → answer + sources
```

Each passage and each question is turned into a 384-dimensional embedding by a local model (`all-MiniLM-L6-v2`). Retrieval ranks passages by cosine similarity between the question vector and each passage vector — so relevance is based on meaning, not shared words.

## Why this is the production approach

- **Semantic matching** — handles synonyms, paraphrasing, and questions worded nothing like the source
- **Runs locally** — the embedding model runs on-device via ONNX (`fastembed`). No embedding API, no per-call cost, and document text never leaves the server
- **Grounded answers** — the LLM only sees retrieved passages, and every answer shows the exact passages it used, with a relevance score

## Tech stack

- **Streamlit** — UI and hosting
- **fastembed** (`all-MiniLM-L6-v2`, ONNX) — semantic embeddings, no PyTorch needed
- **Groq** (`llama-3.3-70b`) — answer generation
- **pypdf** — PDF text extraction

## Run it yourself

**Locally:**
```bash
pip install -r requirements.txt
# add your key: create .streamlit/secrets.toml with GROQ_API_KEY = "gsk_..."
streamlit run app.py
```
On first run it downloads the embedding model (~90 MB) once, then caches it.

**Deploy free on Streamlit Community Cloud:**
1. Push this folder to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, point it at `app.py`
3. In **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_your_key"
   ```
4. A free Groq API key comes from [console.groq.com](https://console.groq.com)

The model downloads automatically on Streamlit Cloud — no extra setup.

## Extending for client work

- **Multi-document libraries** — chat across a whole folder, with a persistent vector store (ChromaDB / Qdrant) instead of re-embedding each session
- **OCR** for scanned PDFs and images
- **Larger / multilingual models** — swap `EMBED_MODEL` for a bigger or multilingual embedding model (one-line change)
- **User accounts and saved chats**

## Two versions in this project

- **`app.py` (this one)** — semantic embeddings. The production-grade approach.
- **A TF-IDF keyword version** also exists — lighter and dependency-free, useful as a fast fallback or for very small documents. Same UI, same grounding, keyword retrieval instead of vectors.

---

_Built by Marek — Python developer specialising in AI automation, RAG systems, and document intelligence._
