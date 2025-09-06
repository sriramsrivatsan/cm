"""
streamlit_ragv2_patched.py

Patched Streamlit RAG app that fixes tokenization and uses a unified SimpleVectorStore
which picks USEARCH if available, otherwise FAISS. Saves outputs to /mnt/data.

Author: ChatGPT (patched)
"""

try:
    import streamlit as st
except Exception:
    # Minimal stub for streamlit to allow non-interactive testing
    class _DummySidebar:
        def header(self, *args, **kwargs): return None
        def selectbox(self, *args, **kwargs): return args[1] if len(args) > 1 else None
        def text_input(self, *args, **kwargs): return ""
        def markdown(self, *args, **kwargs): pass
        def slider(self, *args, **kwargs): return args[3] if len(args) >= 4 else 5
        def checkbox(self, *args, **kwargs): return False
    class _DummySt:
        def set_page_config(self, *a, **k): pass
        def title(self,*a,**k): pass
        def markdown(self,*a,**k): pass
        sidebar = _DummySidebar()
        def file_uploader(self,*a,**k): return None
        def stop(self): raise SystemExit()
        def button(self,*a,**k): return False
        def spinner(self,*a,**k):
            from contextlib import contextmanager
            @contextmanager
            def _cm(): yield
            return _cm()
        def success(self,*a,**k): pass
        def info(self,*a,**k): pass
        def write(self,*a,**k): pass
        def dataframe(self,*a,**k): pass
        def header(self,*a,**k): pass
        def text_area(self,*a,**k): return ""
        def slider(self,*a,**k): return 5
        def checkbox(self,*a,**k): return False
    st = _DummySt()

import pandas as pd
import numpy as np
import os, re, json
from typing import List, Dict, Any, Tuple

# Optional backends
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import openai
except Exception:
    openai = None
try:
    import faiss
except Exception:
    faiss = None
# usearch may or may not be installed
try:
    import usearch
except Exception:
    usearch = None

st.set_page_config(page_title="RAG Patched — Job Postings", layout="wide")

#########################
# Tokenizer and helpers
#########################
class JobTokenizer:
    def __init__(self):
        self.nonword_re = re.compile(r"[^\w\s\-\.@]")
    def tokenize_company(self, company: str) -> List[str]:
        c = clean_text(company)
        return [t.strip() for t in re.split(r"[,/&]| and | & ", c) if t.strip()]
    def tokenize_job_title(self, title: str) -> List[str]:
        t = clean_text(title)
        parts = re.split(r"[-\/|:,]", t)
        tokens = []
        for p in parts:
            p = p.strip()
            if not p: continue
            tokens.extend(p.split())
        return [tok.lower() for tok in tokens if tok]
    def tokenize_location(self, loc: str, location_type: str = "city") -> List[str]:
        l = clean_text(loc)
        return [x.strip() for x in re.split(r"[,-/]", l) if x.strip()]
    def tokenize_salary(self, sal: str) -> List[str]:
        s = clean_text(sal)
        nums = re.findall(r"\$?\d[\d,]*(?:\.\d+)?", s)
        if nums:
            return nums
        return [s] if s else []
    def tokenize_text(self, text: str, method: str = "simple") -> List[str]:
        t = clean_text(text).lower()
        t = self.nonword_re.sub(" ", t)
        tokens = [w for w in t.split() if len(w) > 1]
        return tokens

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

#########################
# App/task detection
#########################
APP_KEYWORDS = {
    "photoshop":"Adobe Photoshop", "lightroom":"Adobe Lightroom", "illustrator":"Adobe Illustrator", "indesign":"Adobe InDesign",
    "premiere":"Adobe Premiere Pro", "premiere pro":"Adobe Premiere Pro", "after effects":"Adobe After Effects", "aftereffects":"Adobe After Effects",
    "final cut pro":"Final Cut Pro", "davinci":"DaVinci Resolve", "resolve":"DaVinci Resolve", "avid":"Avid Media Composer",
    "gimp":"GIMP", "affinity photo":"Affinity Photo", "capture one":"Capture One", "photopea":"Photopea", "pixlr":"Pixlr",
    "sketch":"Sketch", "figma":"Figma", "canva":"Canva", "coreldraw":"CorelDRAW", "procreate":"Procreate", "inshot":"InShot",
    "capcut":"CapCut", "luma fusion":"LumaFusion", "midjourney":"Midjourney", "chatgpt":"ChatGPT", "dall":"DALL·E", "dalle":"DALL·E",
    "runway":"Runway", "stable diffusion":"Stable Diffusion"
}

TASK_KEYWORDS = {
    "photo_tasks": ["retouch", "photo editing", "image editing", "color correction", "product photographer", "masking", "clipping path"],
    "video_tasks": ["video editing", "motion graphics", "color grading", "post-production", "vfx", "cinematography"],
    "design_tasks": ["layout", "typography", "branding", "logo", "illustration", "typography", "print", "ux", "ui", "mockup"]
}

AI_KEYWORDS = ["chatgpt", "gpt-4", "gpt4", "midjourney", "stable diffusion", "dall", "dalle", "runway", "jasper", "bard", "claude", "openai"]


def detect_apps_and_tasks(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Detect apps, tasks, and AI tools with synonyms, abbreviations, and fuzzy matching.
    Uses direct keyword matching first, then abbreviation mapping, then fuzzy matching (difflib).
    """
    import difflib
    t = text.lower()
    tokens = re.findall(r"\w[\w\-\.\+]*", t)  # words-like tokens
    found_apps = set()

    # Direct keyword matches
    for k, v in APP_KEYWORDS.items():
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            found_apps.add(v)

    # Abbreviation / synonym mapping (common)
    SYNONYMS = {
        "ps":"Adobe Photoshop", "photoshop":"Adobe Photoshop", "ae":"Adobe After Effects", "after effects":"Adobe After Effects",
        "pr":"Adobe Premiere Pro", "premiere":"Adobe Premiere Pro", "fcp":"Final Cut Pro", "fcpx":"Final Cut Pro",
        "resolve":"DaVinci Resolve", "davinci":"DaVinci Resolve", "ai":"Adobe Illustrator", "ai (illustrator)":"Adobe Illustrator",
        "xd":"Adobe XD", "fig":"Figma", "figma":"Figma", "lr":"Adobe Lightroom", "lrcc":"Adobe Lightroom",
        "pp":"Adobe Premiere Pro", "ppro":"Adobe Premiere Pro", "psd":"Adobe Photoshop", "xd":"Adobe XD"
    }
    for token in tokens:
        if token in SYNONYMS:
            found_apps.add(SYNONYMS[token])

    # Fuzzy matching on token set against APP_KEYWORDS keys
    app_keys = list(APP_KEYWORDS.keys())
    for token in tokens:
        if len(token) <= 2:
            continue
        matches = difflib.get_close_matches(token, app_keys, n=2, cutoff=0.85)
        for m in matches:
            found_apps.add(APP_KEYWORDS.get(m, m.title()))

    # Task detection (direct)
    found_tasks = set()
    for _, lst in TASK_KEYWORDS.items():
        for k in lst:
            if re.search(r"\b" + re.escape(k) + r"\b", t):
                found_tasks.add(k)

    # AI tools detection
    found_ais = set([a for a in AI_KEYWORDS if re.search(r"\b" + re.escape(a) + r"\b", t)])

    return sorted(found_apps), sorted(found_tasks), sorted(found_ais)

#########################
# Embedding backend
#########################
class EmbeddingBackend:
    def __init__(self, backend: str = "sentence-transformers", model_name: str = "all-MiniLM-L6-v2", openai_key: str = None):
        self.backend = backend
        self.model_name = model_name
        self.openai_key = openai_key
        self.model = None
        if backend == "sentence-transformers" and SentenceTransformer is not None:
            self.model = SentenceTransformer(self.model_name)
        if backend == "openai" and openai is not None:
            if openai_key:
                openai.api_key = openai_key
            else:
                openai.api_key = os.environ.get("OPENAI_API_KEY")
    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "sentence-transformers":
            if self.model is None:
                raise RuntimeError("SentenceTransformer not available. Install sentence-transformers.")
            embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embs.tolist()
        elif self.backend == "openai":
            if openai is None:
                raise RuntimeError("openai package not available")
            model = "text-embedding-3-small"
            resp = openai.Embedding.create(input=texts, model=model)
            return [r['embedding'] for r in resp['data']]
        else:
            raise RuntimeError("Unknown embedding backend")

#########################
# SimpleVectorStore: USEARCH if available else FAISS
#########################

class SimpleVectorStore:
    def __init__(self, dim: int, backend_preference: str = 'usearch'):
        self.dim = dim
        self.backend = None
        self.index = None
        self.metadatas = []
        self.backend_preference = backend_preference
        # Try according to preference: usearch -> faiss -> brute
        try:
            if backend_preference == 'usearch' and usearch is not None:
                self.backend = 'usearch'
                self.index = usearch.Index(ndim=dim, metric='cos')
            elif backend_preference == 'faiss' and faiss is not None:
                self.backend = 'faiss'
                self.index = faiss.IndexFlatIP(dim)
            else:
                # try reverse
                if usearch is not None:
                    self.backend = 'usearch'
                    self.index = usearch.Index(ndim=dim, metric='cos')
                elif faiss is not None:
                    self.backend = 'faiss'
                    self.index = faiss.IndexFlatIP(dim)
                else:
                    self.backend = 'bruteforce'
                    self._brute_vectors = []
        except Exception:
            # fallback to brute-force
            self.backend = 'bruteforce'
            self._brute_vectors = []

    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        import numpy as _np
        vecs = _np.array(vectors, dtype='float32')
        norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        vecs = vecs / norms
        if self.backend == "faiss":
            self.index.add(vecs)
        elif self.backend == 'usearch':
            # usearch expects python lists or numpy arrays; adapt if needed
            try:
                self.index.add(vecs)
            except Exception:
                # if usearch fails, fall back to brute storage
                self.backend = 'bruteforce'
                if not hasattr(self, '_brute_vectors'):
                    self._brute_vectors = []
                self._brute_vectors.extend(vecs.tolist())
        else:
            # brute-force storage
            if not hasattr(self, '_brute_vectors'):
                self._brute_vectors = []
            self._brute_vectors.extend(vecs.tolist())
        for m in metadatas:
            self.metadatas.append(m)

    def search(self, query_vector: List[float], top_k: int = 5):
        import numpy as _np
        q = _np.array(query_vector, dtype='float32').reshape(1, -1)
        q = q / max(1e-12, _np.linalg.norm(q))
        results = []
        if self.backend == 'faiss':
            D, I = self.index.search(q, top_k)
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self.metadatas):
                    continue
                results.append({'score': float(score), 'metadata': self.metadatas[idx]})
        elif self.backend == 'usearch':
            # usearch API: attempt search; fall back to brute if fails
            try:
                labels, distances = self.index.search(q, top_k)
                for score, idx in zip(distances[0], labels[0]):
                    if idx < 0 or idx >= len(self.metadatas):
                        continue
                    results.append({'score': float(score), 'metadata': self.metadatas[idx]})
            except Exception:
                # fallback to brute
                self.backend = 'bruteforce'
        if self.backend == 'bruteforce':
            # compute cosine similarities against stored vectors
            vecs = _np.array(self._brute_vectors, dtype='float32')
            if vecs.size == 0:
                return []
            norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            vecs = vecs / norms
            sims = (vecs @ q[0]).tolist()
            idxs_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
            for idx, score in idxs_scores:
                results.append({'score': float(score), 'metadata': self.metadatas[idx]})
        return results

# Dummy embedding backend (deterministic) used when other backends are unavailable.
class DummyEmbeddingBackend:
    def __init__(self, dim: int = 384):
        self.dim = dim
    def _text_to_vector(self, text: str):
        # deterministic hash-based vector
        import hashlib, numpy as _np
        h = hashlib.sha256(text.encode('utf-8')).digest()
        arr = _np.frombuffer(h, dtype=_np.uint8).astype('float32')
        if arr.size < self.dim:
            repeats = int(_np.ceil(self.dim / arr.size))
            arr = _np.tile(arr, repeats)[:self.dim]
        else:
            arr = arr[:self.dim]
        # normalize to 0-1 then center
        arr = arr / (arr.max() + 1e-12)
        return arr.tolist()
    def embed(self, texts: List[str]):
        return [self._text_to_vector(t) for t in texts]
def preprocess_and_tokenize(df: pd.DataFrame, tokenizer: JobTokenizer) -> pd.DataFrame:
    cols = df.columns.tolist()
    df_clean = df.copy()
    for c in cols:
        df_clean[c] = df_clean[c].fillna("").astype(str).map(clean_text)
    candidates = [c for c in ['Summary job title', 'Displayed job title', 'Job Description', 'Detailed job location'] if c in df_clean.columns]
    df_clean['search_text'] = df_clean[candidates].agg(" ".join, axis=1).str.lower() if candidates else ""
    df_clean['detected_apps'] = df_clean['search_text'].map(lambda t: detect_apps_and_tasks(t)[0])
    df_clean['detected_tasks'] = df_clean['search_text'].map(lambda t: detect_apps_and_tasks(t)[1])
    df_clean['detected_ai'] = df_clean['search_text'].map(lambda t: detect_apps_and_tasks(t)[2])
    df_clean['company_tokens'] = df_clean.get('company', '').map(lambda c: tokenizer.tokenize_company(c))
    df_clean['title_tokens'] = df_clean.apply(lambda r: tokenizer.tokenize_job_title(r.get('Displayed job title','') or r.get('Summary job title','')), axis=1)
    return df_clean

def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    chunks = []
    current = []
    current_len = 0
    for s in sentences:
        if current_len + len(s) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = [s]
            current_len = len(s)
        else:
            current.append(s)
            current_len += len(s)
    if current:
        chunks.append(" ".join(current))
    return chunks

def build_vectorstore(df: pd.DataFrame, embedder: EmbeddingBackend, cols_for_text: List[str] = ['search_text','Job Description'], vector_backend_preference: str = 'usearch'):
    documents = []
    for idx, row in df.iterrows():
        source_id = int(idx)
        text_to_chunk = " ".join([row.get(c,'') for c in cols_for_text if c in row and isinstance(row.get(c,''), str)])
        if not text_to_chunk.strip():
            continue
        chunks = chunk_text(text_to_chunk, max_chars=800)
        for i, ch in enumerate(chunks):
            meta = {
                "source_id": source_id,
                "chunk_index": i,
                "company": row.get('company',''),
                "title": row.get('Displayed job title','') or row.get('Summary job title',''),
                "detected_apps": row.get('detected_apps',[]),
                "detected_tasks": row.get('detected_tasks',[]),
                "text": ch
            }
            documents.append({"text": ch, "meta": meta})
    texts = [d['text'] for d in documents]
    if not texts:
        raise RuntimeError("No text found to build vectorstore.")
    embeddings = embedder.embed(texts)
    dim = len(embeddings[0])
    store = SimpleVectorStore(dim=dim, backend_preference=vector_backend_preference)
    metas = [d['meta'] for d in documents]
    store.add(embeddings, metas)
    return store, texts

#########################
# Streamlit UI
#########################
def main():
    st.title("RAG Patched — Job Postings")
    st.markdown("Preprocess job postings, build a vectorstore (USEARCH/FAISS) and run retrieval + optional generation.")

    uploaded = st.file_uploader("Upload CSV file (or leave empty to use default Book1.csv)", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        default_path = "/mnt/data/Book1.csv"
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
        else:
            st.error("No CSV provided and default not found at /mnt/data/Book1.csv")
            st.stop()

    st.sidebar.header("Settings")
    backend_choice = st.sidebar.selectbox("Embedding backend", options=['sentence-transformers','openai'], index=0)
    openai_key = st.sidebar.text_input("OpenAI API key (optional)", type='password')
    embed_model = st.sidebar.text_input("Embedding model name", value='all-MiniLM-L6-v2' if backend_choice=='sentence-transformers' else 'text-embedding-3-small')
    vector_backend_choice = st.sidebar.selectbox("Vector backend (preference)", options=['usearch','faiss','bruteforce'], index=0)


    tokenizer = JobTokenizer()

    if st.button("Preprocess & Tokenize dataset"):
        with st.spinner("Processing..."):
            processed = preprocess_and_tokenize(df, tokenizer)
            st.session_state['processed_df'] = processed
            st.success("Preprocessing complete — detected apps and tasks added.")
            st.dataframe(processed.head(200))
            out_path = '/mnt/data/Book1_processed_streamlit.csv'
            processed.to_csv(out_path, index=False)
            st.markdown(f"Processed CSV saved to: `{out_path}`")

    if 'processed_df' not in st.session_state:
        st.info("Run preprocessing first.")
        st.stop()

    processed = st.session_state['processed_df']

    if st.button("Build vectorstore (USEARCH preferred, FAISS fallback)"):
        with st.spinner("Building embeddings and vector index..."):
            try:
                embedder = EmbeddingBackend(backend=backend_choice, model_name=embed_model, openai_key=openai_key or None)
                store, texts = build_vectorstore(processed, embedder, cols_for_text=['search_text','Job Description'], vector_backend_preference=vector_backend_choice)
                st.session_state['vector_store'] = store
                st.session_state['vector_texts'] = texts
                st.success("Vectorstore built and stored in session state.")
                # try saving index if backend is faiss
                try:
                    if store.backend == "faiss":
                        import faiss
                        faiss.write_index(store.index, '/mnt/data/streamlit_vector.index')
                        with open('/mnt/data/streamlit_vector.meta.json','w',encoding='utf-8') as f:
                            json.dump(store.metadatas, f, indent=2)
                        st.markdown("FAISS index saved to `/mnt/data/streamlit_vector.index`")
                    else:
                        st.markdown("Using USEARCH in-memory index (save functionality not implemented for generic usearch wrapper).")
                except Exception as e:
                    st.warning(f"Could not save index: {e}")
            except Exception as e:
                st.error(f"Error building vectorstore: {e}")

    if 'vector_store' not in st.session_state:
        st.info("Build a vectorstore to enable retrieval.")
        st.stop()

    store = st.session_state['vector_store']

    st.header("Query the dataset (retrieval)")
    query = st.text_area("Ask a question about the dataset (e.g., 'Which jobs require Photoshop?')", height=120)
    top_k = st.slider("Top K retrieved chunks", min_value=1, max_value=10, value=5)

    if st.button("Retrieve") and query.strip():
        with st.spinner("Embedding query and retrieving..."):
            embedder = EmbeddingBackend(backend=backend_choice, model_name=embed_model, openai_key=openai_key or None)
            q_emb = embedder.embed([query])[0]
            results = store.search(q_emb, top_k=top_k)
            st.write(f"Top {len(results)} results:")
            for r in results:
                md = r['metadata']
                st.markdown(f"**Score:** {r['score']:.4f} — **Title:** {md.get('title','')} — **Company:** {md.get('company','')}")
                st.write("Detected apps:", md.get('detected_apps', []))
                st.write("Detected tasks:", md.get('detected_tasks', []))
                st.write(md.get('text',''))
                st.write("---")
            if openai_key and openai is not None:
                use_llm = st.checkbox("Also generate an LLM answer (requires OpenAI key)", value=False)
                if use_llm:
                    contexts = "\n\n".join([t for t in st.session_state.get('vector_texts',[])][:top_k])
                    prompt = f"Answer the question based on the following retrieved snippets:\n\n{contexts}\n\nQuestion: {query}\nAnswer:"
                    try:
                        openai.api_key = openai_key
                        gresp = openai.ChatCompletion.create(
                            model='gpt-4o' if False else 'gpt-4o-mini' if False else 'gpt-4o',
                            messages=[{"role":"user","content":prompt}],
                            max_tokens=300,
                            temperature=0.0
                        )
                        answer = gresp['choices'][0]['message']['content']
                        st.markdown("### LLM-generated answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Files saved to**: `/mnt/data/` — processed CSV, vector index (if FAISS built).")

if __name__ == '__main__':
    main()
