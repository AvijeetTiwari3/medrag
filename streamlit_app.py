"""
MedRAG - Medical QA Demo (Streamlit)
Deployed on Streamlit Community Cloud (free)
"""

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG - Medical QA System",
    page_icon="stethoscope",
    layout="wide",
)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    chunks = pd.read_parquet("data/demo_chunks.parquet")
    embeddings = np.load("data/demo_embeddings.npy")
    return chunks, embeddings

# ── Retrieval ─────────────────────────────────────────────────
def retrieve(query, chunks, embeddings, model, top_k=5):
    query_emb = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ query_emb.T).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "content": chunks.iloc[idx]["content"],
            "source": chunks.iloc[idx]["source"],
            "score": float(scores[idx]),
        })
    return results

# ── LLM Answer (HF Inference API) ─────────────────────────────
def generate_answer(question, context, hf_token=None):
    try:
        from huggingface_hub import InferenceClient
        token = hf_token or st.secrets.get("HF_TOKEN", None)
        if not token:
            return None
        client = InferenceClient(token=token)
        prompt = f"""You are a helpful medical information assistant.
Answer the question based ONLY on the provided context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        response = client.text_generation(
            prompt,
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_new_tokens=300,
            temperature=0.1,
        )
        return response.strip()
    except Exception:
        return None

# ── UI ────────────────────────────────────────────────────────
st.title("MedRAG - Medical Question Answering System")
st.markdown("""
**Powered by:** RAG + sentence-transformers + ChromaDB + Phi-3 Mini

> This demo uses **3,000 medical documents** from MedQuAD (NIH, CDC, Mayo Clinic).
> Full system runs **14,981 documents** locally with Ollama LLM.

> **Disclaimer:** Research demo only. Do NOT use for actual medical decisions.
""")

st.divider()

col1, col2 = st.columns([2, 3])

with col1:
    question = st.text_area(
        "Ask a Medical Question",
        placeholder="e.g. What are the symptoms of Type 2 Diabetes?",
        height=120,
    )
    top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
    search_btn = st.button("Search", type="primary", use_container_width=True)

    st.markdown("**Example Questions:**")
    examples = [
        "What are the symptoms of Type 2 Diabetes?",
        "How is hypertension treated?",
        "What causes coronary artery disease?",
        "What are symptoms of asthma?",
        "How does chronic kidney disease progress?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            question = ex
            search_btn = True

with col2:
    if search_btn and question:
        with st.spinner("Searching medical documents..."):
            model = load_model()
            chunks, embeddings = load_data()
            results = retrieve(question, chunks, embeddings, model, top_k)

        # Try LLM answer
        with st.spinner("Generating answer..."):
            context = "\n\n".join([f"[{i+1}] {r['content']}" for i, r in enumerate(results)])
            answer = generate_answer(question, context)

        if answer:
            st.markdown("### Answer")
            st.success(answer)
        else:
            st.markdown("### Most Relevant Information")
            st.info(results[0]["content"] if results else "No results found.")

        st.markdown(f"### Retrieved Sources ({len(results)} documents)")
        for i, r in enumerate(results):
            with st.expander(f"[{i+1}] {r['source']} — Relevance: {r['score']:.3f}"):
                st.write(r["content"])

st.divider()
st.markdown("""
**Tech Stack:** sentence-transformers | numpy retrieval | MedQuAD dataset | Streamlit

**Full Project:** [GitHub - AvijeetTiwari3/medrag](https://github.com/AvijeetTiwari3/medrag)

*Hit@3 accuracy: 68% | Hit@5: 74.5% | MRR: 0.567 (evaluated on 200 test samples)*
""")