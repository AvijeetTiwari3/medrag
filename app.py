"""
MedRAG — Gradio Demo App
Deployable on HuggingFace Spaces (free).
"""

import logging
import os
import sys
import time

import gradio as gr
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load Config ──────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

config = load_config()

# ─── Initialise RAG Pipeline ──────────────────────────────────────────────────

rag_pipeline = None

def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        from src.models.rag import RAGPipeline
        logger.info("Initialising RAG pipeline ...")
        rag_pipeline = RAGPipeline()
    return rag_pipeline

# ─── Core Query Function ──────────────────────────────────────────────────────

def answer_question(question: str, top_k: int = 5) -> tuple:
    """Process a medical question and return answer + sources."""
    if not question or len(question.strip()) < 3:
        return "Please enter a valid question (at least 3 characters).", ""

    try:
        pipeline = get_rag_pipeline()
        result = pipeline.query(question, top_k=top_k)

        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])
        latency = result.get("latency_ms", 0)

        # Format sources
        sources_text = f"**Retrieved {len(sources)} sources** (latency: {latency:.0f}ms)\n\n"
        for i, src in enumerate(sources, 1):
            score = src.get("score", 0)
            source_name = src.get("source", "Unknown")
            content = src.get("content", "")[:300]
            sources_text += f"**[{i}] {source_name}** (relevance: {score:.3f})\n"
            sources_text += f"> {content}...\n\n"

        return answer, sources_text

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        error_msg = (
            f"Error: {str(e)}\n\n"
            "Make sure:\n"
            "1. ChromaDB index is built (run: `make ingest`)\n"
            "2. Ollama is running (run: `ollama serve` + `ollama pull phi3:mini`)"
        )
        return error_msg, ""

# ─── Example Questions ────────────────────────────────────────────────────────

EXAMPLES = [
    ["What are the symptoms of Type 2 Diabetes?"],
    ["How is hypertension diagnosed and treated?"],
    ["What causes coronary artery disease?"],
    ["What are the treatment options for asthma?"],
    ["How does chronic kidney disease progress?"],
    ["What is the difference between Type 1 and Type 2 Diabetes?"],
]

# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="MedRAG — Medical QA System",
    theme=gr.themes.Soft(),
    css="""
    .header { text-align: center; margin-bottom: 20px; }
    .answer-box { min-height: 150px; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🏥 MedRAG — Medical Question Answering System
        ### Powered by RAG + Phi-3 Mini + ChromaDB + sentence-transformers
        
        **Ask any medical question** and get answers grounded in the MedQuAD dataset (47K+ medical QA pairs).
        
        > ⚠️ This is a research demo. Do NOT use for actual medical decisions. Consult a healthcare professional.
        """,
        elem_classes="header",
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Medical Question",
                placeholder="e.g. What are the symptoms of Type 2 Diabetes?",
                lines=3,
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Number of documents to retrieve",
            )
            submit_btn = gr.Button("Get Answer", variant="primary", size="lg")

        with gr.Column(scale=3):
            answer_output = gr.Markdown(
                label="Answer",
                elem_classes="answer-box",
                value="*Your answer will appear here...*",
            )

    sources_output = gr.Markdown(label="Retrieved Sources")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[question_input],
        label="Example Questions",
    )

    gr.Markdown(
        """
        ---
        **Model**: Phi-3 Mini (via Ollama, local inference) | 
        **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers) | 
        **Vector DB**: ChromaDB | 
        **Dataset**: MedQuAD (47K QA pairs)
        
        [GitHub Repository](https://github.com/your-username/medrag) | 
        [API Docs](http://localhost:8000/docs)
        """
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, sources_output],
        api_name="query",
    )

    question_input.submit(
        fn=answer_question,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, sources_output],
    )


# ─── Launch ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
