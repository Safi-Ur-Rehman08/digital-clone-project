"""Application entry point."""
import os

import gradio as gr


# Retrieval imports
from retrieval_core import (
    load_chunks,
    build_bm25,
    retrieve_bm25
)

from dense_retrieval import (
    load_model,
    load_embeddings,
    retrieve_dense
)

from hybrid_retrieval import retrieve_hybrid

# Gemini import
from generation import generate_answer

from retrieval_core import (
    load_chunks,
    build_bm25,
    retrieve_bm25
)

from dense_retrieval import (
    load_model,
    load_embeddings,
    retrieve_dense
)

from hybrid_retrieval import retrieve_hybrid

from generation import generate_answer
# ==========================================
# LOAD KNOWLEDGE BASE AND RETRIEVAL MODELS
# ==========================================

print("\nLoading Digital Clone Knowledge Base...")

chunks = load_chunks()

if not chunks:
    raise ValueError(
        "No document chunks found. Check your data folder."
    )

print(f"Total chunks loaded: {len(chunks)}")


# Build BM25 index
print("\nBuilding BM25 index...")

bm25 = build_bm25(chunks)

print("BM25 index ready!")


# Load Dense Retrieval model
print("\nLoading Dense Retrieval model...")

model = load_model()

print("Dense model loaded!")


# Load saved embeddings
print("\nLoading saved embeddings...")

embeddings = load_embeddings()

if len(embeddings) != len(chunks):
    raise ValueError(
        "Number of embeddings does not match number of chunks. "
        "Please regenerate embeddings.npy."
    )

print("Embeddings loaded successfully!")


print("\nDigital Clone is ready!")


# ==========================================
# HELPER FUNCTION FOR DISPLAYING SOURCES
# ==========================================

def display_sources(results):

    output = ""

    for result in results:

        score = result.get(
            "rrf_score",
            result.get("score", "N/A")
        )

        output += (
            f"Source: {result.get('source', 'Unknown')}\n"
            f"Chunk ID: {result.get('chunk_id', 'Unknown')}\n"
            f"Score: {score}\n"
            f"Content:\n{result.get('text', '')}\n"
            f"{'-' * 60}\n\n"
        )

    return output


# ==========================================
# MAIN CHATBOT FUNCTION
# ==========================================

def chatbot(question, mode):

    if not question or not question.strip():

        return (
            "Please enter a question.",
            "No retrieved chunks."
        )

    try:

        print(f"\nQuestion: {question}")
        print(f"Retrieval mode: {mode}")


        # ----------------------------------
        # BM25 RETRIEVAL
        # ----------------------------------

        if mode == "BM25 Only":

            results = retrieve_bm25(
                question,
                chunks,
                bm25,
                top_k=5
            )


        # ----------------------------------
        # DENSE RETRIEVAL
        # ----------------------------------

        elif mode == "Dense Only":

            results = retrieve_dense(
                question,
                chunks,
                model,
                embeddings,
                top_k=5
            )


        # ----------------------------------
        # HYBRID RETRIEVAL USING RRF
        # ----------------------------------

        else:

            results = retrieve_hybrid(
                question,
                chunks,
                bm25,
                model,
                embeddings
            )


        # Check retrieval results
        if not results:

            return (
                "I don't have that in my notes.",
                "No relevant document chunks found."
            )


        # ----------------------------------
        # SEND CONTEXT TO GEMINI
        # ----------------------------------

        print("\nGenerating answer using Gemini...")

        answer = generate_answer(
            question,
            results
        )


        # ----------------------------------
        # DISPLAY RETRIEVED SOURCES
        # ----------------------------------

        sources = display_sources(results)

        print("\nAnswer generated successfully!")

        return answer, sources


    except Exception as e:

        print("\nERROR:", str(e))

        return (
            f"An error occurred: {str(e)}",
            "Check your terminal for details."
        )


# ==========================================
# GRADIO USER INTERFACE
# ==========================================

with gr.Blocks(
    title="My Digital Clone",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # My Digital Clone

        ### AI-powered Personal Knowledge Assistant

        Ask questions about my background, education,
        interests, skills and experiences.

        **Note:** This is an AI-generated digital clone,
        not the actual person.
        """
    )

    question = gr.Textbox(
        label="Ask Me Anything",
        placeholder="For example: What is my name?",
        lines=2
    )

    mode = gr.Radio(
        choices=[
            "BM25 Only",
            "Dense Only",
            "Hybrid"
        ],
        value="Hybrid",
        label="Select Retrieval Method"
    )

    submit = gr.Button(
        "Ask My Digital Clone",
        variant="primary"
    )

    gr.Markdown("## AI Generated Answer")

    answer = gr.Textbox(
        label="Answer",
        lines=6
    )

    gr.Markdown("## Retrieved Knowledge Base Chunks")

    sources = gr.Textbox(
        label="Sources, Chunk IDs and Scores",
        lines=15
    )

    submit.click(
        fn=chatbot,
        inputs=[
            question,
            mode
        ],
        outputs=[
            answer,
            sources
        ]
    )

    question.submit(
        fn=chatbot,
        inputs=[
            question,
            mode
        ],
        outputs=[
            answer,
            sources
        ]
    )


# ==========================================
# LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )

