
import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

# Find the project folder
BASE_DIR = Path(__file__).parent

# File paths
CHUNKS_FILE = BASE_DIR / "chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"

# Model name
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Step 1: Load document chunks
def load_chunks():

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


# Step 2: Load the embedding model
def load_model():

    print("Loading sentence-transformer model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("Model loaded successfully!")

    return model


# Step 3: Generate and save embeddings
def generate_embeddings(chunks, model):

    print("\nGenerating document embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Save embeddings only once
    np.save(EMBEDDINGS_FILE, embeddings)

    print("\nEmbeddings generated successfully!")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved file: {EMBEDDINGS_FILE}")

    return embeddings


# Step 4: Load saved embeddings
def load_embeddings():

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "embeddings.npy not found. Generate embeddings first."
        )

    return np.load(EMBEDDINGS_FILE)


# Step 5: Search using Dense Retrieval
def retrieve_dense(question, chunks, model, embeddings, top_k=5):

    # Convert the question into an embedding
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    # Calculate cosine similarity using dot product
    similarities = np.dot(
        embeddings,
        question_embedding[0]
    )

    # Sort results from highest similarity to lowest
    ranked_indexes = np.argsort(similarities)[::-1]

    results = []

    for index in ranked_indexes[:top_k]:

        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "source": chunks[index]["source"],
            "text": chunks[index]["text"],
            "score": float(similarities[index])
        })

    return results


# Main program: Generate embeddings and test retrieval
if __name__ == "__main__":

    print("Loading document chunks...")

    chunks = load_chunks()

    if not chunks:
        print("No chunks found.")
        raise SystemExit

    print(f"Total chunks loaded: {len(chunks)}")

    # Load model
    model = load_model()

    # Generate embeddings if they do not exist
    if EMBEDDINGS_FILE.exists():

        print("\nExisting embeddings found.")

        embeddings = load_embeddings()

        if len(embeddings) != len(chunks):
            print("Chunk count changed. Regenerating embeddings...")
            embeddings = generate_embeddings(chunks, model)

    else:
        embeddings = generate_embeddings(chunks, model)

    # Ask a test question
    question = input(
        "\nEnter your question about yourself: "
    )

    results = retrieve_dense(
        question,
        chunks,
        model,
        embeddings,
        top_k=5
    )

    print("\nDENSE RETRIEVAL RESULTS")
    print("=" * 50)

    for result in results:

        print(f"\nChunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:300]}...")
        print("-" * 50)
