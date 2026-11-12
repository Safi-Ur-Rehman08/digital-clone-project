
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
# RRF constant
RRF_K = 60


# Step 1: Combine BM25 and Dense rankings
def reciprocal_rank_fusion(bm25_results, dense_results):

    # Store combined scores for every chunk
    fused_scores = {}

    # Store chunk information
    chunk_information = {}

    # Add BM25 ranking scores
    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["chunk_id"]

        rrf_score = 1 / (RRF_K + rank)

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0) + rrf_score
        )

        chunk_information[chunk_id] = result.copy()

    # Add Dense ranking scores
    for rank, result in enumerate(dense_results, start=1):

        chunk_id = result["chunk_id"]

        rrf_score = 1 / (RRF_K + rank)

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0) + rrf_score
        )

        chunk_information[chunk_id] = result.copy()

    # Sort chunks by combined RRF score
    ranked_chunks = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True
    )

    # Prepare final results
    final_results = []

    for chunk_id in ranked_chunks[:5]:

        result = chunk_information[chunk_id]

        final_results.append({
            "chunk_id": chunk_id,
            "source": result["source"],
            "text": result["text"],
            "rrf_score": fused_scores[chunk_id]
        })

    return final_results


# Step 2: Run Hybrid Retrieval
def retrieve_hybrid(question, chunks, bm25, model, embeddings):

    # Retrieve all chunks from BM25
    bm25_results = retrieve_bm25(
        question,
        chunks,
        bm25,
        top_k=len(chunks)
    )

    # Retrieve all chunks from Dense Retrieval
    dense_results = retrieve_dense(
        question,
        chunks,
        model,
        embeddings,
        top_k=len(chunks)
    )

    # Combine rankings using RRF
    hybrid_results = reciprocal_rank_fusion(
        bm25_results,
        dense_results
    )

    return hybrid_results


# Step 3: Test Hybrid Retrieval
if __name__ == "__main__":

    print("Loading document chunks...")

    chunks = load_chunks()

    if not chunks:
        print("No chunks found.")
        raise SystemExit

    print(f"Total chunks: {len(chunks)}")

    # Build BM25 index
    print("\nBuilding BM25 index...")

    bm25 = build_bm25(chunks)

    print("BM25 index ready!")

    # Load Dense Retrieval model
    model = load_model()

    # Load previously saved embeddings
    embeddings = load_embeddings()

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Chunk count and embedding count do not match. "
            "Regenerate embeddings.npy."
        )

    print("Saved embeddings loaded successfully!")

    # Ask question
    question = input(
        "\nEnter your question about yourself: "
    )

    # Run Hybrid Retrieval
    results = retrieve_hybrid(
        question,
        chunks,
        bm25,
        model,
        embeddings
    )

    # Display results
    print("\nHYBRID RETRIEVAL RESULTS")
    print("=" * 55)

    for result in results:

        print(f"\nChunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Text: {result['text'][:300]}...")
        print("-" * 55)
