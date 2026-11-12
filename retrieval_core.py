
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# ==========================================
# LOAD DOCUMENT CHUNKS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "chunks.json"


def load_chunks():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"chunks.json not found at: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Support either a list or a dictionary containing chunks
    if isinstance(data, dict):
        data = data.get("chunks", [])

    if not isinstance(data, list):
        raise ValueError(
            "chunks.json must contain a list of chunks."
        )

    cleaned_chunks = []

    for index, item in enumerate(data):

        if not isinstance(item, dict):
            continue

        text = item.get("text", item.get("content", ""))

        if not text:
            continue

        cleaned_chunks.append({
            "chunk_id": item.get("chunk_id", index),
            "source": item.get(
                "source",
                item.get("filename", "Unknown")
            ),
            "text": str(text)
        })

    print(f"Loaded {len(cleaned_chunks)} chunks.")

    return cleaned_chunks


# ==========================================
# TOKENIZATION
# ==========================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        str(text).lower()
    )


# ==========================================
# BUILD BM25 INDEX
# ==========================================

def build_bm25(chunks):

    if not chunks:
        raise ValueError("Cannot build BM25 with no chunks.")

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# ==========================================
# BM25 RETRIEVAL
# ==========================================

def retrieve_bm25(question, chunks, bm25, top_k=5):

    if not question.strip():
        return []

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        chunk = chunks[index]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"],
            "score": float(scores[index])
        })

    return results
