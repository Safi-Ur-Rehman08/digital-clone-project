
import json
from pathlib import Path

# Project folder location
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Chunk settings
CHUNK_SIZE = 250
OVERLAP = 40

def load_documents():
    documents = []

    # Find all Markdown files in data folder
    for file_path in DATA_DIR.glob("*.md"):

        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": file_path.name,
                "text": text
            })

    return documents


def create_chunks(documents):
    chunks = []
    chunk_number = 1

    for document in documents:

        words = document["text"].split()

        # Skip empty documents
        if not words:
            continue

        start = 0

        while start < len(words):

            end = start + CHUNK_SIZE
            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": f"chunk_{chunk_number}",
                "source": document["source"],
                "text": chunk_text,
                "word_count": len(chunk_words)
            })

            chunk_number += 1

            if end >= len(words):
                break

            # Move forward while keeping overlap
            start += CHUNK_SIZE - OVERLAP

    return chunks


def main():

    print("Loading personal documents...")

    documents = load_documents()

    if not documents:
        print("ERROR: No Markdown documents found in data folder.")
        return

    print(f"Documents loaded: {len(documents)}")

    print("\nCreating chunks...")

    chunks = create_chunks(documents)

    # Save chunks in JSON format
    output_file = BASE_DIR / "chunks.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunks saved to: {output_file}")

    print("\nDocument summary:")

    for document in documents:
        count = sum(
            1 for chunk in chunks
            if chunk["source"] == document["source"]
        )

        print(f"{document['source']}: {count} chunks")

    print("\nStage 4 Step 1 completed successfully!")


if __name__ == "__main__":
    main()
