"""
ingest.py — Load policy documents, embed them, and store in ChromaDB.

Run once (or whenever documents change) before starting the API:
    python ingest.py

Embedding model: all-MiniLM-L6-v2
Embedding backend: ChromaDB's DefaultEmbeddingFunction (ONNX Runtime)
  - Uses the same all-MiniLM-L6-v2 model as sentence-transformers
  - Does not require PyTorch (avoids Windows long-path issues)
  - Downloads model weights automatically on first run

What it does:
1. Reads all .txt files from docs/
2. Splits each file into overlapping chunks (chunk_size=500 chars, overlap=50)
3. Encodes chunks with all-MiniLM-L6-v2 via ONNX Runtime
4. Stores chunks + embeddings + metadata in ChromaDB (persistent, on-disk)
"""

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from constants import DOCS_DIR, DB_DIR, COLLECTION_NAME


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def ingest():
    """Embed all policy documents and persist to ChromaDB."""
    print("[Ingest] Using embedding model: all-MiniLM-L6-v2 (via ChromaDB ONNX backend)")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Delete existing collection if present (re-ingest from scratch)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[Ingest] Dropped existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    # DefaultEmbeddingFunction uses all-MiniLM-L6-v2 via ONNX (no PyTorch needed)
    ef = DefaultEmbeddingFunction()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No .txt files found in {DOCS_DIR}")

    ids, documents, metadatas = [], [], []

    for doc_path in doc_files:
        text = doc_path.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        print(f"  {doc_path.name}: {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            ids.append(f"{doc_path.stem}_chunk_{i}")
            documents.append(chunk)
            metadatas.append({
                "source":      doc_path.name,
                "chunk_index": i,
            })

    # ChromaDB computes embeddings internally using the embedding_function
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"\n[Ingest] Done. {len(ids)} chunks from {len(doc_files)} documents stored.")
    print(f"         Collection: '{COLLECTION_NAME}' in {DB_DIR}")


if __name__ == "__main__":
    ingest()
