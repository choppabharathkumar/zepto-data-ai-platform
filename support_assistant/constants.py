"""
constants.py — Shared configuration for the support_assistant module.

Embedding model: all-MiniLM-L6-v2
  Used via ChromaDB's DefaultEmbeddingFunction (ONNX Runtime backend).
  Same model weights as sentence-transformers, no PyTorch dependency.
"""

from pathlib import Path

MODULE_DIR = Path(__file__).parent
DOCS_DIR   = MODULE_DIR / "docs"
DB_DIR     = MODULE_DIR / "chroma_db"

# ChromaDB collection name
COLLECTION_NAME = "zepto_policies"

# Number of top document chunks to retrieve per query
TOP_K = 3
