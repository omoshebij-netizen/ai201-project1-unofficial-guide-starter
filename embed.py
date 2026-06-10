import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
CHUNKS_PATH = "data/chunks.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "towson_dining"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64  # embed and insert in batches to avoid memory issues


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(chunks: list[dict], model: SentenceTransformer) -> chromadb.Collection:
    """
    Embed all chunks and insert them into a ChromaDB collection.
    Deletes any existing collection with the same name first (clean rebuild).
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection for a clean rebuild
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance for similarity
    )

    total = len(chunks)
    print(f"Embedding {total} chunks with {EMBEDDING_MODEL}...")

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
            }
            for c in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        end = min(start + BATCH_SIZE, total)
        print(f"  Embedded and stored chunks {start+1}–{end} / {total}")

    print(f"\nVector store built. Collection '{COLLECTION_NAME}' has {collection.count()} items.")
    return collection


def test_retrieval(collection: chromadb.Collection, model: SentenceTransformer):
    """Run 3 test queries and print top results to verify retrieval quality."""
    test_queries = [
        "What time should I go to Newell to avoid the lunch rush?",
        "Which dining hall is best for students with celiac disease?",
        "Where can I get good coffee on campus?",
    ]

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST — Top 3 results per query")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        query_embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            print(f"\n  Result {i+1} | Source: {meta['source']} | Distance: {dist:.4f}")
            # Print first 200 chars of chunk for quick inspection
            preview = doc[:200].replace("\n", " ")
            print(f"  Preview: {preview}...")


if __name__ == "__main__":
    # Load chunks produced by ingest.py
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at '{CHUNKS_PATH}'. Run ingest.py first."
        )

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    # Load embedding model (downloads on first run, cached thereafter)
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

    # Build vector store
    collection = build_vector_store(chunks, model)

    # Test retrieval
    test_retrieval(collection, model)

    print("\nEmbed step complete. Run app.py to start the interface.")
