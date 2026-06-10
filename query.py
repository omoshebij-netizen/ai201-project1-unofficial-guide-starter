import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configuration
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "towson_dining"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
GROQ_MODEL = "llama-3.3-70b-versatile"

# System prompt — grounding is enforced here, not suggested
SYSTEM_PROMPT = """You are the Unofficial Towson University Dining Guide, a helpful assistant that answers student questions about campus dining.

CRITICAL RULES:
1. Answer ONLY using information from the provided source documents. Do not use any general knowledge or assumptions not supported by the documents.
2. If the provided documents do not contain enough information to answer the question, say exactly: "I don't have enough information in my sources to answer that. For official hours and menus, check towson.edu/dining."
3. Always end your response with a "Sources:" line listing the document filenames you drew from.
4. Be specific and practical — students are asking because they want actionable advice.
5. If documents provide conflicting information, note the conflict rather than picking one.

Format your response as:
[Your answer here, using only information from the documents]

Sources: [list the source document filenames, comma-separated]"""


# Singleton model/client (lazy initialization)
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_groq_client: Groq | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file. "
                "Get a free key at console.groq.com."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# Retrieval
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and retrieve the top-k most similar chunks from ChromaDB.
    Returns a list of dicts with keys: text, source, chunk_index, distance.
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })

    return chunks


# Generation
def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Document {i+1}: {chunk['source']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate(query: str, chunks: list[dict]) -> str:
    """
    Call the Groq LLM with retrieved context and return the generated answer.
    The system prompt enforces grounding — the model must answer only from
    the provided documents and cite its sources.
    """
    client = _get_groq_client()
    context = _build_context(chunks)

    user_message = f"""Here are the source documents to use for answering the question:

{context}

---

Student question: {query}

Remember: answer ONLY from the documents above. If the documents don't contain enough information, say so and direct the student to towson.edu/dining."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # Low temperature for factual, consistent answers
        max_tokens=600,
    )

    return response.choices[0].message.content


# Main ask() function
def ask(query: str) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, generate grounded answer.

    Returns a dict with:
      - answer: str — the LLM-generated answer
      - sources: list[str] — unique source filenames retrieved
      - chunks: list[dict] — the full retrieved chunk objects (for debugging)
    """
    chunks = retrieve(query, top_k=TOP_K)
    answer = generate(query, chunks)

    # Extract unique sources (programmatically guaranteed, regardless of LLM output)
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }


# CLI for quick testing
if __name__ == "__main__":
    print("Towson Dining Guide — Query Test")
    print("Type a question (or 'quit' to exit)\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = ask(query)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nRetrieved from: {', '.join(result['sources'])}")
        print(f"\nChunk distances: {[round(c['distance'], 3) for c in result['chunks']]}")
        print("\n" + "-" * 50 + "\n")
