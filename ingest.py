import os
import json
import re

# Configuration
DOCS_DIR = "docs"
OUTPUT_PATH = "data/chunks.json"
CHUNK_SIZE = 400       # characters — fits 2-4 sentences of review text
CHUNK_OVERLAP = 80     # characters — prevents key facts from being split at boundaries


# Cleaning helpers
def clean_document(raw_text: str) -> str:
    """
    Remove metadata headers, normalize whitespace, and strip boilerplate
    from raw document text. Keeps the substantive review/tip content.
    """
    lines = raw_text.split("\n")
    cleaned_lines = []

    skip_patterns = [
        r"^Source:",
        r"^URL:",
        r"^---+$",
        r"^Overall Rating:",
        r"^Dining Dollars Accepted:",
        r"^Tiger Bucks Accepted:",
        r"^Meal Swipes:",
        r"^\[Compiled from",
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        should_skip = any(re.match(pat, stripped) for pat in skip_patterns)
        if should_skip:
            continue

        cleaned_lines.append(stripped)

    # Collapse multiple blank lines into single blank lines
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# Chunking
def recursive_character_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into chunks of approximately chunk_size characters with overlap,
    preferring to split at paragraph and sentence boundaries.

    Strategy:
    1. First split on double newlines (paragraph boundaries)
    2. If a paragraph is still too large, split on sentence-ending punctuation
    3. Merge small pieces together up to chunk_size
    4. Apply overlap by including the tail of the previous chunk in each new chunk
    """
    # Step 1: split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Step 2: split any oversized paragraphs into sentences
    sentences = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            sentences.append(para)
        else:
            # Split on sentence boundaries
            parts = re.split(r'(?<=[.!?])\s+', para)
            for part in parts:
                if part.strip():
                    sentences.append(part.strip())

    # Step 3: merge sentences into chunks up to chunk_size
    chunks = []
    current_lines: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len + 1 <= chunk_size:
            current_lines.append(sent)
            current_len += sent_len + 1
        else:
            if current_lines:
                chunks.append(" ".join(current_lines))
            current_lines = [sent]
            current_len = sent_len

    if current_lines:
        chunks.append(" ".join(current_lines))

    # Step 4: apply overlap — prepend tail of previous chunk to each chunk
    if overlap <= 0 or len(chunks) <= 1:
        return [c for c in chunks if len(c.strip()) > 20]

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        # Take the last `overlap` characters of the previous chunk
        tail = prev[-overlap:].strip()
        # Start at the first space to avoid mid-word prepend
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1:]
        new_chunk = (tail + " " + chunks[i]).strip() if tail else chunks[i]
        overlapped.append(new_chunk)

    return [c for c in overlapped if len(c.strip()) > 20]


# Main pipeline
def load_and_chunk_documents(docs_dir: str) -> list[dict]:
    """
    Load all .txt files from docs_dir, clean them, chunk them,
    and return a list of chunk dicts with metadata.
    """
    all_chunks = []
    doc_files = sorted([f for f in os.listdir(docs_dir) if f.endswith(".txt")])

    print(f"Found {len(doc_files)} documents in '{docs_dir}/'")

    for doc_file in doc_files:
        filepath = os.path.join(docs_dir, doc_file)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned = clean_document(raw_text)
        chunks = recursive_character_split(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc_file}__chunk_{i:04d}",
                "source": doc_file,
                "chunk_index": i,
                "text": chunk_text,
            })

        print(f"  {doc_file}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def inspect_sample_chunks(chunks: list[dict], n: int = 5):
    """Print n representative chunks for manual inspection."""
    import random
    sample = random.sample(chunks, min(n, len(chunks)))
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS FOR INSPECTION")
    print("=" * 60)
    for i, chunk in enumerate(sample):
        print(f"\n--- Chunk {i+1} | Source: {chunk['source']} | Index: {chunk['chunk_index']} ---")
        print(chunk["text"])
        print(f"(length: {len(chunk['text'])} chars)")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    chunks = load_and_chunk_documents(DOCS_DIR)
    inspect_sample_chunks(chunks, n=5)

    # Save to disk for embed.py
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nChunks saved to {OUTPUT_PATH}")

    # Basic sanity checks
    lengths = [len(c["text"]) for c in chunks]
    print(f"\nChunk length stats:")
    print(f"  Min: {min(lengths)} chars")
    print(f"  Max: {max(lengths)} chars")
    print(f"  Avg: {sum(lengths) // len(lengths)} chars")

    sources = {}
    for c in chunks:
        sources[c["source"]] = sources.get(c["source"], 0) + 1
    print(f"\nChunks per document:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
