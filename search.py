# search.py
import os
import chromadb
from sentence_transformers import SentenceTransformer

MIN_SCORE = 0.2
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, max_chars=500, overlap=50):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                end = start + max_chars
                chunks.append(para[start:end])
                start += max_chars - overlap
    return chunks

def load_and_chunk_notes(folder="notes"):
    all_chunks = []
    all_sources = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks))
    return all_chunks, all_sources

# --- Vector database ---
client = chromadb.Client()  # lives in memory, rebuilt each time the app starts
collection = client.get_or_create_collection(
    name="notes",
    metadata={"hnsw:space": "cosine"},  # compare by cosine, same as before
)

chunks, sources = load_and_chunk_notes()
collection.upsert(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embed_model.encode(chunks).tolist(),
    metadatas=[{"source": s} for s in sources],
)

def search(query, k=3):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    matches = []
    for i in range(len(results["documents"][0])):
        chunk = results["documents"][0][i]
        source = results["metadatas"][0][i]["source"]
        score = 1 - results["distances"][0][i]
        matches.append((source, chunk, score))

    return matches