import os
import uuid
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

MIN_SCORE = 0.25
DUPLICATE_SCORE = 0.9


def _make_collection():
    """Open the memory store. Falls back to an in-memory store if the disk is not writable
    (for example on Streamlit Community Cloud)."""
    # On Streamlit Cloud the app folder (/mount/src) may not be writable, so use /tmp there.
    default_path = "/tmp/memory_db" if os.path.exists("/mount/src") else "memory_db"
    path = os.environ.get("MEMORY_DB_PATH", default_path)
    meta = {"hnsw:space": "cosine"}
    try:
        client = chromadb.PersistentClient(path=path)
        collection = client.get_or_create_collection("memories", metadata=meta)
        # Probe: make sure we can really write before trusting this store.
        collection.add(ids=["_probe"], documents=["probe"], embeddings=[[0.0] * 384])
        collection.delete(ids=["_probe"])
        return collection
    except Exception:
        client = chromadb.EphemeralClient()
        return client.get_or_create_collection("memories", metadata=meta)


collection = _make_collection()


def save_memory(text):
    embedding = model.encode(text).tolist()
    if collection.count() > 0:
        hit = collection.query(query_embeddings=[embedding], n_results=1)
        if 1 - hit["distances"][0][0] > DUPLICATE_SCORE:
            return "Already remembered."
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"saved_at": datetime.now().isoformat()}],
    )
    return "Saved."


def recall(query, k=3):
    if collection.count() == 0:
        return []
    embedding = model.encode(query).tolist()
    result = collection.query(query_embeddings=[embedding], n_results=min(k, collection.count()))
    memories = []
    for doc, distance in zip(result["documents"][0], result["distances"][0]):
        if 1 - distance >= MIN_SCORE:
            memories.append(doc)
    return memories


if __name__ == "__main__":
    print(save_memory("My exam is on December 5"))
    print(recall("when is my exam?"))
