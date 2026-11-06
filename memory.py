import uuid
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

# Same embedding model as search.py
model = SentenceTransformer("all-MiniLM-L6-v2")

# PersistentClient writes to the "memory_db" folder, so memories survive restarts.
client = chromadb.PersistentClient(path="memory_db")
collection = client.get_or_create_collection(
    "memories", metadata={"hnsw:space": "cosine"}
)

MIN_SCORE = 0.25      # below this similarity, a memory is not returned
DUPLICATE_SCORE = 0.9  # above this, the fact is treated as already saved


def save_memory(text):
    embedding = model.encode(text).tolist()

    # Skip near-duplicates so "my exam is Dec 5" is not stored ten times.
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
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, collection.count()),
    )

    memories = []
    for doc, distance in zip(result["documents"][0], result["distances"][0]):
        if 1 - distance >= MIN_SCORE:  # cosine distance -> similarity
            memories.append(doc)
    return memories


if __name__ == "__main__":
    # Run once as-is. Then comment out the save line and run again:
    # if recall still finds the exam date, memory survives across runs.
    #print(save_memory("My exam is on December 5"))
    print(recall("when is my exam?"))
