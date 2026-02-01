import faiss
import numpy as np


class VectorStore:
    def __init__(self, embeddings):
        # Normalize embeddings
        faiss.normalize_L2(embeddings)

        # Create FAISS index (cosine similarity)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])

        # Add embeddings to index
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        # Normalize query
        faiss.normalize_L2(query_embedding)

        # Search top-k
        _, idx = self.index.search(query_embedding, k)
        return idx[0]


# sample test
if __name__ == "__main__":
    embeddings = np.random.rand(10, 4).astype("float32")

    store = VectorStore(embeddings)

    query_embedding = np.random.rand(1, 4).astype("float32")

    indices = store.search(query_embedding, k=3)
    print("Top 3 indices:", indices)
