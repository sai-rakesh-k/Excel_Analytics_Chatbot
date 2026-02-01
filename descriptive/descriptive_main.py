import pandas as pd

from embeddings import embed
from vector_store import VectorStore
from rag import explain

# LOAD DATA
df = pd.read_excel(
    r"C:\Users\Rakesh\OneDrive\Documents\pragyan_rag_excel\clean_unique_reviews_1000_rows.xlsx"
)

texts = df["Review"].dropna().astype(str).tolist()


# PREPROCESS (RUN ONCE)
embeddings = embed(texts)
vector_store = VectorStore(embeddings)


# QUESTION TYPES
def descriptive_question(question):
    q_emb = embed([question])
    idx = vector_store.search(q_emb, k=6)
    context = [texts[i] for i in idx]
    return explain(context)


# TEST
print("\n--- DESCRIPTIVE QUESTION ---")
print(descriptive_question("Why are customers unhappy with delivery?"))


