import pandas as pd

from cluster.embeddings import embed
from cluster.clustering import add_clusters
from descriptive.vector_store import VectorStore
from cluster.rag import explain

# LOAD DATA
df = pd.read_excel(
    r"C:\Users\Rakesh\OneDrive\Documents\pragyan_rag_excel\clean_unique_reviews_1000_rows.xlsx"
)

texts = df["Review"].dropna().astype(str).tolist()


# PREPROCESS (RUN ONCE)
embeddings = embed(texts)
vector_store = VectorStore(embeddings)
df = add_clusters(df, embeddings, k=6)

# QUESTION TYPES
def descriptive_question(question):
    q_emb = embed([question])
    idx = vector_store.search(q_emb, k=6)
    context = [texts[i] for i in idx]
    return explain(context)

def common_issue_question():
    dominant_cluster = df["cluster"].value_counts().idxmax()
    context = (
        df[df["cluster"] == dominant_cluster]["Review"]
        .head(8)
        .astype(str)
        .tolist()
    )
    return explain(context)

# TEST
print("\n--- DESCRIPTIVE QUESTION ---")
print(descriptive_question("Why are customers unhappy with delivery?"))

print("\n--- MOST COMMON ISSUE (CLUSTER-BASED) ---")
print(common_issue_question())
