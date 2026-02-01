import pandas as pd

from embeddings import embed
from clustering import add_clusters
from rag import explain

# LOAD DATA
df = pd.read_excel(
    r"C:\Users\Rakesh\OneDrive\Documents\pragyan_rag_excel\clean_unique_reviews_1000_rows.xlsx"
)

texts = df["Review"].dropna().astype(str).tolist()


# PREPROCESS (RUN ONCE)
embeddings = embed(texts)
df = add_clusters(df, embeddings, k=6)

# QUESTION TYPES


def common_issue_question():
    dominant_cluster = df["cluster"].value_counts().idxmax()
    context = (
        df[df["cluster"] == dominant_cluster]["Review"]
        .head(8)
        .astype(str)
        .tolist()
    )
    return explain(context)



print("\n--- MOST COMMON ISSUE (CLUSTER-BASED) ---")
print(common_issue_question())
