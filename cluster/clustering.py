from sklearn.cluster import KMeans

def add_clusters(df, embeddings, k=6):
    """
    Creates cluster IDs for text.
    Used ONLY for 'most/common/major' questions.
    """
    km = KMeans(  # kmeans group similar things together
        n_clusters=k, # number of clusters
        random_state=42, #Gives same result every time for same input
        n_init="auto"   # try multiple starting points, then keep the best grouping
    )
    df["cluster"] = km.fit_predict(embeddings) # assign cluster IDs (0,...,k-1)
    return df

# sample code to test the function
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    #  Create dummy data (5 rows)
    df = pd.DataFrame({
        "text": [
            "bad service",
            "call dropped",
            "agent was polite",
            "billing issue",
            "quick resolution"
        ]
    })

    #  Fake embeddings (5 rows × 4 numbers)
    embeddings = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.31, 0.39],
        [0.9, 0.8, 0.7, 0.6],
        [0.11, 0.19, 0.29, 0.41],
        [0.88, 0.79, 0.69, 0.59],
    ])

    #  Run clustering
    df = add_clusters(df, embeddings, k=5)

    #  Print result
    print("\nFinal DataFrame:")
    print(df)
