import os

#  HARD SILENCE EVERYTHING
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"   # disables random processes on output 

from transformers import logging
logging.set_verbosity_error()

from sentence_transformers import SentenceTransformer

_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True # madatory if it should work in offline
)
def embed(texts):
    return _model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False # avoids giving long output 
    ).astype("float32") # faiss requires float32

if __name__ == "__main__":
    print(embed(["bad service", "great support"]).shape)
