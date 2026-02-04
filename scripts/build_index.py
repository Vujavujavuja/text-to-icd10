"""
Build FAISS index from enriched dataset.

Used during Docker build to regenerate the FAISS index
instead of relying on Git LFS.
"""

import os
import sys
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CACHE_DIR = "data/cache"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    pkl_path = os.path.join(CACHE_DIR, "enriched_dataset.pkl")
    index_path = os.path.join(CACHE_DIR, "icd10_index.faiss")

    print(f"Loading enriched dataset from {pkl_path}...")
    df = pd.read_pickle(pkl_path)
    print(f"Loaded {len(df)} codes")

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for {len(df)} descriptions...")
    embeddings = model.encode(
        df["description"].tolist(),
        show_progress_bar=True,
        batch_size=64,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    print(f"Index built: {index.ntotal} vectors")

    faiss.write_index(index, index_path)
    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"Saved FAISS index: {index_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
