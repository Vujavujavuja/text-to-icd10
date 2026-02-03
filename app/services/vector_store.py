"""
Vector Store Service

Manages FAISS index for semantic search.
Index is pre-built in preprocessing notebook.
"""

import os
import numpy as np
import pandas as pd
import faiss
from loguru import logger
from typing import List, Dict, Optional


class VectorStore:
    """Service for managing FAISS index and performing vector search."""

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize VectorStore.

        Args:
            cache_dir: Directory containing pre-built FAISS index
        """
        self.cache_dir = cache_dir
        self.index: Optional[faiss.Index] = None
        self.metadata_df: Optional[pd.DataFrame] = None

    def load_index(self):
        """
        Load pre-built FAISS index from preprocessing.

        Raises:
            FileNotFoundError: If FAISS index file doesn't exist
        """
        index_path = os.path.join(self.cache_dir, 'icd10_index.faiss')

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Please run notebooks/preprocessing.ipynb first to generate the index."
            )

        self.index = faiss.read_index(index_path)
        logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} vectors from {index_path}")

    def set_metadata(self, df: pd.DataFrame):
        """
        Set metadata DataFrame for result lookup.

        Args:
            df: DataFrame with code, description, chapter, synonyms columns
        """
        self.metadata_df = df
        logger.info(f"Set metadata for {len(df)} ICD-10 codes")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Search FAISS index and return results with metadata.

        Args:
            query_embedding: Query embedding vector (1 x embedding_dim)
            k: Number of results to return

        Returns:
            List of dictionaries with code, description, chapter, synonyms, distance, rank

        Raises:
            RuntimeError: If index or metadata is not loaded
        """
        if self.index is None:
            raise RuntimeError(
                "FAISS index not loaded. Call load_index() first."
            )

        if self.metadata_df is None:
            raise RuntimeError(
                "Metadata not set. Call set_metadata() first."
            )

        # FAISS search (expects float32)
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )

        # Format results with metadata
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            row = self.metadata_df.iloc[idx]

            results.append({
                'code': row['code'],
                'description': row['description'],
                'chapter': row['chapter'],
                'synonyms': row.get('synonyms', []) if isinstance(row.get('synonyms'), list) else [],
                'distance': float(distance),
                'rank': i + 1
            })

        return results

    def is_ready(self) -> bool:
        """
        Check if vector store is ready.

        Returns:
            True if index and metadata are loaded, False otherwise
        """
        return self.index is not None and self.metadata_df is not None

    def get_index_size(self) -> int:
        """
        Get number of vectors in index.

        Returns:
            Number of vectors in index, or 0 if not loaded
        """
        return self.index.ntotal if self.index is not None else 0
