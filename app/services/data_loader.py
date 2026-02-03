"""
Data Loader Service

Loads pre-computed ICD-10 dataset artifacts from preprocessing notebook.
"""

import os
import json
import pandas as pd
from loguru import logger
from typing import Optional


class DataLoader:
    """Service for loading pre-computed ICD-10 dataset artifacts."""

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize DataLoader.

        Args:
            cache_dir: Directory containing pre-computed artifacts
        """
        self.cache_dir = cache_dir
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Optional[dict] = None

    def load_enriched_dataset(self) -> pd.DataFrame:
        """
        Load pre-computed enriched dataset.

        The dataset should contain:
        - code: ICD-10 code
        - description: Code description
        - chapter: ICD-10 chapter name
        - synonyms: List of alternative descriptions

        Returns:
            DataFrame with enriched ICD-10 codes

        Raises:
            FileNotFoundError: If enriched dataset file doesn't exist
        """
        pkl_path = os.path.join(self.cache_dir, 'enriched_dataset.pkl')

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Enriched dataset not found at {pkl_path}. "
                "Please run notebooks/preprocessing.ipynb first to generate the required artifacts."
            )

        self.df = pd.read_pickle(pkl_path)
        logger.info(f"Loaded {len(self.df)} enriched ICD-10 codes from {pkl_path}")

        return self.df

    def load_metadata(self) -> dict:
        """
        Load preprocessing metadata.

        Metadata includes:
        - version: Dataset version
        - dataset: Dataset name
        - row_count: Number of codes
        - embedding_model: Model used for embeddings
        - embedding_dimension: Embedding dimension
        - created_at: Creation timestamp
        - chapter_count: Number of unique chapters

        Returns:
            Metadata dictionary

        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        meta_path = os.path.join(self.cache_dir, 'metadata.json')

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata not found at {meta_path}. "
                "Please run notebooks/preprocessing.ipynb first."
            )

        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded metadata: {self.metadata}")
        return self.metadata

    def is_loaded(self) -> bool:
        """
        Check if dataset is loaded.

        Returns:
            True if dataset is loaded, False otherwise
        """
        return self.df is not None

    def get_code_count(self) -> int:
        """
        Get total number of codes.

        Returns:
            Number of codes in dataset
        """
        return len(self.df) if self.is_loaded() else 0

    def get_chapter_count(self) -> int:
        """
        Get number of unique chapters.

        Returns:
            Number of unique chapters
        """
        return self.df['chapter'].nunique() if self.is_loaded() else 0
