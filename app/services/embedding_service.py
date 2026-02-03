"""
Embedding Service

Handles query encoding using sentence-transformers.
Dataset embeddings are pre-computed in preprocessing notebook.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import Optional


class EmbeddingService:
    """Service for encoding text queries into embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize EmbeddingService.

        Args:
            model_name: Name of sentence-transformers model
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None

    def load_model(self):
        """
        Load sentence-transformers model.

        This is called during FastAPI startup.
        """
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"✅ Embedding model loaded successfully")

    def encode_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector (1 x embedding_dim)

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError(
                "Embedding model not loaded. Call load_model() first."
            )

        # Encode returns (1, embedding_dim) array for single query
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding

    def is_ready(self) -> bool:
        """
        Check if model is loaded and ready.

        Returns:
            True if model is ready, False otherwise
        """
        return self.model is not None

    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Get dimension from model configuration
        return self.model.get_sentence_embedding_dimension()
