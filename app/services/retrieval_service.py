"""
Retrieval Service

Implements two-step RAG algorithm:
1. Semantic retrieval using FAISS
2. Hierarchical validation and re-ranking
"""

from typing import List, Dict
from loguru import logger

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.utils.chapter_mapping import detect_chapter_from_query


class RetrievalService:
    """Service for two-step retrieval algorithm."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        chapter_boost_factor: float = 1.2
    ):
        """
        Initialize RetrievalService.

        Args:
            embedding_service: Service for encoding queries
            vector_store: Service for vector search
            chapter_boost_factor: Boost factor for chapter matches (default: 1.2 = 20% boost)
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chapter_boost_factor = chapter_boost_factor

    def semantic_search(self, query_text: str, top_k: int = 20) -> List[Dict]:
        """
        Step 1: Semantic retrieval using FAISS.

        Uses sentence embeddings to find top_k candidates based on
        semantic similarity.

        Args:
            query_text: Clinical text query
            top_k: Number of candidates to retrieve

        Returns:
            List of candidate dictionaries with initial confidence scores
        """
        # 1. Embed query
        query_embedding = self.embedding_service.encode_query(query_text)

        # 2. Search FAISS index
        candidates = self.vector_store.search(query_embedding, k=top_k)

        # 3. Calculate confidence scores (convert L2 distance to 0-1 score)
        for candidate in candidates:
            # L2 distance → similarity score
            # Using inverse distance: score = 1 / (1 + distance)
            # This maps: distance=0 → score=1.0, distance=∞ → score=0.0
            candidate['confidence_score'] = 1.0 / (1.0 + candidate['distance'])
            candidate['explanation'] = "Semantic match with query terms"

        logger.debug(f"Semantic search found {len(candidates)} candidates")
        return candidates

    def hierarchical_validation(
        self,
        query_text: str,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Step 2: Hierarchical validation and re-ranking.

        Detects implied ICD-10 chapter from query and boosts
        candidates that match the detected chapter.

        Args:
            query_text: Clinical text query
            candidates: List of candidates from semantic search

        Returns:
            Re-ranked candidates with updated confidence scores
        """
        # 1. Detect implied chapter from query
        implied_chapter = detect_chapter_from_query(query_text)

        if implied_chapter:
            logger.debug(f"Detected implied chapter: {implied_chapter}")
        else:
            logger.debug("No specific chapter detected from query")

        # 2. Boost candidates that match implied chapter
        for candidate in candidates:
            if implied_chapter and candidate['chapter'] == implied_chapter:
                # Apply hierarchical boost
                candidate['confidence_score'] *= self.chapter_boost_factor

                # Update explanation
                candidate['explanation'] += f" Matches {implied_chapter} hierarchy."

        # 3. Cap confidence scores at 1.0 and sort by score
        for candidate in candidates:
            candidate['confidence_score'] = min(candidate['confidence_score'], 1.0)

        # Sort by confidence score (descending)
        ranked_candidates = sorted(
            candidates,
            key=lambda x: x['confidence_score'],
            reverse=True
        )

        logger.debug(
            f"Hierarchical validation complete. "
            f"Top score: {ranked_candidates[0]['confidence_score']:.3f}"
        )

        return ranked_candidates

    def retrieve_codes(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Full two-step retrieval pipeline.

        Combines semantic search and hierarchical validation to
        retrieve the most relevant ICD-10 codes.

        Args:
            query_text: Clinical text query
            top_k: Number of final results to return

        Returns:
            List of top_k ICD-10 codes with metadata
        """
        logger.info(f"Retrieving codes for query: '{query_text}'")

        # Step 1: Semantic retrieval (fetch more for better re-ranking)
        initial_candidates = top_k * 2  # Fetch 2x for better re-ranking
        candidates = self.semantic_search(query_text, top_k=initial_candidates)

        # Step 2: Hierarchical validation and re-ranking
        ranked_candidates = self.hierarchical_validation(query_text, candidates)

        # Return top K after re-ranking
        final_results = ranked_candidates[:top_k]

        logger.info(
            f"Retrieved {len(final_results)} codes. "
            f"Top match: {final_results[0]['code']} "
            f"(score: {final_results[0]['confidence_score']:.3f})"
        )

        return final_results
