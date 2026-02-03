"""
Application Configuration

Uses Pydantic Settings for type-safe configuration management.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Dataset Configuration
    dataset_name: str = "awacke1/ICD10-Clinical-Terminology"
    cache_dir: str = "data/cache"

    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval Configuration
    top_k: int = 5
    initial_candidates: int = 20
    min_confidence_score: float = 0.5  # Minimum confidence to return a result

    # API Configuration
    log_level: str = "INFO"

    # Hierarchical Boosting
    chapter_boost_factor: float = 1.2

    # LLM Configuration (OpenRouter)
    openrouter_api_key: Optional[str] = None
    llm_model: str = "anthropic/claude-haiku-4.5"
    llm_enabled: bool = False  # Set to True when API key is provided
    llm_temperature: float = 0.0  # Deterministic for clinical extraction

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
