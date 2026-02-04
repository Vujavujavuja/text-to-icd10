"""
FastAPI Application Entry Point

Main application with lifespan management for loading pre-computed artifacts.
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.services.data_loader import DataLoader
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retrieval_service import RetrievalService
from app.services.validation_service import ValidationService
from app.services.llm_client import LLMClient
from app.services.clinical_extraction_service import ClinicalExtractionService
from app.services.code_explanation_service import CodeExplanationService
from app.api import routes


# Configure logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

# Add file logging
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/icd10_api.log",
    rotation="1 day",
    retention="7 days",
    level=settings.log_level
)


# Global service instances
data_loader = DataLoader(cache_dir=settings.cache_dir)
embedding_service = EmbeddingService(model_name=settings.embedding_model)
vector_store = VectorStore(cache_dir=settings.cache_dir)
validation_service = ValidationService()

# LLM services (initialized if API key is provided)
llm_client = None
clinical_extraction_service = None
code_explanation_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.

    Startup: Load pre-computed artifacts
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting ICD-10 Coding Assistant API")
    logger.info("=" * 60)

    try:
        # Load pre-computed enriched dataset
        logger.info("Loading pre-computed ICD-10 dataset...")
        df = data_loader.load_enriched_dataset()
        logger.info(f"✅ Loaded {len(df)} ICD-10 codes")

        # Load metadata
        metadata = data_loader.load_metadata()
        logger.info(f"Dataset version: {metadata.get('version')}")
        logger.info(f"Embedding model: {metadata.get('embedding_model')}")
        logger.info(f"Chapter count: {metadata.get('chapter_count')}")

        # Load pre-built FAISS index
        logger.info("Loading pre-built FAISS index...")
        vector_store.load_index()
        logger.info(f"✅ Loaded FAISS index with {vector_store.get_index_size()} vectors")

        # Set metadata for vector store
        vector_store.set_metadata(df)

        # Load embedding model (for query encoding only)
        logger.info("Loading embedding model for query encoding...")
        embedding_service.load_model()
        logger.info(f"✅ Embedding dimension: {embedding_service.get_embedding_dimension()}")

        # Initialize retrieval service
        retrieval_service = RetrievalService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            chapter_boost_factor=settings.chapter_boost_factor
        )

        # Initialize LLM services (if enabled)
        global llm_client, clinical_extraction_service, code_explanation_service
        llm_enabled = False

        if settings.openrouter_api_key:
            logger.info("LLM services enabled - initializing...")
            logger.info(f"Using model: {settings.llm_model}")

            try:
                llm_client = LLMClient(
                    api_key=settings.openrouter_api_key,
                    model=settings.llm_model
                )
                clinical_extraction_service = ClinicalExtractionService(llm_client)
                code_explanation_service = CodeExplanationService(llm_client)
                llm_enabled = True
                logger.info("✅ LLM services initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize LLM services: {e}")
                logger.warning("Continuing without LLM enhancement")
        else:
            logger.info("LLM services disabled (no OPENROUTER_API_KEY)")
            logger.info("To enable LLM features, set OPENROUTER_API_KEY in .env")

        # Initialize routes with all services
        routes.init_routes(
            retrieval_svc=retrieval_service,
            validation_svc=validation_service,
            clinical_extraction_svc=clinical_extraction_service,
            code_explanation_svc=code_explanation_service,
            llm_client_instance=llm_client if llm_enabled else None,
            llm_enabled_flag=llm_enabled
        )

        logger.info("=" * 60)
        logger.info("✅ All services initialized successfully!")
        logger.info(f"LLM Enhancement: {'ENABLED' if llm_enabled else 'DISABLED'}")
        logger.info("API is ready to serve requests")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"❌ Failed to load pre-computed artifacts: {e}")
        logger.error(
            "Please run notebooks/preprocessing.ipynb first to generate "
            "the required artifacts (enriched_dataset.pkl, icd10_index.faiss, metadata.json)"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        sys.exit(1)

    yield

    # Shutdown
    logger.info("Shutting down ICD-10 Coding Assistant API...")


# Create FastAPI app
app = FastAPI(
    title="ICD-10 Coding Assistant",
    description="FastAPI backend for mapping clinical text to ICD-10 codes using hierarchy-aware RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(routes.router, tags=["ICD-10 Suggestions"])


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and readiness information.
    """
    return {
        "status": "healthy",
        "dataset_loaded": data_loader.is_loaded(),
        "faiss_ready": vector_store.is_ready(),
        "embedding_service_ready": embedding_service.is_ready(),
        "code_count": data_loader.get_code_count(),
        "chapter_count": data_loader.get_chapter_count(),
        "index_size": vector_store.get_index_size()
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ICD-10 Coding Assistant API",
        "version": "1.0.0",
        "description": "Hierarchy-aware RAG system for mapping clinical text to ICD-10 codes",
        "endpoints": {
            "suggest": "POST /suggest - Get ICD-10 code suggestions",
            "health": "GET /health - Check service health",
            "docs": "GET /docs - OpenAPI documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level=settings.log_level.lower()
    )
