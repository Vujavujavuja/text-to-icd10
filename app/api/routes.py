"""
API Routes

Defines all API endpoints for the ICD-10 Coding Assistant.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import Optional

from app.models.schemas import (
    SuggestRequest, SuggestResponse, CodeResult, ValidationStatus,
    ClinicalSuggestRequest, EnhancedSuggestResponse, EnhancedCodeResult
)
from app.services.retrieval_service import RetrievalService
from app.services.validation_service import ValidationService
from app.services.clinical_extraction_service import ClinicalExtractionService
from app.services.code_explanation_service import CodeExplanationService
from app.utils.code_formatter import CodeFormatter


# Global service instances (initialized in main.py)
retrieval_service: RetrievalService = None
validation_service: ValidationService = None
clinical_extraction_service: Optional[ClinicalExtractionService] = None
code_explanation_service: Optional[CodeExplanationService] = None
llm_client = None
llm_enabled: bool = False

router = APIRouter()


def init_routes(
    retrieval_svc: RetrievalService,
    validation_svc: ValidationService,
    clinical_extraction_svc: Optional[ClinicalExtractionService] = None,
    code_explanation_svc: Optional[CodeExplanationService] = None,
    llm_client_instance = None,
    llm_enabled_flag: bool = False
):
    """
    Initialize routes with service dependencies.

    Called from main.py after services are initialized.

    Args:
        retrieval_svc: Retrieval service instance
        validation_svc: Validation service instance
        clinical_extraction_svc: Clinical extraction service (optional)
        code_explanation_svc: Code explanation service (optional)
        llm_client_instance: LLM client instance (optional)
        llm_enabled_flag: Whether LLM services are enabled
    """
    global retrieval_service, validation_service, clinical_extraction_service, code_explanation_service, llm_client, llm_enabled
    retrieval_service = retrieval_svc
    validation_service = validation_svc
    clinical_extraction_service = clinical_extraction_svc
    code_explanation_service = code_explanation_svc
    llm_client = llm_client_instance
    llm_enabled = llm_enabled_flag


@router.post("/suggest", response_model=SuggestResponse)
async def suggest_codes(request: SuggestRequest):
    """
    Suggest ICD-10 codes for clinical text.

    Implements two-step RAG algorithm:
    1. Semantic retrieval using FAISS
    2. Hierarchical validation and re-ranking

    Args:
        request: SuggestRequest with clinical text

    Returns:
        SuggestResponse with list of suggested codes

    User-specified output format:
    {
      "results": [
        {
          "code": "E11.621",
          "description": "...",
          "chapter": "IV. Endocrine...",
          "confidence_score": 0.92,
          "explanation": "...",
          "synonyms": [...],
          "validation_status": {
            "valid": true,
            "message": "Valid 11th Edition Code."
          }
        }
      ]
    }
    """
    try:
        logger.info(f"Received suggest request: '{request.text}'")

        # Step 1 & 2: Semantic retrieval + hierarchical validation
        candidates = retrieval_service.retrieve_codes(request.text, top_k=5)

        # Step 3: Filter by minimum confidence score
        filtered_candidates = [
            c for c in candidates
            if c['confidence_score'] >= request.min_confidence
        ]

        logger.info(
            f"Filtered {len(candidates)} candidates to {len(filtered_candidates)} "
            f"with min_confidence={request.min_confidence}"
        )

        # Step 4: Format results and validate
        results = []
        for candidate in filtered_candidates:
            # Format code with dots (E11621 → E11.621)
            # Note: Dataset may already have dots, formatter handles both cases
            formatted_code = CodeFormatter.normalize_code(candidate['code'])

            # Validate against Chronicle (mock)
            validation = validation_service.validate_against_chronicle(formatted_code)

            results.append(
                CodeResult(
                    code=formatted_code,
                    description=candidate['description'],
                    chapter=candidate['chapter'],
                    confidence_score=round(candidate['confidence_score'], 2),
                    explanation=candidate['explanation'],
                    synonyms=candidate.get('synonyms', []),
                    validation_status=ValidationStatus(**validation)
                )
            )

        logger.info(f"Returning {len(results)} suggestions")
        return SuggestResponse(results=results)

    except Exception as e:
        logger.error(f"Error in suggest_codes: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.post("/suggest/clinical", response_model=EnhancedSuggestResponse)
async def suggest_codes_clinical(request: ClinicalSuggestRequest):
    """
    Enhanced ICD-10 code suggestion with LLM-based clinical text processing.

    Handles complex clinical documents:
    - Clinical notes (assessment, progress, discharge summary)
    - Labs, medications, imaging (contextual support)
    - Procedures (for ICD-10-PCS in future)

    Returns:
    - Suggested codes with explanations
    - Documentation gaps
    - Human review flags

    Args:
        request: ClinicalSuggestRequest with clinical notes and optional context

    Returns:
        EnhancedSuggestResponse with enhanced results and documentation gaps
    """
    try:
        logger.info(f"Clinical suggest request with LLM={request.enable_llm_extraction}")

        # CORRECT PIPELINE: RAG first, then ONE LLM call for everything

        # Step 1: RAG/Semantic search (fast, no LLM)
        logger.info(f"Running RAG search with query: {request.clinical_notes[:100]}...")
        candidates = retrieval_service.retrieve_codes(request.clinical_notes, top_k=5)

        filtered_candidates = [
            c for c in candidates
            if c['confidence_score'] >= request.min_confidence
        ]

        logger.info(f"RAG found {len(filtered_candidates)} candidates above min_confidence={request.min_confidence}")

        # Step 2: ONE LLM call to process clinical text + all candidates
        if request.enable_llm_explanations and llm_enabled and llm_client:
            logger.info("Calling LLM to analyze clinical text + all candidate codes...")
            try:
                # Build prompt with clinical text + all candidates
                candidates_text = "\n".join([
                    f"{i+1}. {c['code']}: {c['description']}"
                    for i, c in enumerate(filtered_candidates)
                ])

                system_prompt = """You are a medical coding expert. Analyze the clinical text and the candidate ICD-10 codes.

Return valid JSON with this structure:
{
  "clinical_entities": {
    "symptoms": ["list of symptoms"],
    "anatomical_sites": ["list of sites"],
    "laterality": "left|right|bilateral|null",
    "severity": "mild|moderate|severe|null",
    "chronicity": "acute|chronic|null"
  },
  "documentation_gaps": ["list of missing information"],
  "code_analysis": [
    {
      "code": "E11.621",
      "explanation": "Why this code matches",
      "confidence_adjustment": 0.9,
      "requires_review": false,
      "supporting_evidence": ["quotes from clinical text"]
    }
  ]
}"""

                user_prompt = f"""Clinical Text:
{request.clinical_notes}

Candidate ICD-10 Codes:
{candidates_text}

Analyze the clinical text, extract entities, identify documentation gaps, and explain each candidate code."""

                response = await llm_client.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    temperature=0.0
                )
                logger.info(f"LLM returned response, length: {len(response)} chars")

                # Parse LLM response
                import json
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()

                llm_result = json.loads(response)
                logger.info("LLM analysis completed successfully")

                # Extract data from LLM response
                clinical_entities = llm_result.get("clinical_entities", {})
                documentation_gaps = llm_result.get("documentation_gaps", [])
                code_analysis = {item["code"]: item for item in llm_result.get("code_analysis", [])}

            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                clinical_entities = {}
                documentation_gaps = ["LLM analysis failed - using basic results"]
                code_analysis = {}
        else:
            clinical_entities = {}
            documentation_gaps = []
            code_analysis = {}

        # Step 3: Format results
        results = []
        for candidate in filtered_candidates:
            formatted_code = CodeFormatter.normalize_code(candidate['code'])
            validation = validation_service.validate_against_chronicle(formatted_code)

            # Get LLM analysis for this code if available
            if formatted_code in code_analysis:
                analysis = code_analysis[formatted_code]
                explanation = analysis.get("explanation", candidate['explanation'])
                confidence_adjustment = analysis.get("confidence_adjustment", 1.0)
                requires_review = analysis.get("requires_review", False)
                supporting_evidence = analysis.get("supporting_evidence", [])
                adjusted_confidence = min(candidate['confidence_score'] * confidence_adjustment, 1.0)
            else:
                explanation = candidate['explanation']
                adjusted_confidence = candidate['confidence_score']
                requires_review = False
                supporting_evidence = []

            results.append(
                EnhancedCodeResult(
                    code=formatted_code,
                    description=candidate['description'],
                    chapter=candidate['chapter'],
                    confidence_score=round(adjusted_confidence, 2),
                    explanation=explanation,
                    supporting_evidence=supporting_evidence,
                    synonyms=candidate.get('synonyms', []),
                    validation_status=ValidationStatus(**validation),
                    requires_human_review=requires_review
                )
            )

        # Sort by confidence
        results.sort(key=lambda x: x.confidence_score, reverse=True)

        logger.info(f"Returning {len(results)} results")

        return EnhancedSuggestResponse(
            results=results,
            documentation_gaps=documentation_gaps,
            extracted_entities=clinical_entities
        )

    except Exception as e:
        logger.error(f"Error in clinical suggest: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing clinical request: {str(e)}"
        )
