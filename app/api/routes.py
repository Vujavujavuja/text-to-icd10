"""
API Routes

Defines all API endpoints for the ICD-10 Coding Assistant.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import Any, Dict, List, Optional

from app.config import settings
from app.models.schemas import (
    SuggestRequest, SuggestResponse, CodeResult, ValidationStatus,
    ClinicalSuggestRequest, EnhancedSuggestResponse, EnhancedCodeResult
)
from app.services.retrieval_service import RetrievalService
from app.services.validation_service import ValidationService
from app.services.clinical_extraction_service import ClinicalExtractionService
from app.services.code_explanation_service import CodeExplanationService
from app.utils.code_formatter import CodeFormatter
from app.utils.llm_json import parse_llm_json


# Upper bound on the LLM's confidence multiplier for a candidate code
MAX_CONFIDENCE_ADJUSTMENT = 1.5


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


def build_clinical_context(request: ClinicalSuggestRequest) -> str:
    """
    Combine clinical notes with the optional structured inputs
    (procedures, labs, medications, imaging, encounter metadata)
    into one text block for the LLM.
    """
    sections = [request.clinical_notes.strip()]

    def add_list(title: str, items: Optional[List[str]]):
        items = [str(i).strip() for i in (items or []) if str(i).strip()]
        if items:
            sections.append(f"{title}:\n" + "\n".join(f"- {i}" for i in items))

    def add_dict(title: str, values: Optional[Dict[str, Any]]):
        if values:
            sections.append(f"{title}:\n" + "\n".join(f"- {k}: {v}" for k, v in values.items()))

    add_list("Procedures", request.procedures)
    add_dict("Labs", request.labs)
    add_list("Medications", request.medications)
    add_list("Imaging", request.imaging)
    add_dict("Encounter", request.encounter_metadata)

    return "\n\n".join(sections)


def _coerce_adjustment(value: Any) -> float:
    """Parse the LLM's confidence multiplier, defaulting to 1.0 (no change)."""
    try:
        adjustment = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(adjustment, MAX_CONFIDENCE_ADJUSTMENT))


def _index_code_analysis(items: Any) -> Dict[str, Dict[str, Any]]:
    """Key the LLM's per-code analysis by normalized code, skipping malformed items."""
    analysis = {}
    if not isinstance(items, list):
        return analysis
    for item in items:
        if isinstance(item, dict) and item.get("code"):
            analysis[CodeFormatter.normalize_code(str(item["code"]))] = item
    return analysis


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
            "message": "Valid ICD-10-CM code."
          }
        }
      ]
    }
    """
    try:
        logger.info(f"Received suggest request: '{request.text}'")

        # Step 1 & 2: Semantic retrieval + hierarchical validation
        candidates = retrieval_service.retrieve_codes(request.text, top_k=settings.top_k)

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

            # Validate against the loaded ICD-10-CM code set
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
        logger.info(f"Clinical suggest request with LLM extraction={request.enable_llm_extraction}")

        # PIPELINE: Extract entities → RAG with focused queries → LLM explains

        # Notes plus optional procedures/labs/medications/imaging/encounter data
        clinical_context = build_clinical_context(request)

        # Step 1: LLM extraction to build focused search queries
        extracted_entities = None
        if request.enable_llm_extraction and llm_enabled and clinical_extraction_service:
            logger.info("Step 1: LLM entity extraction from clinical text...")
            try:
                extracted_entities = await clinical_extraction_service.extract_from_clinical_note(
                    clinical_context
                )
                logger.info(f"Extracted primary diagnosis: {extracted_entities.primary_diagnosis}")
                logger.info(f"Comorbidities: {extracted_entities.comorbidities}")
                logger.info(f"Enriched query: {extracted_entities.enriched_query}")
            except Exception as e:
                logger.error(f"LLM extraction failed, falling back to raw text: {e}")
                extracted_entities = None

        # Step 2: Build search queries from extracted entities
        if extracted_entities and extracted_entities.primary_diagnosis != "Unknown":
            search_queries = [extracted_entities.primary_diagnosis]
            search_queries.extend(extracted_entities.comorbidities)
            search_queries.extend(extracted_entities.procedures)
            logger.info(f"Step 2: Running RAG with {len(search_queries)} focused queries: {search_queries}")
        else:
            search_queries = [request.clinical_notes]
            logger.info("Step 2: Running RAG with raw clinical text (no extraction)")

        # Step 3: Run RAG for each query (top 3 per query for diversity), merge results
        all_candidates = {}
        per_query_top_k = 3
        for query in search_queries:
            if not query:
                continue
            candidates = retrieval_service.retrieve_codes(query, top_k=per_query_top_k)
            for c in candidates:
                code = c['code']
                if code not in all_candidates or c['confidence_score'] > all_candidates[code]['confidence_score']:
                    all_candidates[code] = c

        # Filter by min confidence, sort, take top results
        max_results = max(5, len(search_queries) * 2)
        filtered_candidates = sorted(
            [c for c in all_candidates.values() if c['confidence_score'] >= request.min_confidence],
            key=lambda x: x['confidence_score'],
            reverse=True
        )[:max_results]

        logger.info(f"RAG found {len(filtered_candidates)} candidates from {len(all_candidates)} unique codes")

        # Step 4: LLM analysis of candidates
        clinical_entities = {}
        documentation_gaps = []
        code_analysis = {}

        if request.enable_llm_explanations and llm_enabled and llm_client and filtered_candidates:
            logger.info("Step 4: LLM analysis of candidate codes...")
            try:
                candidates_text = "\n".join([
                    f"{i+1}. {CodeFormatter.normalize_code(c['code'])}: {c['description']}"
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
      "explanation": "Why this code matches or does not match the clinical text",
      "confidence_adjustment": 0.9,
      "requires_review": false,
      "supporting_evidence": ["quotes from clinical text"]
    }
  ]
}

Important: Analyze EVERY candidate code, using the exact code string from the candidate list.
confidence_adjustment is a multiplier on the retrieval score between 0.0 and 1.5:
1.0 keeps the score, above 1.0 for codes the text clearly supports, below 1.0 for incorrect or irrelevant codes."""

                user_prompt = f"""Clinical Text:
{clinical_context}

Candidate ICD-10 Codes:
{candidates_text}

Analyze the clinical text, extract entities, identify documentation gaps, and explain each candidate code."""

                response = await llm_client.generate(
                    prompt=user_prompt,
                    system=system_prompt
                )
                logger.info(f"LLM analysis response length: {len(response)} chars")

                llm_result = parse_llm_json(response)
                if not isinstance(llm_result, dict):
                    raise ValueError("LLM analysis response is not a JSON object")
                logger.info("LLM analysis completed successfully")

                entities = llm_result.get("clinical_entities")
                clinical_entities = entities if isinstance(entities, dict) else {}
                gaps = llm_result.get("documentation_gaps")
                documentation_gaps = [str(g) for g in gaps if g] if isinstance(gaps, list) else []
                code_analysis = _index_code_analysis(llm_result.get("code_analysis"))

            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                documentation_gaps = ["LLM analysis failed - using basic results"]

        # Use extraction entities if no LLM analysis entities
        if not clinical_entities and extracted_entities:
            clinical_entities = {
                "symptoms": extracted_entities.symptoms,
                "anatomical_sites": extracted_entities.anatomical_sites,
                "laterality": extracted_entities.laterality,
                "severity": extracted_entities.severity,
                "chronicity": extracted_entities.chronicity,
            }
        if not documentation_gaps and extracted_entities:
            documentation_gaps = extracted_entities.documentation_gaps

        # Step 5: Format results
        results = []
        for candidate in filtered_candidates:
            formatted_code = CodeFormatter.normalize_code(candidate['code'])
            validation = validation_service.validate_against_chronicle(formatted_code)

            if formatted_code in code_analysis:
                analysis = code_analysis[formatted_code]
                explanation = str(analysis.get("explanation") or candidate['explanation'])
                confidence_adjustment = _coerce_adjustment(analysis.get("confidence_adjustment", 1.0))
                requires_review = bool(analysis.get("requires_review", False))
                evidence = analysis.get("supporting_evidence")
                supporting_evidence = [str(e) for e in evidence if e] if isinstance(evidence, list) else []
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
