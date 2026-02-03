"""
Pydantic models for request/response validation.

User-specified output format for /suggest endpoint.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SuggestRequest(BaseModel):
    """Request model for /suggest endpoint."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Clinical text to map to ICD-10 codes"
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0-1) to return a result. Only codes above this threshold will be returned."
    )


class ValidationStatus(BaseModel):
    """Validation status from Chronicle (or mock service)."""

    valid: bool = Field(..., description="Whether the code is valid")
    message: str = Field(..., description="Validation message")


class CodeResult(BaseModel):
    """Single ICD-10 code result with metadata."""

    code: str = Field(..., description="ICD-10 code with dot notation (e.g., E11.621)")
    description: str = Field(..., description="Code description")
    chapter: str = Field(..., description="ICD-10 chapter name")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    explanation: str = Field(..., description="Explanation of why this code was suggested")
    synonyms: List[str] = Field(default_factory=list, description="Alternative descriptions")
    validation_status: ValidationStatus = Field(..., description="Validation result")


class SuggestResponse(BaseModel):
    """Response model for /suggest endpoint."""

    results: List[CodeResult] = Field(
        ...,
        description="List of suggested ICD-10 codes"
    )


# ============================================================================
# Enhanced Schemas for LLM-based Clinical Endpoint
# ============================================================================


class ClinicalSuggestRequest(BaseModel):
    """Enhanced request supporting complex clinical documents."""

    # Primary input (required)
    clinical_notes: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Clinical text: assessment, progress notes, discharge summary"
    )

    # Optional contextual inputs
    procedures: Optional[List[str]] = Field(
        default=None,
        description="Procedure descriptions"
    )
    labs: Optional[Dict[str, str]] = Field(
        default=None,
        description="Lab results (e.g., {'HbA1c': '9.4%'})"
    )
    medications: Optional[List[str]] = Field(
        default=None,
        description="Current medications"
    )
    imaging: Optional[List[str]] = Field(
        default=None,
        description="Imaging findings"
    )
    encounter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Demographics, encounter type, LOS"
    )

    # Configuration
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0-1) to return a result"
    )
    enable_llm_extraction: bool = Field(
        default=True,
        description="Use LLM for clinical entity extraction"
    )
    enable_llm_explanations: bool = Field(
        default=True,
        description="Use LLM for code explanations (enabled by default when LLM is configured)"
    )


class EnhancedCodeResult(BaseModel):
    """Enhanced code result with LLM-generated explanation."""

    code: str = Field(..., description="ICD-10 code with dot notation (e.g., E11.621)")
    description: str = Field(..., description="Code description")
    chapter: str = Field(..., description="ICD-10 chapter name")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    explanation: str = Field(..., description="LLM-generated reasoning for this code")
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Quotes from clinical text supporting this code"
    )
    synonyms: List[str] = Field(default_factory=list, description="Alternative descriptions")
    validation_status: ValidationStatus = Field(..., description="Validation result")
    requires_human_review: bool = Field(
        default=False,
        description="Flag indicating if human review is needed"
    )


class EnhancedSuggestResponse(BaseModel):
    """Enhanced response with documentation gaps."""

    results: List[EnhancedCodeResult] = Field(
        ...,
        description="List of suggested ICD-10 codes with explanations"
    )
    documentation_gaps: List[str] = Field(
        default_factory=list,
        description="Missing clinical specificity (laterality, depth, stages, etc.)"
    )
    extracted_entities: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Clinical entities extracted by LLM"
    )
