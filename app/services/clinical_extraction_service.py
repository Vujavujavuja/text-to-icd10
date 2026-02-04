"""
Clinical Extraction Service

Extracts structured clinical entities from unstructured clinical text using LLM.
"""

import json
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger

from app.services.llm_client import LLMClient


class ExtractedEntities(BaseModel):
    """Structured clinical entities extracted by LLM."""

    primary_diagnosis: str
    symptoms: List[str] = []
    anatomical_sites: List[str] = []
    laterality: Optional[str] = None
    severity: Optional[str] = None
    chronicity: Optional[str] = None  # acute, chronic, recurrent
    comorbidities: List[str] = []
    procedures: List[str] = []  # "percutaneous coronary angioplasty"
    exclusions: List[str] = []  # "no osteomyelitis"
    enriched_query: str  # For RAG
    documentation_gaps: List[str] = []


class ClinicalExtractionService:
    """Extract structured clinical information from unstructured text."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize clinical extraction service.

        Args:
            llm_client: LLM client for API calls
        """
        self.llm_client = llm_client

    async def extract_from_clinical_note(
        self,
        clinical_text: str
    ) -> ExtractedEntities:
        """
        Extract clinical entities from unstructured clinical text.

        Uses LLM to parse:
        - Diagnosis/conditions
        - Symptoms/findings
        - Anatomical sites
        - Modifiers (laterality, severity, chronicity)
        - Documentation gaps

        Args:
            clinical_text: Unstructured clinical text (notes, discharge summary, etc.)

        Returns:
            Structured entities + enriched query for RAG
        """
        system_prompt = """You are a medical coding assistant. Extract clinical entities from the provided text and generate an enriched query for ICD-10 code search.

Return valid JSON with this structure:
{
  "primary_diagnosis": "Main diagnosis/condition",
  "symptoms": ["symptom1", "symptom2"],
  "anatomical_sites": ["site1", "site2"],
  "laterality": "left|right|bilateral|null",
  "severity": "mild|moderate|severe|null",
  "chronicity": "acute|chronic|recurrent|null",
  "comorbidities": ["condition1", "condition2"],
  "procedures": ["procedure performed or status post procedure"],
  "exclusions": ["excluded condition"],
  "enriched_query": "Expanded clinical search query",
  "documentation_gaps": ["Missing laterality", "Ulcer depth not specified"]
}

Guidelines:
- Extract ALL codeable diagnoses, comorbidities, and procedures
- Include procedure status (e.g. "status post percutaneous coronary angioplasty") in procedures
- Identify missing specificity as documentation gaps
- Generate enriched query combining diagnosis + modifiers
- Return empty arrays if no items found
- Use null for optional fields if not found"""

        user_prompt = f"""Clinical Text:
{clinical_text}

Extract clinical entities and generate enriched query for ICD-10 coding."""

        try:
            # Call LLM
            logger.info("Calling LLM for clinical entity extraction...")
            response = await self.llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.0
            )

            # Parse JSON response
            logger.debug(f"LLM response: {response[:500]}...")  # Log first 500 chars

            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            entities_dict = json.loads(response)
            entities = ExtractedEntities(**entities_dict)

            logger.info(f"Successfully extracted entities: {entities.primary_diagnosis}")
            logger.info(f"Enriched query: {entities.enriched_query}")
            logger.info(f"Documentation gaps: {len(entities.documentation_gaps)}")

            return entities

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: return minimal entities with original text as query
            logger.error(f"Failed to parse LLM response: {e}")
            logger.warning("Falling back to basic extraction")

            return ExtractedEntities(
                primary_diagnosis="Unknown",
                enriched_query=clinical_text,
                documentation_gaps=["LLM extraction failed - using raw text"]
            )

        except Exception as e:
            logger.error(f"Error in clinical extraction: {e}")
            # Fallback
            return ExtractedEntities(
                primary_diagnosis="Unknown",
                enriched_query=clinical_text,
                documentation_gaps=["Extraction error - using raw text"]
            )
