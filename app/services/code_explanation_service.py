"""
Code Explanation Service

Generates explanations for ICD-10 code suggestions using LLM.
"""

import json
from typing import Dict, Any
from loguru import logger

from app.services.llm_client import LLMClient
from app.services.clinical_extraction_service import ExtractedEntities


class CodeExplanationService:
    """Generate explanations for suggested ICD-10 codes."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize code explanation service.

        Args:
            llm_client: LLM client for API calls
        """
        self.llm_client = llm_client

    async def explain_code_match(
        self,
        code: str,
        code_description: str,
        extracted_entities: ExtractedEntities,
        original_text: str
    ) -> Dict[str, Any]:
        """
        Generate explanation for why a code matches the clinical input.

        Args:
            code: ICD-10 code (e.g., E11.621)
            code_description: Description of the code
            extracted_entities: Clinical entities extracted from input
            original_text: Original clinical text

        Returns:
            {
                "explanation": "This code matches because...",
                "confidence_adjustment": 0.95,  # Adjust RAG confidence
                "requires_review": false,
                "supporting_evidence": ["extracted from clinical text"]
            }
        """
        system_prompt = """You are a medical coding auditor. Explain why an ICD-10 code matches the clinical documentation.

Return JSON:
{
  "explanation": "Clear explanation of match",
  "confidence_adjustment": 0.0-1.0,
  "requires_review": true/false,
  "supporting_evidence": ["quote from clinical text"]
}

Guidelines:
- Confidence 0.9-1.0: Perfect match with all specificity
- Confidence 0.7-0.9: Good match, minor gaps
- Confidence 0.5-0.7: Reasonable match, needs review
- Confidence <0.5: Questionable match
- Requires review if: documentation gaps, ambiguous wording, multiple interpretations
- Supporting evidence should be direct quotes from the clinical text"""

        # Build clinical context
        clinical_context = []
        clinical_context.append(f"Primary Diagnosis: {extracted_entities.primary_diagnosis}")
        if extracted_entities.symptoms:
            clinical_context.append(f"Symptoms: {', '.join(extracted_entities.symptoms)}")
        if extracted_entities.anatomical_sites:
            clinical_context.append(f"Anatomical Sites: {', '.join(extracted_entities.anatomical_sites)}")
        if extracted_entities.laterality:
            clinical_context.append(f"Laterality: {extracted_entities.laterality}")
        if extracted_entities.severity:
            clinical_context.append(f"Severity: {extracted_entities.severity}")
        if extracted_entities.comorbidities:
            clinical_context.append(f"Comorbidities: {', '.join(extracted_entities.comorbidities)}")
        if extracted_entities.documentation_gaps:
            clinical_context.append(f"Documentation Gaps: {', '.join(extracted_entities.documentation_gaps)}")

        user_prompt = f"""Clinical Context:
{chr(10).join(clinical_context)}

Original Clinical Text:
{original_text}

Suggested ICD-10 Code:
{code}: {code_description}

Explain the match and rate confidence."""

        try:
            # Call LLM
            logger.debug(f"Generating explanation for code {code}...")
            response = await self.llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.0
            )

            # Parse JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            result = json.loads(response)

            # Validate fields
            if "explanation" not in result:
                result["explanation"] = f"Semantic match with {extracted_entities.primary_diagnosis}"
            if "confidence_adjustment" not in result:
                result["confidence_adjustment"] = 0.8
            if "requires_review" not in result:
                result["requires_review"] = False
            if "supporting_evidence" not in result:
                result["supporting_evidence"] = []

            # Ensure confidence is in range
            result["confidence_adjustment"] = max(0.0, min(1.0, result["confidence_adjustment"]))

            logger.debug(f"Generated explanation for {code}: {result['explanation'][:100]}...")

            return result

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback
            logger.error(f"Failed to parse explanation response: {e}")
            return {
                "explanation": f"Semantic match with clinical query",
                "confidence_adjustment": 0.8,
                "requires_review": False,
                "supporting_evidence": []
            }

        except Exception as e:
            logger.error(f"Error in code explanation: {e}")
            # Fallback
            return {
                "explanation": f"Match based on semantic similarity",
                "confidence_adjustment": 0.75,
                "requires_review": True,
                "supporting_evidence": []
            }
