"""
Helpers for parsing JSON returned by an LLM.

LLMs often wrap JSON in markdown fences, add prose around it, or leave
small syntax errors such as trailing commas. This module tolerates those.
"""

import json
import re
from typing import Any


def _strip_code_fence(text: str) -> str:
    """Return the contents of the first markdown code fence, if any."""
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _repair(text: str) -> str:
    """Fix common LLM JSON mistakes (trailing commas, missing commas)."""
    fixed = re.sub(r',\s*}', '}', text)
    fixed = re.sub(r',\s*]', ']', fixed)
    fixed = re.sub(r'}\s*{', '},{', fixed)
    fixed = re.sub(r'"\s*\n\s*"', '",\n"', fixed)
    return fixed


def parse_llm_json(text: str) -> Any:
    """
    Parse JSON from an LLM response.

    Tries, in order: the fenced/stripped text as-is, the outermost {...}
    object within it, and a repaired version of that object.

    Raises:
        json.JSONDecodeError: If no valid JSON can be recovered
    """
    candidate = _strip_code_fence(text)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        candidate = candidate[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return json.loads(_repair(candidate))
