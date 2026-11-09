"""
Validation Service

Validates ICD-10-CM codes against the code set loaded at startup.
Returns validation status in user-specified format.
"""

from typing import Dict, Any, Iterable


def _normalize(code: str) -> str:
    """Normalize a code for lookup: no dots, uppercase, no whitespace."""
    return str(code).replace('.', '').upper().strip()


class ValidationService:
    """Validates ICD-10-CM codes against the loaded code set."""

    def __init__(self, codes: Iterable[str]):
        """
        Initialize ValidationService.

        Args:
            codes: All ICD-10-CM codes in the loaded dataset (dotted or not)
        """
        self.valid_codes = {_normalize(code) for code in codes if code}

        # A code that has more specific codes beneath it (e.g. E11 -> E11.621)
        # is a category header, not a complete (billable) code.
        self.codes_with_children = {
            code[:length]
            for code in self.valid_codes
            for length in range(3, len(code))
        }

    def validate_against_chronicle(self, code: str) -> Dict[str, Any]:
        """
        Validate an ICD-10-CM code against the loaded code set.

        Args:
            code: ICD-10 code to validate (e.g., 'E11.621')

        Returns:
            Dictionary with 'valid' (bool) and 'message' (str) keys
        """
        normalized = _normalize(code)

        if normalized not in self.valid_codes:
            return {
                'valid': False,
                'message': 'Code not found in the ICD-10-CM code set.'
            }

        if normalized in self.codes_with_children:
            return {
                'valid': False,
                'message': 'Category header; a more specific code is required.'
            }

        return {
            'valid': True,
            'message': 'Valid ICD-10-CM code.'
        }

    def validate_batch(self, codes: list) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple codes at once.

        Args:
            codes: List of ICD-10 codes

        Returns:
            Dictionary mapping code to validation result
        """
        return {
            code: self.validate_against_chronicle(code)
            for code in codes
        }
