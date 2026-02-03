"""
ICD-10 Code Formatting Utilities

Handles conversion between dotted and non-dotted ICD-10 code formats.
"""


class CodeFormatter:
    """Utility class for ICD-10 code formatting."""

    @staticmethod
    def remove_dots(code: str) -> str:
        """
        Remove dots from ICD-10 code for searching.

        Args:
            code: ICD-10 code (e.g., 'E11.621')

        Returns:
            Code without dots (e.g., 'E11621')
        """
        return code.replace('.', '')

    @staticmethod
    def add_dots(code: str) -> str:
        """
        Add dot notation to ICD-10 code for output.

        Standard format: dot after 3rd character (e.g., E11.621)

        Args:
            code: ICD-10 code without dots (e.g., 'E11621')

        Returns:
            Code with dots (e.g., 'E11.621')
        """
        # Remove any existing dots first
        code = code.replace('.', '')

        # Add dot after 3rd character if code is longer than 3 chars
        if len(code) <= 3:
            return code

        return f"{code[:3]}.{code[3:]}"

    @staticmethod
    def normalize_code(code: str) -> str:
        """
        Normalize ICD-10 code to standard format.

        Args:
            code: ICD-10 code in any format

        Returns:
            Normalized code with dots (e.g., 'E11.621')
        """
        # Remove dots, convert to uppercase, then add dots back
        code = code.replace('.', '').upper().strip()
        return CodeFormatter.add_dots(code)
