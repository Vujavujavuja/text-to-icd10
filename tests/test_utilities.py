"""
Quick test script to verify utility functions work correctly.

Run this before preprocessing to ensure utilities are working.
"""

from app.utils.chapter_mapping import get_chapter_from_code, detect_chapter_from_query
from app.utils.code_formatter import CodeFormatter


def test_chapter_mapping():
    """Test chapter mapping for various code types."""
    print("=" * 60)
    print("Testing Chapter Mapping")
    print("=" * 60)

    test_cases = [
        ("E11.621", "IV. Endocrine, nutritional and metabolic diseases"),
        ("E11621", "IV. Endocrine, nutritional and metabolic diseases"),
        ("C50", "II. Neoplasms"),
        ("D52", "III. Diseases of the blood and blood-forming organs"),
        ("D20", "II. Neoplasms"),
        ("H05", "VII. Diseases of the eye and adnexa"),
        ("H60", "VIII. Diseases of the ear and mastoid process"),
        ("I21", "IX. Diseases of the circulatory system"),
        ("J44", "X. Diseases of the respiratory system"),
        ("K21", "XI. Diseases of the digestive system"),
    ]

    all_passed = True
    for code, expected_chapter in test_cases:
        result = get_chapter_from_code(code)
        status = "[PASS]" if result == expected_chapter else "[FAIL]"
        print(f"{status} {code:10} -> {result}")
        if result != expected_chapter:
            print(f"   Expected: {expected_chapter}")
            all_passed = False

    print(f"\nChapter mapping: {'[PASS]' if all_passed else '[FAIL]'}")
    return all_passed


def test_chapter_detection():
    """Test chapter detection from query text."""
    print("\n" + "=" * 60)
    print("Testing Chapter Detection from Queries")
    print("=" * 60)

    test_cases = [
        ("patient with diabetes", "IV. Endocrine, nutritional and metabolic diseases"),
        ("lung cancer", "II. Neoplasms"),
        ("heart attack", "IX. Diseases of the circulatory system"),
        ("asthma exacerbation", "X. Diseases of the respiratory system"),
        ("depression and anxiety", "V. Mental, Behavioral and Neurodevelopmental disorders"),
    ]

    all_passed = True
    for query, expected_chapter in test_cases:
        result = detect_chapter_from_query(query)
        status = "[PASS]" if result == expected_chapter else "[WARN]"
        print(f"{status} '{query}'")
        print(f"   Detected: {result}")
        if result != expected_chapter:
            print(f"   Expected: {expected_chapter}")

    print(f"\nChapter detection: [COMPLETED] (heuristic-based)")
    return True


def test_code_formatter():
    """Test code formatting utilities."""
    print("\n" + "=" * 60)
    print("Testing Code Formatter")
    print("=" * 60)

    test_cases = [
        ("E11621", "E11.621", "Add dots"),
        ("E11.621", "E11621", "Remove dots"),
        ("C50", "C50", "Short code (no dots)"),
        ("E11.621", "E11.621", "Normalize (already formatted)"),
        ("e11621", "E11.621", "Normalize (lowercase)"),
    ]

    all_passed = True

    for input_code, expected, description in test_cases:
        if description == "Add dots":
            result = CodeFormatter.add_dots(input_code)
        elif description == "Remove dots":
            result = CodeFormatter.remove_dots(input_code)
        else:
            result = CodeFormatter.normalize_code(input_code)

        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"{status} {description:25} {input_code:10} -> {result:10} (expected: {expected})")
        if result != expected:
            all_passed = False

    print(f"\nCode formatter: {'[PASS]' if all_passed else '[FAIL]'}")
    return all_passed


if __name__ == "__main__":
    print("\nTesting ICD-10 Utility Functions\n")

    results = []
    results.append(test_chapter_mapping())
    results.append(test_chapter_detection())
    results.append(test_code_formatter())

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if all(results):
        print("[PASS] All utility tests passed!")
        print("\nYou can now run notebooks/preprocessing.ipynb")
    else:
        print("[FAIL] Some tests failed. Please review the output above.")

    print()
