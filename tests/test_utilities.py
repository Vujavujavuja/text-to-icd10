"""
Unit tests for utility functions and the validation service.

These need no model, index or server: run with `pytest tests/test_utilities.py`.
"""

import json

import pytest

from app.services.validation_service import ValidationService
from app.utils.chapter_mapping import get_chapter_from_code, detect_chapter_from_query
from app.utils.code_formatter import CodeFormatter
from app.utils.llm_json import parse_llm_json


ENDOCRINE = "IV. Endocrine, nutritional and metabolic diseases"
NEOPLASMS = "II. Neoplasms"
EAR = "VIII. Diseases of the ear and mastoid process"
CIRCULATORY = "IX. Diseases of the circulatory system"
RESPIRATORY = "X. Diseases of the respiratory system"
MENTAL = "V. Mental, Behavioral and Neurodevelopmental disorders"
INFECTIOUS = "I. Certain infectious and parasitic diseases"
NEURO_DEV = MENTAL


@pytest.mark.parametrize("code, expected", [
    ("E11.621", ENDOCRINE),
    ("E11621", ENDOCRINE),
    ("C50", NEOPLASMS),
    ("D52", "III. Diseases of the blood and blood-forming organs"),
    ("D20", NEOPLASMS),
    ("H05", "VII. Diseases of the eye and adnexa"),
    ("H60", EAR),
    ("I21", CIRCULATORY),
    ("J44", RESPIRATORY),
    ("K21", "XI. Diseases of the digestive system"),
])
def test_chapter_from_code(code, expected):
    assert get_chapter_from_code(code) == expected


@pytest.mark.parametrize("query, expected", [
    ("patient with diabetes", ENDOCRINE),
    ("lung cancer", NEOPLASMS),
    ("heart attack", CIRCULATORY),
    ("heart failure", CIRCULATORY),
    ("asthma exacerbation", RESPIRATORY),
    ("depression and anxiety", MENTAL),
    # Uppercase keywords used to never match lowercased text
    ("Patient is HIV positive", INFECTIOUS),
    ("Acute exacerbation of COPD", RESPIRATORY),
    ("child with ADHD", NEURO_DEV),
    # Plurals still match
    ("multiple tumors", NEOPLASMS),
])
def test_chapter_detection(query, expected):
    assert detect_chapter_from_query(query) == expected


@pytest.mark.parametrize("query", [
    "67-year-old male",       # "ear" inside "year"
    "early morning symptoms is clear",  # "ear" inside "early"/"clear"
])
def test_chapter_detection_ignores_substrings(query):
    assert detect_chapter_from_query(query) != EAR


def test_labor_does_not_match_laboratory():
    assert detect_chapter_from_query("laboratory results reviewed") is None


@pytest.mark.parametrize("func, code, expected", [
    (CodeFormatter.add_dots, "E11621", "E11.621"),
    (CodeFormatter.remove_dots, "E11.621", "E11621"),
    (CodeFormatter.normalize_code, "C50", "C50"),
    (CodeFormatter.normalize_code, "E11.621", "E11.621"),
    (CodeFormatter.normalize_code, "e11621", "E11.621"),
])
def test_code_formatter(func, code, expected):
    assert func(code) == expected


@pytest.fixture
def validator():
    return ValidationService(["E11621", "E1165", "E119", "I2109", "R51"])


@pytest.mark.parametrize("code", ["E11.621", "E11621", "e11.65", "R51"])
def test_validation_accepts_known_codes(validator, code):
    assert validator.validate_against_chronicle(code) == {
        "valid": True, "message": "Valid ICD-10-CM code."
    }


def test_validation_is_deterministic(validator):
    results = {validator.validate_against_chronicle("E11.621")["valid"] for _ in range(200)}
    assert results == {True}


def test_validation_rejects_unknown_code(validator):
    result = validator.validate_against_chronicle("Z99.999")
    assert result["valid"] is False
    assert "not found" in result["message"]


def test_validation_flags_category_headers():
    validator = ValidationService(["E11", "E11621", "E1165"])
    result = validator.validate_against_chronicle("E11")
    assert result["valid"] is False
    assert "more specific" in result["message"]


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    'Here is the result:\n{"a": 1}\nHope that helps!',
    '{"a": 1,}',
])
def test_parse_llm_json(text):
    assert parse_llm_json(text) == {"a": 1}


def test_parse_llm_json_repairs_trailing_commas_in_lists():
    assert parse_llm_json('{"gaps": ["x", "y",],}') == {"gaps": ["x", "y"]}


def test_parse_llm_json_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("no json here")
