"""
ICD-10 Chapter Mapping Utilities

Provides mapping from ICD-10 codes to their respective chapters and
chapter detection from clinical text queries.
"""

from typing import Optional
import re


# Complete ICD-10 Chapter Mapping (WHO Classification)
CHAPTER_MAPPING = {
    'A': 'I. Certain infectious and parasitic diseases',
    'B': 'I. Certain infectious and parasitic diseases',
    'C': 'II. Neoplasms',
    'D00-D49': 'II. Neoplasms',
    'D50-D89': 'III. Diseases of the blood and blood-forming organs',
    'E': 'IV. Endocrine, nutritional and metabolic diseases',
    'F': 'V. Mental, Behavioral and Neurodevelopmental disorders',
    'G': 'VI. Diseases of the nervous system',
    'H00-H59': 'VII. Diseases of the eye and adnexa',
    'H60-H95': 'VIII. Diseases of the ear and mastoid process',
    'I': 'IX. Diseases of the circulatory system',
    'J': 'X. Diseases of the respiratory system',
    'K': 'XI. Diseases of the digestive system',
    'L': 'XII. Diseases of the skin and subcutaneous tissue',
    'M': 'XIII. Diseases of the musculoskeletal system and connective tissue',
    'N': 'XIV. Diseases of the genitourinary system',
    'O': 'XV. Pregnancy, childbirth and the puerperium',
    'P': 'XVI. Certain conditions originating in the perinatal period',
    'Q': 'XVII. Congenital malformations, deformations and chromosomal abnormalities',
    'R': 'XVIII. Symptoms, signs and abnormal clinical and laboratory findings',
    'S': 'XIX. Injury, poisoning and certain other consequences of external causes',
    'T': 'XIX. Injury, poisoning and certain other consequences of external causes',
    'V': 'XX. External causes of morbidity',
    'W': 'XX. External causes of morbidity',
    'X': 'XX. External causes of morbidity',
    'Y': 'XX. External causes of morbidity',
    'Z': 'XXI. Factors influencing health status and contact with health services',
    'U': 'XXII. Codes for special purposes'
}

# Keyword-based chapter detection for hierarchical boosting
CHAPTER_KEYWORDS = {
    'I. Certain infectious and parasitic diseases': [
        'infection', 'infectious', 'bacteria', 'virus', 'parasite', 'sepsis',
        'tuberculosis', 'HIV', 'hepatitis', 'COVID'
    ],
    'II. Neoplasms': [
        'cancer', 'tumor', 'neoplasm', 'carcinoma', 'malignant', 'benign',
        'metastasis', 'lymphoma', 'leukemia', 'sarcoma'
    ],
    'III. Diseases of the blood and blood-forming organs': [
        'anemia', 'blood', 'coagulation', 'hemophilia', 'thrombocytopenia',
        'bleeding', 'clotting'
    ],
    'IV. Endocrine, nutritional and metabolic diseases': [
        'diabetes', 'thyroid', 'endocrine', 'metabolic', 'obesity',
        'malnutrition', 'vitamin deficiency', 'gout', 'hyperthyroid'
    ],
    'V. Mental, Behavioral and Neurodevelopmental disorders': [
        'depression', 'anxiety', 'psychosis', 'mental', 'psychiatric',
        'bipolar', 'schizophrenia', 'ADHD', 'autism', 'dementia'
    ],
    'VI. Diseases of the nervous system': [
        'neurological', 'epilepsy', 'seizure', 'parkinson', 'alzheimer',
        'migraine', 'neuropathy', 'multiple sclerosis', 'nerve'
    ],
    'VII. Diseases of the eye and adnexa': [
        'eye', 'vision', 'blindness', 'cataract', 'glaucoma', 'retina',
        'visual', 'optic', 'ocular'
    ],
    'VIII. Diseases of the ear and mastoid process': [
        'ear', 'hearing', 'deafness', 'tinnitus', 'otitis', 'mastoid',
        'auditory'
    ],
    'IX. Diseases of the circulatory system': [
        'heart', 'cardiac', 'hypertension', 'stroke', 'cardiovascular',
        'arrhythmia', 'myocardial infarction', 'coronary', 'vascular'
    ],
    'X. Diseases of the respiratory system': [
        'lung', 'respiratory', 'asthma', 'pneumonia', 'COPD', 'bronchitis',
        'pulmonary', 'breathing', 'cough'
    ],
    'XI. Diseases of the digestive system': [
        'stomach', 'intestinal', 'digestive', 'gastric', 'liver', 'cirrhosis',
        'ulcer', 'gallbladder', 'pancreas', 'bowel'
    ],
    'XII. Diseases of the skin and subcutaneous tissue': [
        'skin', 'dermatitis', 'rash', 'eczema', 'psoriasis', 'ulcer',
        'abscess', 'cellulitis', 'wound'
    ],
    'XIII. Diseases of the musculoskeletal system and connective tissue': [
        'bone', 'joint', 'arthritis', 'fracture', 'osteoporosis', 'back pain',
        'musculoskeletal', 'rheumatoid', 'muscle'
    ],
    'XIV. Diseases of the genitourinary system': [
        'kidney', 'renal', 'urinary', 'bladder', 'prostate', 'ureter',
        'nephritis', 'genital'
    ],
    'XV. Pregnancy, childbirth and the puerperium': [
        'pregnancy', 'pregnant', 'childbirth', 'labor', 'delivery',
        'obstetric', 'maternal', 'fetal', 'prenatal'
    ],
    'XVI. Certain conditions originating in the perinatal period': [
        'newborn', 'neonatal', 'perinatal', 'birth', 'premature'
    ],
    'XVII. Congenital malformations, deformations and chromosomal abnormalities': [
        'congenital', 'birth defect', 'chromosomal', 'malformation',
        'genetic', 'syndrome'
    ],
    'XVIII. Symptoms, signs and abnormal clinical and laboratory findings': [
        'symptom', 'abnormal', 'finding', 'pain', 'fever', 'fatigue',
        'dizziness', 'weakness'
    ],
    'XIX. Injury, poisoning and certain other consequences of external causes': [
        'injury', 'fracture', 'trauma', 'poisoning', 'burn', 'wound',
        'laceration', 'accident', 'fall'
    ],
    'XX. External causes of morbidity': [
        'accident', 'fall', 'collision', 'assault', 'suicide', 'external cause'
    ],
    'XXI. Factors influencing health status and contact with health services': [
        'screening', 'examination', 'history of', 'follow-up', 'counseling',
        'vaccination', 'prophylactic'
    ]
}


def get_chapter_from_code(code: str) -> str:
    """
    Extract ICD-10 chapter from code.

    Args:
        code: ICD-10 code (e.g., 'E11.621', 'E11621', 'C50', 'D52')

    Returns:
        Full chapter name (e.g., 'IV. Endocrine, nutritional and metabolic diseases')
    """
    if not code:
        return 'Unknown'

    # Remove dots for consistent processing
    code = code.replace('.', '').upper()

    # Get first letter
    first_letter = code[0]

    # Special handling for D codes (split by numeric range)
    if first_letter == 'D':
        try:
            numeric_part = int(code[1:3]) if len(code) >= 3 else 0
            if 0 <= numeric_part <= 49:
                return CHAPTER_MAPPING['D00-D49']
            elif 50 <= numeric_part <= 89:
                return CHAPTER_MAPPING['D50-D89']
        except (ValueError, IndexError):
            pass

    # Special handling for H codes (split by numeric range)
    if first_letter == 'H':
        try:
            numeric_part = int(code[1:3]) if len(code) >= 3 else 0
            if 0 <= numeric_part <= 59:
                return CHAPTER_MAPPING['H00-H59']
            elif 60 <= numeric_part <= 95:
                return CHAPTER_MAPPING['H60-H95']
        except (ValueError, IndexError):
            pass

    # Default: lookup by first letter
    return CHAPTER_MAPPING.get(first_letter, 'Unknown')


def detect_chapter_from_query(text: str) -> Optional[str]:
    """
    Detect implied ICD-10 chapter from clinical text query.

    Uses keyword matching to identify the most likely chapter
    for hierarchical boosting in retrieval.

    Args:
        text: Clinical text query

    Returns:
        Chapter name if detected, None otherwise
    """
    if not text:
        return None

    text_lower = text.lower()

    # Score each chapter by keyword matches
    chapter_scores = {}
    for chapter, keywords in CHAPTER_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            chapter_scores[chapter] = score

    # Return chapter with highest score, if any
    if chapter_scores:
        return max(chapter_scores, key=chapter_scores.get)

    return None


def get_all_chapters() -> list:
    """
    Get list of all unique ICD-10 chapters.

    Returns:
        List of chapter names
    """
    return sorted(set(CHAPTER_MAPPING.values()))
