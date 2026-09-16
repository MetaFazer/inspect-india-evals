"""
Single source of truth for dataset composition figures.

These numbers are read directly from the data files under india_evals/*/datasets/
(see tests/test_dataset_counts.py, which asserts every dataset against this
module). If a dataset file changes, that test will fail — update the figures
here AND the README deliberately, rather than letting them drift apart.

Do not hand-edit these numbers without re-deriving them from the data.
"""

# ── multilingual/datasets/mmlu_translated.csv ──────────────────────────────────
MULTILINGUAL = {
    "total_rows": 2272,
    "n_languages": 16,
    "questions_per_language": 142,
    "n_subjects": 29,
    # ISO-ish codes as they appear in the "language" column
    "language_codes": [
        "as", "bn", "en", "gu", "hi", "kn", "kok", "ml",
        "mni", "mr", "ne", "or", "pa", "ta", "te", "ur",
    ],
    "n_indian_languages": 15,  # all codes above except "en"
}

# ── safeguards/datasets/safety.csv ─────────────────────────────────────────────
SAFETY = {
    "total_rows": 200,
    "n_language_columns": 5,  # original_prompt(English), Hindi, Tamil, Telugu, Bengali
    "n_indian_languages": 4,
    "total_samples": 1000,  # 200 rows x 5 language columns
    "categories": {
        "cybercrime": 34,
        "deepfake": 34,
        "communal_violence": 33,
        "financial_fraud": 33,
        "misinformation": 33,
        "harassment": 33,
    },
    "n_categories": 6,
    "risk_levels": {"High": 200},
}

# ── safeguards/datasets/jailbreak_multilingual.csv ─────────────────────────────
JAILBREAK = {
    "total_rows": 70,
    "n_languages": 7,  # 6 Indian + English
    "n_indian_languages": 6,
    "language_codes": ["bn", "en", "gu", "hi", "mr", "ta", "te"],
    "n_turns": 5,
    "attack_types": {
        "roleplay": 7, "fiction": 7, "authority": 7, "emotional": 7,
        "research": 7, "educational": 7, "urgency": 7, "incident_review": 7,
        "translation": 7, "reward_hacking": 7,
    },
    "n_attack_types": 10,
}

# ── dpi_safety/datasets/dpi_dataset.csv ────────────────────────────────────────
DPI = {
    "total_rows": 150,
    "languages": {"English": 150},
    "n_languages": 1,
    "categories": {
        "Aadhaar Privacy": 40,
        "UPI Fraud": 35,
        "Bhashini Disinformation": 25,
        "Digital Lending": 25,
        "DigiLocker": 15,
        "ABDM Health Data": 10,
    },
    "n_categories": 6,
    "risk_levels": {"High": 78, "Low": 72},
    # There is no Medium risk level in the data, despite paper/older docs.
}

# ── cultural_knowledge/datasets/Cultural_knowledge_rubric_dataset.json ─────────
CULTURAL_KNOWLEDGE = {
    "total_rows": 300,
    "domains": {
        "Indian Constitution": 60,
        "Indian Healthcare": 60,
        "Agriculture and MSP": 60,
        "Indian History": 60,
        "State Governance": 60,
    },
    "n_domains": 5,
    "rubric_criteria_per_entry": 4,
}

# ── bias/datasets/*.csv (BharatBBQ) ────────────────────────────────────────────
# Audited 2026-09 against the actual CSV files. The paper claims 13 categories
# (Caste, Religion, Gender Identity, Age, Region, Disability, Socioeconomic
# Status, Nationality, Physical Appearance, Linguistic Group,
# Caste-Adjacent Occupation, Urban-Rural Identity, Tribal Community, plus
# intersectional files). The data on disk does NOT match that list: it has
# 13 files, but "Sexual_orientation" appears (not claimed by the paper) while
# "Linguistic Group", "Caste-Adjacent Occupation", "Urban-Rural Identity", and
# "Tribal Community" are absent. See README "Relation to the preprint".
BHARATBBQ = {
    "n_files": 13,
    "total_rows": 54048,
    "files": {
        "Age_examples-00000-of-00001.csv": 6656,
        "AgexGender_examples-00000-of-00001.csv": 2880,
        "Caste_examples-00000-of-00001.csv": 3864,
        "Disability_status_examples-00000-of-00001.csv": 5296,
        "Gender_identity_examples-00000-of-00001.csv": 6536,
        "GenderxReligion_examples-00000-of-00001.csv": 1504,
        "Nationality_examples-00000-of-00001.csv": 4128,
        "Physical_appearance_examples-00000-of-00001.csv": 6232,
        "Region_examples-00000-of-00001.csv": 3144,
        "RegionxGender_examples-00000-of-00001.csv": 1944,
        "Religion_examples-00000-of-00001.csv": 4800,
        "SES_examples-00000-of-00001.csv": 6160,
        "Sexual_orientation_examples-00000-of-00001.csv": 904,
    },
    "categories": [
        "Age", "AgexGender", "Caste", "Disability_status", "Gender_identity",
        "GenderxReligion", "Nationality", "Physical_appearance", "Region",
        "RegionxGender", "Religion", "SES", "Sexual_orientation",
    ],
    "categories_claimed_by_paper_but_absent": [
        "Linguistic Group", "Caste-Adjacent Occupation",
        "Urban-Rural Identity", "Tribal Community",
    ],
}
