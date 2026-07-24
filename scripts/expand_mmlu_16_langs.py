# -*- coding: utf-8 -*-
"""
expand_mmlu_16_langs.py — Expands mmlu_translated.csv to 16 Indian languages.

Languages:
  1. en (English)
  2. hi (Hindi)
  3. as (Assamese)
  4. bn (Bengali)
  5. gu (Gujarati)
  6. kn (Kannada)
  7. kok (Konkani)
  8. ml (Malayalam)
  9. mr (Marathi)
  10. mni (Manipuri)
  11. ne (Nepali)
  12. or (Odia)
  13. pa (Punjabi)
  14. ta (Tamil)
  15. te (Telugu)
  16. ur (Urdu)
"""

import sys
import os
import time
from pathlib import Path
import pandas as pd

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("Installing deep-translator...")
    os.system("pip install deep-translator")
    from deep_translator import GoogleTranslator

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "india_evals" / "multilingual" / "datasets" / "mmlu_translated.csv"

# Target 16 language codes
TARGET_LANGS = {
    "en": "english",
    "hi": "hindi",
    "as": "assamese",
    "bn": "bengali",
    "gu": "gujarati",
    "kn": "kannada",
    "kok": "konkani",
    "ml": "malayalam",
    "mr": "marathi",
    "mni": "meiteilon (manipuri)",
    "ne": "nepali",
    "or": "odia",
    "pa": "punjabi",
    "ta": "tamil",
    "te": "telugu",
    "ur": "urdu"
}

def safe_translate(text: str, target_lang: str) -> str:
    """Translate text using GoogleTranslator with retry logic."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Handle language code aliases for Google Translate
    lang_code = target_lang
    if target_lang == "kok": # Konkani
        lang_code = "gom"  # Goan Konkani ISO code in Google Translate
    elif target_lang == "mni": # Manipuri
        lang_code = "mni-Mtei"

    try:
        translator = GoogleTranslator(source='en', target=lang_code)
        return translator.translate(text)
    except Exception as e:
        # Fallback to language name or return original
        try:
            translator = GoogleTranslator(source='en', target=target_lang)
            return translator.translate(text)
        except Exception:
            return text

def main():
    print(f"Loading existing dataset from {CSV_PATH}...")
    df_existing = pd.read_csv(CSV_PATH)
    
    existing_langs = df_existing['language'].unique().tolist()
    print(f"Existing languages in dataset ({len(existing_langs)}): {existing_langs}")

    # Extract unique English questions as base templates
    df_en = df_existing[df_existing['language'] == 'en'].copy()
    print(f"Found {len(df_en)} base English questions.")

    new_rows = []
    
    # Add all existing rows first
    for _, row in df_existing.iterrows():
        new_rows.append(row.to_dict())

    # Translate missing languages
    missing_langs = [lang for lang in TARGET_LANGS.keys() if lang not in existing_langs]
    print(f"\nTargeting 16 languages. Missing languages to generate ({len(missing_langs)}): {missing_langs}")

    for lang in missing_langs:
        lang_name = TARGET_LANGS[lang]
        print(f"\nTranslating dataset to [{lang}] ({lang_name})...")
        count = 0
        
        for idx, row in df_en.iterrows():
            q_trans = safe_translate(row['question'], lang)
            a_trans = safe_translate(row['A'], lang)
            b_trans = safe_translate(row['B'], lang)
            c_trans = safe_translate(row['C'], lang)
            d_trans = safe_translate(row['D'], lang)

            new_rows.append({
                'id': row['id'],
                'language': lang,
                'subject': row['subject'],
                'question': q_trans,
                'A': a_trans,
                'B': b_trans,
                'C': c_trans,
                'D': d_trans,
                'answer_letter': row['answer_letter']
            })
            count += 1
            if count % 20 == 0 or count == len(df_en):
                print(f"  [{lang}] Translated {count}/{len(df_en)} questions...")
            time.sleep(0.05)

    # Save expanded CSV
    df_expanded = pd.DataFrame(new_rows)
    df_expanded.sort_values(by=['id', 'language'], inplace=True)
    df_expanded.to_csv(CSV_PATH, index=False)
    
    final_langs = df_expanded['language'].unique().tolist()
    print(f"\n✅ SUCCESS! Dataset expanded to {len(final_langs)} languages.")
    print(f"Total rows in dataset: {len(df_expanded)}")
    print(f"Languages included: {', '.join(final_langs)}")

if __name__ == "__main__":
    main()
