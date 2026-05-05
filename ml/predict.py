import os
import re
import sys

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_SERVICE_DIR = os.path.join(BASE_DIR, "ml_service")
DATA_PATH = os.path.join(ML_SERVICE_DIR, "data", "full_raw_codes.csv")

if ML_SERVICE_DIR not in sys.path:
    sys.path.insert(0, ML_SERVICE_DIR)

_detector = None
_detector_error = None
_fallback_df = None


def _format_code(raw_code):
    raw_code = str(raw_code)
    return raw_code[:3] + "." + raw_code[3:] if len(raw_code) > 3 else raw_code


def _load_detector():
    global _detector, _detector_error
    if os.environ.get("USE_NEURAL_ICD", "True").lower() != "true":
        return None

    if _detector is not None or _detector_error is not None:
        return _detector

    try:
        from src.detector import ICD10Detector

        _detector = ICD10Detector(DATA_PATH)
    except Exception as exc:
        _detector_error = exc
        print(f"ICD neural detector unavailable, using fallback search: {exc}")

    return _detector


def _load_fallback_df():
    global _fallback_df
    if _fallback_df is None:
        raw_df = pd.read_csv(DATA_PATH, header=None, quoting=1)
        _fallback_df = pd.DataFrame({
            "code": raw_df[2].apply(_format_code),
            "description": raw_df[4].astype(str),
        })
    return _fallback_df


def _fallback_predict(description, top_k=3):
    df = _load_fallback_df()
    terms = [
        term
        for term in re.findall(r"\w+", description.lower())
        if len(term) > 3 and term not in {"patient", "with", "presents", "symptoms", "acute"}
    ]

    if not terms:
        return []

    scored = []
    for row in df.itertuples(index=False):
        text = row.description.lower()
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, row.code, row.description))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, code, matched_description in scored[:top_k]:
        confidence = min(0.95, 0.35 + (score / max(len(terms), 1)) * 0.6)
        results.append({
            "icd_code": code,
            "description": matched_description,
            "confidence": confidence,
            "match_type": "keyword_fallback",
            "reasoning": f"Matched clinical terms against ICD description '{matched_description}'.",
        })
    return results


def predict_icd(description):
    """
    Main entry point for Django views.
    Uses the BiLSTM detector when available, with a keyword fallback for deployment demos.
    """
    detector = _load_detector()
    if detector is not None:
        try:
            raw_results = detector.detect(description, top_k=3)
            return [
                {
                    "icd_code": res["code"],
                    "description": res["description"],
                    "confidence": res["score"],
                    "match_type": "hybrid_bilstm",
                    "reasoning": res["reasoning"],
                }
                for res in raw_results
            ]
        except Exception as exc:
            print(f"Error in hybrid prediction, using fallback search: {exc}")

    return _fallback_predict(description)


if __name__ == "__main__":
    test_desc = "The patient presents with gastrointestinal illness caused by foodborne Clostridium perfringens intoxication."
    print(predict_icd(test_desc))
