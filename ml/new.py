import pandas as pd
import re

# =========================
# TEXT NORMALIZATION
# =========================
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# LOAD ICD DATASET
# =========================
def load_icd_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    df["norm_desc"] = df["Description"].apply(normalize)

    return df[["ICDCode", "Description", "norm_desc"]]

# =========================
# ICD PREDICTION (RULE BASED)
# =========================
def predict_icd_from_dataset(text: str, df):
    t = normalize(text)

    # 1️⃣ EXACT MATCH
    exact = df[df["norm_desc"] == t]
    if not exact.empty:
        row = exact.iloc[0]
        return {
            "icd_code": row["ICDCode"],
            "description": row["Description"],
            "confidence": 1.0,
            "match_type": "exact"
        }

    # 2️⃣ LONGEST PARTIAL MATCH
    best_len = 0
    best_row = None

    for _, row in df.iterrows():
        if row["norm_desc"] in t:
            if len(row["norm_desc"]) > best_len:
                best_len = len(row["norm_desc"])
                best_row = row

    if best_row is not None:
        return {
            "icd_code": best_row["ICDCode"],
            "description": best_row["Description"],
            "confidence": 0.95,
            "match_type": "partial"
        }

    # 3️⃣ KEYWORD FALLBACK (CHOLERA, TYPHOID, ETC.)
    for _, row in df.iterrows():
        keywords = row["norm_desc"].split()
        overlap = len(set(keywords) & set(t.split()))
        if overlap >= 3:
            return {
                "icd_code": row["ICDCode"],
                "description": row["Description"],
                "confidence": 0.85,
                "match_type": "keyword"
            }

    return {
        "icd_code": None,
        "description": None,
        "confidence": 0.0,
        "match_type": "no_match"
    }
