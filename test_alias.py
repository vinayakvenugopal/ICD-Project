import sys
import numpy as np
import pickle
import os

# Trick to load pickles from newer numpy versions
sys.modules['numpy._core'] = np

BASE_DIR = r"c:\Users\acer\Downloads\Telegram Desktop\ICD (2)\ICD"
ML_DIR = os.path.join(BASE_DIR, "ml")
LABEL_ENCODER_PATH = os.path.join(ML_DIR, "icd_label_encoder.pkl")
TOKENIZER_PATH = os.path.join(ML_DIR, "tokenizer.pkl")

print("Attempting to load label_encoder with alias...")
try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    print(f"Success! LabelEncoder loaded. Classes: {len(le.classes_)}")
except Exception as e:
    print(f"Failed label_encoder: {e}")

print("\nAttempting to load tokenizer with alias...")
try:
    # Tokenizer might also have issues if it involves numpy
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Success! Tokenizer loaded.")
except Exception as e:
    print(f"Failed tokenizer: {e}")
