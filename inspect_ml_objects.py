import pickle
import os

BASE_DIR = r"c:\Users\acer\Downloads\Telegram Desktop\ICD (2)\ICD"
ML_DIR = os.path.join(BASE_DIR, "ml")
VOCAB_PATH = os.path.join(ML_DIR, "vocab.pkl")
TOKENIZER_PATH = os.path.join(ML_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(ML_DIR, "icd_label_encoder.pkl")

print("Checking vocab.pkl...")
try:
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"vocab.pkl type: {type(vocab)}")
    if isinstance(vocab, dict):
        print(f"Vocab size: {len(vocab)}")
        print(f"Sample items: {list(vocab.items())[:5]}")
except Exception as e:
    print(f"Error loading vocab.pkl: {e}")

print("\nChecking label_encoder.pkl...")
try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    print(f"label_encoder type: {type(le)}")
except Exception as e:
    print(f"Error loading label_encoder.pkl: {e}")

print("\nChecking if we can import Keras Tokenizer...")
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    print("Successfully imported Keras Tokenizer")
except Exception as e:
    print(f"Error importing Tokenizer: {e}")
