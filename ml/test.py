# =========================
# IMPORTS
# =========================
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "bilstm_icd_model (1).h5"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "icd_label_encoder.pkl"

MAX_LEN = 150   # MUST be same as training

# =========================
# LOAD MODEL & OBJECTS
# =========================
model = load_model(MODEL_PATH)
tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))

print("✅ ICD Prediction Model Loaded Successfully")

# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# ICD PREDICTION FUNCTION
# =========================
def predict_icd(description):
    description = clean_text(description)

    sequence = tokenizer.texts_to_sequences([description])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)

    icd_code = label_encoder.inverse_transform(predicted_class)[0]
    confidence = np.max(prediction)

    return icd_code, confidence

# =========================
# USER INPUT (TEST)
# =========================
if __name__ == "__main__":
    user_input = "The patient has been diagnosed with early-onset Alzheimer’s disease, characterized by cognitive decline occurring at a younger age."
    
    icd, conf = predict_icd(user_input)
    
    print("\n🩺 Clinical Description:")
    print(user_input)
    
    print("\n📌 Predicted ICD Code:")
    print(icd)
    
    print("\n📊 Confidence Score:")
    print(round(conf * 100, 2), "%")
