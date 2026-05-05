import sys
import types
import numpy as np
import pickle
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# =========================
# CUSTOM UNPICKLER
# =========================
class KerasCompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map numpy._core to numpy.core
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        
        # Map keras.src to tensorflow.keras
        if module.startswith('keras.src'):
            module = module.replace('keras.src', 'tensorflow.keras')
            module = module.replace('.legacy', '')
            # Some versions might just use 'keras' instead of 'tensorflow.keras'
            try:
                return super().find_class(module, name)
            except ImportError:
                module = module.replace('tensorflow.keras', 'keras')
        
        return super().find_class(module, name)

def load_pkl(path):
    with open(path, "rb") as f:
        return KerasCompatibleUnpickler(f).load()

# =========================
# CONFIGURATION
# =========================
BASE_DIR = r"c:\Users\acer\Downloads\Telegram Desktop\ICD (2)\ICD"
ML_DIR = os.path.join(BASE_DIR, "ml")
MODEL_PATH = os.path.join(ML_DIR, "bilstm_icd_model (1).h5")
TOKENIZER_PATH = os.path.join(ML_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(ML_DIR, "icd_label_encoder.pkl")
DATASET_PATH = os.path.join(ML_DIR, "ICDCodeSet.csv")

MAX_LEN = 150

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

def generate_cm():
    print("Loading pre-trained objects with compatibility layer...")
    try:
        le = load_pkl(LABEL_ENCODER_PATH)
        print(f"Success: LabelEncoder loaded.")
    except Exception as e:
        print(f"Failed to load LabelEncoder: {e}")
        return

    try:
        tokenizer = load_pkl(TOKENIZER_PATH)
        print(f"Success: Tokenizer loaded.")
    except Exception as e:
        print(f"Failed to load Tokenizer: {e}")
        return

    print("Loading Dataset...")
    df = pd.read_csv(DATASET_PATH)
    df['CleanDescription'] = df['Description'].apply(clean_text)
    
    num_classes = len(le.classes_)
    MAX_WORDS = 30000 
    
    # Rebuild Sequential Model
    print(f"Building Model Architecture...")
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    print("Loading Model Weights...")
    try:
        # We try load_model with compile=False first because it might handle names better
        # then we transfer weights.
        temp_model = load_model(MODEL_PATH, compile=False)
        model.set_weights(temp_model.get_weights())
        print("Success: Weights loaded via temp_model.")
    except Exception as e:
        print(f"load_model failed: {e}. Trying weights only...")
        try:
            model.load_weights(MODEL_PATH)
            print("Success: Weights loaded via load_weights.")
        except Exception as e2:
            print(f"Critical Failure: {e2}")
            return

    print("Preparing Test Split...")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_subset = test_df.sample(n=500, random_state=42)
    
    print(f"Running inference on {len(test_subset)} samples...")
    sequences = tokenizer.texts_to_sequences(test_subset['CleanDescription'])
    X_test = pad_sequences(sequences, maxlen=MAX_LEN)
    
    predictions = model.predict(X_test, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)
    
    y_pred_codes = le.inverse_transform(predicted_indices)
    y_true_codes = test_subset['ICDCode'].values
    
    y_true_chapters = [str(c)[0].upper() for c in y_true_codes]
    y_pred_chapters = [str(c)[0].upper() for c in y_pred_codes]
    
    labels = sorted(list(set(y_true_chapters + y_pred_chapters)))
    
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true_chapters, y_pred_chapters, labels=labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Corrected Keras Model: Chapter-Level Confusion Matrix')
    plt.xlabel('Predicted Chapter')
    plt.ylabel('True Chapter')
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(out_dir, "confusion_matrix_final.png")
    plt.savefig(output_path)
    print(f"Done! Confusion matrix saved to {output_path}")
    
    report = classification_report(y_true_chapters, y_pred_chapters, labels=labels)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    generate_cm()
