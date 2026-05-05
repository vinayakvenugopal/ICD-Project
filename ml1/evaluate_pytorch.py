import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import re
import os

# =========================
# MODEL DEFINITION (Matching ml1/model.py or ml_service)
# =========================
class BiLSTM_ICD(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Check if the checkpoint has a specific output_dim or if it goes straight to num_classes
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        cat_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.fc(cat_hidden)
        return output

# =========================
# CONFIGURATION
# =========================
BASE_DIR = r"c:\Users\acer\Downloads\Telegram Desktop\ICD (2)\ICD"
MODEL_PATH = os.path.join(BASE_DIR, "ml", "icd_model.pt")
DATASET_PATH = os.path.join(BASE_DIR, "ml", "ICDCodeSet.csv")
MAX_LEN = 50 # From ml1/dataset.py

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

def tokenize(text, vocab, max_len=50):
    tokens = text.split()
    encoded = [vocab.get(t, 1) for t in tokens][:max_len]
    return encoded + [0] * (max_len - len(encoded))

ICD_CHAPTERS = {
    'A': 'Infectious', 'B': 'Infectious', 
    'C': 'Neoplasms', 'D': 'Neoplasms/Blood',
    'E': 'Endocrine', 'F': 'Mental', 'G': 'Nervous',
    'H': 'Eye/Ear', 'I': 'Circulatory', 'J': 'Respiratory',
    'K': 'Digestive', 'L': 'Skin', 'M': 'Musculoskeletal',
    'N': 'Genitourinary', 'O': 'Pregnancy', 'P': 'Perinatal',
    'Q': 'Congenital', 'R': 'Syndromes', 'S': 'Injury',
    'T': 'Poisoning', 'V': 'External', 'W': 'External',
    'X': 'External', 'Y': 'External', 'Z': 'Health Services'
}

def generate_cm():
    print("Loading PyTorch model and metadata...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    VOCAB_PATH = os.path.join(BASE_DIR, "ml", "vocab.pkl")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocab loaded. Size: {len(vocab)}")

    label2idx = checkpoint['label2idx']
    idx2label = checkpoint['idx2label']
    num_classes = len(label2idx)
    
    model = BiLSTM_ICD(len(vocab), 128, 128, num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("Model loaded successfully.")

    print("Loading Dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    print("Preparing Test Split (80/20)...")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_subset = test_df.sample(n=500, random_state=42)
    
    y_true = []
    y_pred = []
    
    print(f"Running predictions on {len(test_subset)} samples...")
    with torch.no_grad():
        for _, row in test_subset.iterrows():
            text = clean_text(row['Description'])
            true_code = row['ICDCode']
            
            tokens = torch.tensor([tokenize(text, vocab, MAX_LEN)])
            logits = model(tokens)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_code = idx2label[pred_idx]
            
            # Map to Chapter Description
            t_char = str(true_code)[0].upper()
            p_char = str(pred_code)[0].upper()
            
            y_true.append(ICD_CHAPTERS.get(t_char, t_char))
            y_pred.append(ICD_CHAPTERS.get(p_char, p_char))

    labels = sorted(list(set(y_true + y_pred)))
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=labels, yticklabels=labels)
    plt.title('PyTorch ICD-10 Model: Chapter-Level Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Medical Chapter', fontsize=12)
    plt.ylabel('Actual Medical Chapter', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_dir = r"c:\Users\acer\Downloads\Telegram Desktop\ICD (2)\ICD\ml1"
    output_path = os.path.join(out_dir, "confusion_matrix_pytorch.png")
    plt.savefig(output_path)
    print(f"Done! Confusion matrix saved to {output_path}")
    
    report = classification_report(y_true, y_pred, labels=labels)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    import pickle
    generate_cm()
