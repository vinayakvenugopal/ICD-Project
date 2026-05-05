import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import pickle
from src.models.bilstm_model import BiLSTMEncoder, Vocab
from src.data_loader import DataLoader as ICDDataLoader

class ICDDataset(Dataset):
    def __init__(self, descriptions, categories, vocab, max_len=20):
        self.data = [vocab.encode(d, max_len) for d in descriptions]
        self.labels = categories
        self.descriptions = descriptions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

def contrastive_loss(out1, out2, label, margin=1.0):
    # This is a simplified contrastive loss for demo
    # We want similarity to be high for the same concept
    cos_sim = nn.functional.cosine_similarity(out1, out2)
    return 1 - cos_sim.mean() # Minimize (1 - similarity)

def train_bilstm():
    data_path = os.path.join(os.getcwd(), "data", "full_raw_codes.csv")
    loader = ICDDataLoader(data_path)
    df = loader.load_data()
    descriptions = df['description'].tolist()
    
    # Extract categories (first 3 chars of code)
    categories = [str(c)[:3] for c in df['code'].tolist()]
    unique_cats = sorted(list(set(categories)))
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    label_indices = [cat_to_idx[c] for c in categories]

    print("Building Vocabulary...")
    vocab = Vocab()
    vocab.build_vocab(descriptions)
    
    # Save metadata
    os.makedirs("src/models/weights", exist_ok=True)
    metadata = {"vocab": vocab, "cat_to_idx": cat_to_idx, "idx_to_cat": {v: k for k, v in cat_to_idx.items()}}
    with open("src/models/weights/model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    dataset = ICDDataset(descriptions, label_indices, vocab)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    print(f"Initializing BiLSTM (Vocab: {len(vocab)}, Classes: {len(unique_cats)})...")
    # Add hidden layer for classification training
    model = BiLSTMEncoder(vocab_size=len(vocab), output_dim=128)
    classifier = nn.Linear(128, len(unique_cats))
    
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print("Starting Supervised Training (3 Epochs)...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        for i, (batch, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            features = model(batch)
            logits = classifier(features)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 200 == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
    # Save model weights
    torch.save(model.state_dict(), "src/models/weights/bilstm_icd10.pth")
    print("Training complete. BiLSTM weights saved.")

if __name__ == "__main__":
    train_bilstm()
