# import pickle
# import torch
# import torch.nn as nn
# import faiss
# import numpy as np

# EMBED_DIM = 128
# HIDDEN_DIM = 128
# MAX_LEN = 50
# K = 5
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Load files
# with open("vocab.pkl", "rb") as f:
#     word2idx = pickle.load(f)

# with open("faiss_meta.pkl", "rb") as f:
#     meta = pickle.load(f)

# codes = meta["codes"]
# index = faiss.read_index("icd_faiss.index")

# class BiLSTM(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

#     def forward(self, x):
#         x = self.embedding(x)
#         out, _ = self.lstm(x)
#         return out.mean(dim=1)

# model = BiLSTM(len(word2idx), EMBED_DIM, HIDDEN_DIM).to(DEVICE)
# model.load_state_dict(torch.load("icd_model.pt", map_location=DEVICE)["model_state"], strict=False)
# model.eval()

# def text_to_seq(text):
#     ids = [word2idx.get(w, word2idx["<unk>"]) for w in text.lower().split()]
#     ids = ids[:MAX_LEN] + [0] * (MAX_LEN - len(ids))
#     return torch.tensor([ids]).to(DEVICE)

# def predict(text):
#     with torch.no_grad():
#         v = model(text_to_seq(text)).cpu().numpy().astype("float32")
#     D, I = index.search(v, K)
#     return [(codes[i], float(D[0][idx])) for idx, i in enumerate(I[0])]


# while True:
#     q = input("\nEnter clinical description: ")
#     if q.lower() == "exit":
#         break
#     results = predict(q)
#     print("\nTop ICD predictions:")
#     for code, score in results:
#         print(f"{code}")




















import pickle
import torch
import torch.nn as nn
import faiss
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBED_DIM = 128
HIDDEN_DIM = 128
MAX_LEN = 50
K = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load vocab
with open(os.path.join(BASE_DIR, "vocab.pkl"), "rb") as f:
    word2idx = pickle.load(f)

# Load meta
with open(os.path.join(BASE_DIR, "faiss_meta.pkl"), "rb") as f:
    meta = pickle.load(f)

codes = meta["codes"]

# Load FAISS index
index = faiss.read_index(os.path.join(BASE_DIR, "icd_faiss.index"))

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return out.mean(dim=1)

# Load model ONCE
model = BiLSTM(len(word2idx), EMBED_DIM, HIDDEN_DIM).to(DEVICE)
state = torch.load(os.path.join(BASE_DIR, "icd_model.pt"), map_location=DEVICE)
model.load_state_dict(state["model_state"], strict=False)
model.eval()

def text_to_seq(text):
    ids = [word2idx.get(w, word2idx["<unk>"]) for w in text.lower().split()]
    ids = ids[:MAX_LEN] + [0] * (MAX_LEN - len(ids))
    return torch.tensor([ids]).to(DEVICE)

def predict_icd(text):
    with torch.no_grad():
        v = model(text_to_seq(text)).cpu().numpy().astype("float32")
    D, I = index.search(v, K)

    return [
        {
            "code": codes[i],
            "score": float(D[0][idx])
        }
        for idx, i in enumerate(I[0])
    ]
