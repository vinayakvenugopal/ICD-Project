import torch
import torch.nn.functional as F
from dataset import ICDDataset
from model import BiLSTM_ICD

dataset = ICDDataset("data/ICDCodeSet.csv")

model = BiLSTM_ICD(
    vocab_size=len(dataset.vocab),
    embed_dim=128,
    hidden_dim=128,
    num_classes=len(dataset.label2idx)
)

model.load_state_dict(torch.load("icd_bilstm.pth"))
model.eval()

def predict_icd(text):
    encoded = torch.tensor([dataset.encode_text(text)])
    logits, emb = model(encoded)

    probs = F.softmax(logits, dim=1)
    idx = torch.argmax(probs, dim=1).item()

    return dataset.idx2label[idx], probs[0][idx].item()

# Example
code, confidence = predict_icd("severe chest pain and sweating")
print("Predicted ICD:", code)
print("Confidence:", confidence)
