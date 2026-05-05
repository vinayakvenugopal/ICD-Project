import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ICDDataset
from model import BiLSTM_ICD, contrastive_loss
from tqdm import tqdm

dataset = ICDDataset(r"C:\Users\acer\Downloads\Telegram Desktop\ICD (2)\ICD\ml\ICDCodeSet.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = BiLSTM_ICD(
    vocab_size=len(dataset.vocab),
    embed_dim=128,
    hidden_dim=128,
    num_classes=len(dataset.label2idx)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ce_loss = nn.CrossEntropyLoss()

epochs = 50

for epoch in range(epochs):
    total_loss = 0
    model.train()

    for x, y in tqdm(loader):
        optimizer.zero_grad()

        logits, embeddings = model(x)
        loss_cls = ce_loss(logits, y)

        # Contrastive pairs (simple batch-wise)
        emb1, emb2 = embeddings[:-1], embeddings[1:]
        labels = (y[:-1] != y[1:]).float()
        loss_con = contrastive_loss(emb1, emb2, labels)

        loss = loss_cls + 0.3 * loss_con
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "icd_bilstm.pth")
