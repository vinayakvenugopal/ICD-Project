import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_ICD(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.bilstm(emb)

        h_cat = torch.cat((h[0], h[1]), dim=1)
        logits = self.fc(h_cat)

        return logits, h_cat


def contrastive_loss(emb1, emb2, label, margin=1.0):
    dist = F.pairwise_distance(emb1, emb2)
    loss = (1 - label) * dist.pow(2) + label * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()
