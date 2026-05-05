import torch
import torch.nn as nn
import collections
import re

class Vocab:
    def __init__(self, min_freq=1):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.min_freq = min_freq
        self.counts = collections.Counter()

    def build_vocab(self, sentences):
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            self.counts.update(tokens)
        
        for word, freq in self.counts.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def encode(self, text, max_len=20):
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
        padding = [0] * (max_len - len(indices))
        return indices + padding

    def __len__(self):
        return len(self.word2idx)

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, output_dim=128):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        # 2 * hidden_dim because it's bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x) # (batch, seq_len, emb_dim)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final forward and backward hidden states
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        cat_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        output = self.fc(cat_hidden)
        # Normalize to unit vector for contrastive similarity (cosine similarity)
        output = nn.functional.normalize(output, p=2, dim=1)
        return output
