import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter
from synonym_expander import expand_with_synonyms

class ICDDataset(Dataset):
    def __init__(self, csv_path, max_len=50):
        self.data = pd.read_csv(csv_path)
        self.texts = self.data['Description'].astype(str)
        self.labels = self.data['ICDCode']

        self.max_len = max_len
        self.vocab = self.build_vocab(self.texts)
        self.label2idx = {l: i for i, l in enumerate(self.labels.unique())}
        self.idx2label = {i: l for l, i in self.label2idx.items()}

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        vocab = {w: i + 2 for i, (w, _) in enumerate(counter.items())}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    def encode_text(self, text):
        text = expand_with_synonyms(text.lower())
        tokens = text.split()
        encoded = [self.vocab.get(t, 1) for t in tokens][:self.max_len]
        return encoded + [0] * (self.max_len - len(encoded))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(self.encode_text(self.texts.iloc[idx]))
        y = torch.tensor(self.label2idx[self.labels.iloc[idx]])
        return x, y
