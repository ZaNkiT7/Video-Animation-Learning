import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer


# SIMPLE TEXT TO TRAIN (NO FILE NEEDED)
training_text = """
hello world this is a small test model that predicts the next word
hello world this is another test example for next word prediction
machine learning is fun and python is powerful
learning python every day makes coding easier
"""


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length=5):
        self.tokenizer = tokenizer
        self.tokens = tokenizer(text)
        self.seq_length = seq_length

        self.data = []
        for i in range(len(self.tokens) - seq_length):
            x = self.tokens[i:i+seq_length]
            y = self.tokens[i+seq_length]
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


tokenizer = get_tokenizer("basic_english")
dataset = TextDataset(training_text, tokenizer)

vocab = list(set(dataset.tokens))
vocab_size = len(vocab)

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}


def tokens_to_indices(tokens):
    return [word_to_idx[t] for t in tokens]


class IndexedDataset(TextDataset):
    def __getitem__(self, idx):
        X, Y = super().__getitem__(idx)
        X_ids = torch.tensor(tokens_to_indices(X))
        Y_id = torch.tensor(word_to_idx[Y])
        return X_ids, Y_id


indexed_dataset = IndexedDataset(training_text, tokenizer)
data_loader = DataLoader(indexed_dataset, batch_size=4, shuffle=True)


class NeuralNetModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        last = out[:, -1]
        return self.fc(last)


model = NeuralNetModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for X, Y in data_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(data_loader):.4f}")


def predict_next_word(sequence):
    seq_tokens = tokenizer(sequence)
    seq_idx = torch.tensor(tokens_to_indices(seq_tokens)).unsqueeze(0)

    with torch.no_grad():
        output = model(seq_idx)
        predicted_idx = torch.argmax(output, dim=1).item()

    return idx_to_word[predicted_idx]


print("\nModel Ready.\n")
user_inp = input("Enter sequence of 5 words: ")
print("Predicted next word:", predict_next_word(user_inp))
