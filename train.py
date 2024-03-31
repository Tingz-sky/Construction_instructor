# coding:utf-8
'''
**************************************************
@File   ：1724 -> test
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2024/2/6 19:50
**************************************************
'''
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from collections import Counter
import random
from transformer_model import TransformerNano, TransformerNano_base


# --- Step 1: Dataset Creation ---
class NanoLanguageDataset(Dataset):
    def __init__(self, sentences, vocab, word_to_ix, seq_len=19):
        self.sentences = sentences
        self.vocab = vocab
        self.word_to_ix = word_to_ix
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = [self.word_to_ix[word] for word in sentence.split()]
        # Pad sequences to fixed length
        tokens.extend([self.word_to_ix['<PAD>']] * (self.seq_len - len(tokens)))
        return torch.tensor(tokens[:-1]).to(device), torch.tensor(tokens[1:]).to(device)

def generate_sentence(model, start_words, word_to_ix, ix_to_word, max_len=19):
    model.eval()
    words = start_words.split()
    state = torch.tensor([[word_to_ix[word] for word in words]])
    for _ in range(max_len - len(words)):
        with torch.no_grad():
            output = model(state)
            last_word_logits = output[0, -1, :]
            predicted_word_ix = torch.argmax(last_word_logits).item()
            state = torch.cat([state, torch.tensor([[predicted_word_ix]])], dim=1)
            if ix_to_word[predicted_word_ix] == "<EOS>":
                break
            words.append(ix_to_word[predicted_word_ix])
    return ' '.join(words)

if __name__ == '__main__':
    # Prepare vocabulary and sentences
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read sentences into a Python list
    with open('data/sentences.txt', 'r') as file:
        sentences = [line.strip() for line in file]
    with open('data/vocabs.txt', 'r') as file:
        vocab = [line.strip() for line in file]
    vocab.append("<EOS>")
    vocab.append("<PAD>")
    # Establish a dictionary to store the vocab index
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    # vocab = ["<PAD>", "<EOS>", "hello", "world", "good", "morning", "the", "is", "bright", "and", "beautiful"]
    # sentences = ["hello world", "good morning", "the world is bright and beautiful"]

    # Create dataset
    dataset = NanoLanguageDataset(sentences, vocab, word_to_ix)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    epochs = 100


    # --- Step 2: Model Definition ---

    # Model instantiation
    model = TransformerNano(vocab_size=len(vocab),
                            embed_dim=128,
                            num_heads=8,
                            num_layers=8,
                            seq_len=19).to(device)

    # Hand-write model
    # model = TransformerNano_base(vocab_size=len(vocab),
    #                         embed_size=32,
    #                         num_layers=2,
    #                         heads=2,
    #                         device="cpu",
    #                         ff_dim=128,
    #                         dropout=0.1,
    #                         max_length=10)


    # --- Training Process ---
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.000001)

    for epoch in range(epochs):  # Small number of epochs for demonstration
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.transpose(1, 2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    model_dir = 'model.pt'
    torch.save(model.state_dict(), model_dir)
