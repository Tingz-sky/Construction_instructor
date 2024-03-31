# coding:utf-8
'''
**************************************************
@File   ：Final_project -> train_gpt2
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2024/3/25 12:59
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
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm


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

class ConstructionDS(Dataset):
    """
    Define a custom dataset class
    :param sentences: a list a strings that are full sentences for training
    :param tokenizer: a tokenizer to tokenize the sentences
    :param max_length: the max length of each tokenized sentence
    """

    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=self.max_length, truncation=True)
        return inputs

def lm_collate_fn(batch, device):
    x = [item.data['input_ids'] for item in batch]  # List (len B) of varying lengths
    y = [item.data['attention_mask'] for item in batch]  # List (len B) of the same lengths as x
    # maxlen = max([len(s) for s in x])
    maxlen = max([s.shape[1] for s in x])

    padded_x, padded_y = [], []
    for sx, sy in zip(x, y):
        padded_x.append(torch.cat([sx.squeeze(), torch.ones(maxlen - sx.shape[1])]))
        padded_y.append(torch.cat([sy.squeeze(), torch.ones(maxlen - sy.shape[1])]))
    for i in range(len(batch)):
        batch[i].data['input_ids'] = padded_x[i].reshape(1, -1)
        batch[i].data['attention_mask'] = padded_y[i].reshape(1, -1)
    return torch.stack(padded_x).long().to(device), torch.stack(padded_y).long().to(device)

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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
    # dataset = NanoLanguageDataset(sentences, vocab, word_to_ix)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    epochs = 10

    ###########
    sentences_train, sentences_test = train_test_split(sentences, test_size=0.1, random_state=42)
    # using a small dataset for development purposes
    # sentences_train, sentences_test = train_test_split(sentences_test, test_size=0.1, random_state=42)

    # Create a custom dataset
    dataset_train = ConstructionDS(sentences_train, tokenizer)
    dataset_test = ConstructionDS(sentences_test, tokenizer)

    # Set up DataLoader
    # dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    # dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True,
                                  collate_fn=lambda batch: lm_collate_fn(batch, device))
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True,
                                 collate_fn=lambda batch: lm_collate_fn(batch, device))
    dataloader_train_len = len(dataloader_train)
    dataloader_test_len = len(dataloader_test)


    # --- Step 2: Model Definition ---

    # Model instantiation
    # model = TransformerNano(vocab_size=len(vocab),
    #                         embed_dim=128,
    #                         num_heads=8,
    #                         num_layers=8,
    #                         seq_len=19).to(device)
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)


    # --- Training Process ---
    # model.train()
    # criterion = nn.CrossEntropyLoss()
    # # optimizer = Adam(model.parameters(), lr=0.00001)
    #
    # for epoch in range(epochs):  # Small number of epochs for demonstration
    #     for inputs, targets in dataloader:
    #         optimizer.zero_grad()
    #         outputs = model(input_ids=inputs, labels=targets)
    #         loss = outputs.loss
    #         # outputs = outputs.transpose(1, 2)
    #         # loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    # model_dir = 'model'
    # # torch.save(model.state_dict(), model_dir)
    # model.save_pretrained(model_dir)
    # # tokenizer.save_pretrained(model_dir)

    # Fine-tune loop
    model.train()
    for epoch in range(epochs):
        total_loss_train = 0
        total_loss_test = 0

        progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # Move batch to device
            batch = {"input_ids": batch[0], "attention_mask": batch[1]}
            # batch = {key: value[0].to(device) for key, value in batch.items()}

            # Forward pass
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            # Update progress bar
            progress_bar.set_postfix({"Loss": total_loss_train / dataloader_train_len})

        for batch_idx, batch in enumerate(dataloader_test, 1):
            # Move batch to device
            # batch = {key: value[0].to(device) for key, value in batch.items()}
            batch = {"input_ids": batch[0], "attention_mask": batch[1]}

            # Forward pass
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            total_loss_test += loss.item()
        total_loss_test /= dataloader_test_len
        opt_lr = optimizer.param_groups[0]['lr']

        model.save_pretrained('model_gpt2')
        tokenizer.save_pretrained('model_gpt2')