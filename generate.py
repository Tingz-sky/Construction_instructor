# coding:utf-8
'''
**************************************************
@File   ：Final_project -> generate
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2024/3/24 21:59
**************************************************
'''
import gradio as gr
import torch
from transformer_model import TransformerNano
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW

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
    with open('data/vocabs.txt', 'r') as file:
        vocab = [line.strip() for line in file]
    vocab.append("<EOS>")
    vocab.append("<PAD>")
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {ix: word for word, ix in word_to_ix.items()}
    model = TransformerNano(vocab_size=len(vocab),
                                embed_dim=128,
                                num_heads=8,
                                num_layers=8,
                                seq_len=19)
    model_path = 'model.pt'
    dic = torch.load(model_path)
    model.load_state_dict(dic)
    input_sentence = 'door misaligned: 2 centimeters right from intended position'
    generated_sentence = generate_sentence(model, input_sentence, word_to_ix, ix_to_word)
    print(generated_sentence)