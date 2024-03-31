# coding:utf-8
'''
**************************************************
@File   ：Final_project -> experiment
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2024/3/30 19:31
**************************************************
'''
import gradio as gr
import torch
from transformer_model import TransformerNano
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import pandas as pd

def generate():
    # tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    prompt = 'ceiling expanded: 15 centimeters above from intended position'
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # words = prompt.split()
    # state = torch.tensor([[word_to_ix[word] for word in words]]).to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, do_sample=True, max_length=50, temperature=1, top_p=1,
                   repetition_penalty=1)
    generate_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    start_index = generate_sentence.find('position,') + len('position,')

    # Extract the sentence after "position,"
    extracted_sentence = generate_sentence[start_index:].strip()

    # Remove excess quotation marks at the end
    cleaned_sentence = extracted_sentence.rstrip('"')

    ##########################################
    # length = len(words)
    # for i in range(19 - length):
    #     words.append(ix_to_word[outputs[:,length+i].item()])

    print(cleaned_sentence)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate()
