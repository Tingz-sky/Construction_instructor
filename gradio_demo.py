# coding:utf-8
'''
**************************************************
@File   ：Final_project -> gradio_demo
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2024/3/26 20:48
**************************************************
'''
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def instruction_helper(error_type, error_num, component, direction, unit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model_gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
    input_senetence = f"{component} {error_type}: {error_num} {unit} {direction} from intended position"
    length = len(input_senetence)
    input_ids = tokenizer(input_senetence, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, do_sample=True, max_length=50, temperature=1, top_p=1,
                   repetition_penalty=1)
    generate_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    start_index = generate_sentence.find('position,') + len('position,')

    # Extract the sentence after "position,"
    extracted_sentence = generate_sentence[start_index:].strip()

    # Remove excess quotation marks at the end
    cleaned_sentence = extracted_sentence.rstrip('"')
    return cleaned_sentence

if __name__ == '__main__':
    error_types = ['misaligned', 'incorrect size', 'too high', 'too low', 'shifted', 'expanded', 'contracted', 'cracked',
               'leaking', 'obstructed']
    component_types = ['door', 'window', 'beam', 'wall', 'floor', 'ceiling', 'pipe', 'foundation', 'roof', 'stair']
    direction_types = ['left', 'right', 'above', 'below', 'north', 'south', 'east', 'west']
    measurement_units = ['meters', 'centimeters', 'feet']
    error_type = gr.Dropdown(choices=error_types, label="Select an error type", info="Select an error type")
    error_num = gr.Number(label="Error Number Value", info="Please enter the value here", minimum=0, maximum=50)
    component = gr.Dropdown(choices=component_types, label="Select a component", info="Select a component")
    direction = gr.Dropdown(choices=direction_types, label="Select a direction", info="Select a direction")
    unit = gr.Dropdown(choices=measurement_units, label="Select a measurement unit", info="Select a measurement unit")
    output = gr.Text(label="Construction Instruction")
    demo = gr.Interface(fn=instruction_helper,
                        inputs=[error_type, error_num, component, direction, unit],
                        outputs=output)
    demo.launch(inbrowser=True)