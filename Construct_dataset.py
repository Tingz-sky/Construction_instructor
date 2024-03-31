# coding:utf-8
'''
**************************************************
@File   ：Final_project -> Construct_dataset
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2024/3/14 16:01
**************************************************
'''
import itertools
import pandas as pd

components = ['door', 'window', 'beam', 'wall', 'floor', 'ceiling', 'pipe', 'foundation', 'roof', 'stair']
directions = ['left', 'right', 'above', 'below', 'north', 'south', 'east', 'west']
numbers = [1, 2, 3, 5, 10, 15, 50, 1, 2, 3]  # Note: 1, 2, 3 repeated for feet
measurement_units = ['meters', 'centimeters', 'feet']
error_types = ['misaligned', 'incorrect size', 'too high', 'too low', 'shifted', 'expanded', 'contracted', 'cracked',
               'leaking', 'obstructed']


def generate_corrective_action(error_type, direction, number, unit):
    corrective_measure = f"{number} {unit}"
    if error_type in ['misaligned', 'shifted']:
        if direction in ['left', 'right']:
            return f"shift it {corrective_measure} to the {'right' if direction == 'left' else 'left'}"
        elif direction in ['above', 'below']:
            return f"move it {corrective_measure} {'down' if direction == 'above' else 'up'}"
    elif error_type in ['incorrect size', 'expanded', 'contracted']:
        return f"adjust the size by making it {corrective_measure} {'narrower' if direction in ['left', 'right'] else 'shorter'}"
    elif error_type in ['too high', 'too low']:
        return f"adjust the height by moving it {corrective_measure} {'down' if error_type == 'too high' else 'up'}"
    elif error_type in ['cracked', 'leaking', 'obstructed']:
        return "inspect and repair the affected area"
    return "perform a detailed inspection to determine the correct action"


dataset = []

for component, error_type, direction, number, unit in itertools.product(components, error_types, directions, numbers,
                                                                        measurement_units):
    error_description = f"{component} {error_type}: {number} {unit} {direction} from intended position , "
    corrective_action = generate_corrective_action(error_type, direction, number, unit)
    # corrective_instruction = f"Corrective Action: {corrective_action}."
    corrective_instruction = corrective_action
    dataset.append(error_description + corrective_instruction)

# Write sentences to a text file
with open('data/sentences.txt', 'w') as file:
    for sentence in dataset:
        file.write(sentence + '\n')

# Generate vocabulary list
vocab_set = set()
for sentence in dataset:
    words = sentence[:].lower().split()  # Remove the period and convert to lowercase
    vocab_set.update(words)

vocab_list = sorted(list(vocab_set))
with open('data/vocabs.txt', 'w') as file:
    for vocab in vocab_list:
        file.write(vocab + '\n')


# df = pd.DataFrame(dataset)
# csv_path = 'data/construction_errors_dataset.csv'
# df.to_csv(csv_path, index=False)
# print(f"Dataset saved to {csv_path}")

# df.to_pickle('data/dataset.pkl')


