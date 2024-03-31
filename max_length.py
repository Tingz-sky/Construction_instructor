# coding:utf-8
'''
**************************************************
@File   ：Final_project -> max_length
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2024/3/25 11:08
**************************************************
'''
# Python script to find the maximum number of words in sentences in a file
# Assumes each line in the file represents one sentence

def max_words_in_sentence(file_path):
    max_words = 0  # Initialize max words to 0
    with open(file_path, 'r') as file:  # Open the file for reading
        for sentence in file:  # Read each line (sentence) from the file
            word_count = len(sentence.strip().split())  # Count the words in the sentence
            if word_count > max_words:  # If this sentence has the new maximum word count
                max_words = word_count  # Update max_words
    return max_words

if __name__ == "__main__":
    file_path = 'data/sentences.txt'  # Specify the path to your file
    max_words = max_words_in_sentence(file_path)  # Call the function
    print(f"The maximum number of words in the sentences in the file is: {max_words}")  # Output the result

