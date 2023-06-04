import numpy as np
import string

characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# def extract_character(data):
#     set_words = set([character for line in data for character in line])
#     int_to_vocab = {word_i: word for word_i, word in enumerate(list(set_words))}
#     vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

#     return int_to_vocab, vocab_to_int

def make_misspellings(sentence, threshold):
    misspellings = ''
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        if random < threshold or sentence[i] == ' ' or sentence[i] in string.punctuation:
            misspellings += sentence[i]
        else:
            new_random = np.random.uniform(0,1,1)
            # 25% chance characters will swap locations
            # Transpostion
            if new_random >= 0.75:
                if i == (len(sentence) - 1):
                    # if last character in sentence, it will be typed
                    misspellings += sentence[i]
                else:
                    # else, swap the order of the current character and the next one.
                    # if the next character is SPACE, we skip that.
                    if sentence[i+1] != ' ' and sentence[i+1] not in string.punctuation:
                        misspellings += sentence[i+1]
                        misspellings += sentence[i]
                        i += 1
                    else:
                        misspellings += sentence[i]
                    
            # 25% chance a lowercase character will replace the current character
            # Subsitution
            elif new_random >= 0.5:
                random_letter = np.random.choice(characters, 1)[0]
                misspellings += random_letter

            # 25% chance a lowercase character will be inserted to the sentence
            # Insertion
            elif new_random >= 0.25:
                random_letter = np.random.choice(characters, 1)[0]
                r = np.random.uniform(0,1,1)
                if r >= 0.5:
                    misspellings += random_letter
                    misspellings += sentence[i]
                else:
                    misspellings += sentence[i]
                    misspellings += random_letter
                
            # 25%: skip a character
            # Deletion
            else:
                pass
        i += 1

    return misspellings

def generate_input(opt):
    inputs = []
    with open(opt.data_file, "r") as f:
        lines = f.read().split("\n")
        for line in lines:
            inputs.append(line)
    f.close()

    # extract_character(inputs)
    misspelling_inputs = []
    for input in inputs:
        b = make_misspellings(input, opt.threshold)
        misspelling_inputs.append(b)

    return misspelling_inputs