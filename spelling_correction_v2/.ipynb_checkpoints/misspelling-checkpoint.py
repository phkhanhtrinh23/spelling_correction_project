import numpy as np
import string

characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n',
              'o','p','q','r','s','t','u','v','w','x','y','z']

def make_misspelling(sentence, threshold):
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