import argparse
import torch
from transformer import get_model
from preprocess import *
from beam_search import beam_search
from torch.autograd import Variable

def process_sentence(sentence, model, opt, SRC, TRG):
    model.eval()
    indexed = []
    sentence = preprocess(sentence)
    for tok in sentence:
        indexed.append(SRC.vocab.stoi[tok])
    sentence = Variable(torch.LongTensor([indexed]))
    sentence = beam_search(sentence, model, SRC, TRG, opt)
    return sentence

def process(opt, model, SRC, TRG):
    sentences = opt.text
    correct_sentences = []
    for sentence in sentences:
        correct_sentences.append(process_sentence(sentence, model, opt, SRC, TRG).capitalize()) 
        # .capitalize()): convert the first letter to uppercase letter
    return correct_sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=bool, default=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-src_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-trg_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-cuda_device', type=str, default="cpu")
    parser.add_argument('-max_strlen', type=int, default=300)

    opt = parser.parse_args()

    SRC, TRG = create_files(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    
    while True:
        opt.text = input("Enter a filename to process (type \"quit\" to escape):\n")
        if opt.text == "quit":
            break
        
        sentences = []
        with open(opt.text, "r", encoding='utf-8') as f:
            for text in f.readlines():
                sentences.append(text[:-1]) # skip endline character
        opt.text = sentences
        print("Processing...")
        correct_sentences = process(opt, model, SRC, TRG)
        if os.path.exists("output.txt"):
            os.remove("output.txt")
        with open("output.txt","w") as f:
            for sentence in correct_sentences:
                f.write(sentence + "\n")
        f.close()
        print("Finished.")
        # except:
        #     print("Error: Cannot open text file.")
        #     continue

if __name__ == '__main__':
    main()
