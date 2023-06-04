import os
import re
import pandas as pd
import spacy
from torchtext.legacy import data
from generate_input import *
from iterator import *
import pickle

class CustomTokenizer(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(r"[^0-9a-zA-Z' -]+","", sentence)
        sentence = sentence.lower()
        # return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
        return [self.nlp.tokenizer(char).text for char in sentence]

def preprocess(sentence):
    sentence = re.sub(r"[^0-9a-zA-Z' -]+","", sentence)
    sentence = sentence.lower()
    return sentence

def create_files(opt):
    t_src = CustomTokenizer(opt.src_lang)
    t_trg = CustomTokenizer(opt.trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)
    
    if opt.load_weights is True:
        try:
            print("Loading presaved SRC and TRG files...")
            SRC = pickle.load(open(f'weights/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'weights/TRG.pkl', 'rb'))
            print("Finished.")
        except:
            print("Error: Cannot open SRC.pkl and TRG.pkl files.")
            quit()

    return SRC, TRG

def create_data(opt, SRC, TRG, repeat=0):
    if repeat == 0:
        print("Generating new dataset...")
    else:
        print("Generating new dataset for the next epoch...")

    opt.src_data = generate_input(opt)

    if opt.data_file is not None and repeat == 0:
        try:
            opt.trg_data = open(opt.data_file).read().strip().split('\n')
        except:
            print("Error: '" + opt.data_file + "' file not found.")
            quit()
    
    t_src = CustomTokenizer(opt.src_lang)
    t_trg = CustomTokenizer(opt.trg_lang)

    raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    if os.path.exists('temp.csv'):
        os.remove('temp.csv')

    df.to_csv("temp.csv", index=False)
    
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./temp.csv', format='csv', fields=data_fields)

    train_iter = CustomIterator(
        train,
        batch_size=opt.batch_size, 
        device=opt.cuda_device,
        repeat=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        batch_size_fn=batch_size_fn,
        train=True,
        shuffle=True
        )
    
    SRC.build_vocab(train)
    TRG.build_vocab(train)

    if opt.checkpoint > 0:
        if os.path.exists('weights/SRC.pkl'):
            os.remove('weights/SRC.pkl')

        if os.path.exists('weights/TRG.pkl'):
            os.remove('weights/TRG.pkl')

        try:
            print("\tSaving SRC and TRG files...")
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
            print("\tFinished saving SRC and TRG files.")
        except:
            print("Error: Saving data to ./weights/<filename>.pkl is not successful.")
            quit()
    
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_length(train_iter)

    print("Finished.")

    return train_iter

def get_length(train):
    for i, b in enumerate(train):
        pass
    return i
