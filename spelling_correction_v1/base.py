import torch
from transformer import get_model
from preprocess import *
from beam_search import beam_search
from torch.autograd import Variable

class ArgumentOpt:
    def __init__(self):
        self.load_weights=True
        self.k=3
        self.src_lang="en_core_web_sm"
        self.trg_lang="en_core_web_sm"
        self.d_model=512
        self.n_layers=6
        self.heads=8
        self.dropout=0.1
        self.max_strlen=300
        self.cuda=True
        self.cuda_device="cuda"

class SpellingCorrection:
    def __init__(self):
        self.opt = ArgumentOpt()
        self.SRC, self.TRG = create_files(self.opt)
        self.model = get_model(self.opt, len(self.SRC.vocab), len(self.TRG.vocab))

    def translate_sentence(self, sentence):
        self.model.eval()
        indexed = []
        sentence = preprocess(sentence)
        for tok in sentence:
            indexed.append(self.SRC.vocab.stoi[tok])
        sentence = Variable(torch.LongTensor([indexed]))
        if self.opt.cuda == True:
            sentence = sentence.to(self.opt.cuda_device)
        sentence = beam_search(sentence, self.model, self.SRC, self.TRG, self.opt)
        return sentence.capitalize()

    def __call__(self, sentence):
        return self.translate_sentence(sentence)