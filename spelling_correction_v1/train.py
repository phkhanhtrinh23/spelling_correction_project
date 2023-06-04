import argparse
import time
import torch
from transformer import get_model
from scheduler import CosineWithWarmRestarts
from preprocess import *
from layers import *
import torch.nn.functional as F
from torchmetrics import F1Score

def train_model(model, opt, SRC, TRG):
    print("Training model...")
    model.train()

    if opt.checkpoint == True:
        cptime = time.time()

    if os.path.exists('weights/model'):
        os.remove('weights/model')
    
    if os.path.exists('weights/log_train.txt'):
        os.remove('weights/log_train.txt')
    
    f = open('weights/log_train.txt', 'w')
    f1_metric = F1Score(ignore_index=opt.trg_pad).to(opt.cuda_device)

    for epoch in range(opt.epochs):
        cptime = time.time()

        f.write(f"Epoch {epoch + 1}/{opt.epochs}\n")
        print((f"Epoch {epoch + 1}/{opt.epochs}:"))

        for _, batch in enumerate(opt.train): 
            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)

            if opt.cuda == True:
                src = src.to(opt.cuda_device)
                trg_input = trg_input.to(opt.cuda_device)
                src_mask = src_mask.to(opt.cuda_device)
                trg_mask = trg_mask.to(opt.cuda_device)

            preds = model(src, trg_input, src_mask, trg_mask)
            
            ys = trg[:, 1:].contiguous().view(-1)
            if opt.cuda == True:
                ys = ys.to(opt.cuda_device)

            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            f1_score = f1_metric(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            opt.optimizer.step()

            if opt.scheduler == True: 
                opt.sched.step()

        f.write(f"Time: {time.time() - cptime}.\n")
        print(f"Time: {time.time() - cptime}.")

        f.write(f"Loss: {loss.item()}\n")
        print(f"Loss: {loss.item()}")

        f.write(f"F1 Score: {f1_score}\n")
        print(f"F1 Score: {f1_score}")

        if opt.checkpoint == True:
            f.write(f"Save model after {epoch + 1} epoch(s).\n")
            print(f"Save model after {epoch + 1} epoch(s).")
            torch.save(model.state_dict(), 'weights/model')
        
        opt.train = create_data(opt, SRC, TRG, repeat=1)

        f.write("\n")
        print()
    
    f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=bool, default=False)
    parser.add_argument('-data_file', type=str, default="data/english.txt")
    parser.add_argument('-src_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-trg_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-cuda_device', type=str, default="cuda")
    parser.add_argument('-batch_size', type=int, default=8192)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-threshold', type=int, default=0.94)
    parser.add_argument('-max_strlen', type=int, default=300)
    parser.add_argument('-checkpoint', type=bool, default=True)
    parser.add_argument('-scheduler', type=bool, default=True)

    opt = parser.parse_args()

    SRC, TRG = create_files(opt)
    opt.train = create_data(opt, SRC, TRG)
    
    print(f'Length of SRC vocab: {len(SRC.vocab)}, TRG vocab: {len(TRG.vocab)}')
    
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.scheduler == True:
        opt.sched = CosineWithWarmRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint == True:
        print("Model is saved at the end of each epoch.")
    
    train_model(model, opt, SRC, TRG)

if __name__ == "__main__":
    main()
