from datetime import datetime
import json
import re
import torch
import argparse
import logging
import csv
import os
import random
import numpy as np
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, PeftConfig, PeftModel
from collator import DataCollatorCustom
from torch.utils.data import DataLoader
from transformers import default_data_collator
from misspelling import make_misspelling

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class EvalDataCollator:
    def __init__(self,
                filename,
                tokenizer,
                max_length=256,
                threshold=0.95,
                special_character="[SEP]",
                ):
        self.tokenizer = tokenizer
        self.filename = filename
        self.max_length = max_length
        self.threshold = threshold
        self.special_character = special_character

    def load_dataset(self):
        filename = self.filename
        info = dict()
        tokens_list, original_list, labels_list = [], [], []
        with open(filename, "r", encoding="utf-8") as f:
            for sentence in f.readlines():
                # Remove unused character or punctuation
                sentence = re.sub(r"[^0-9a-zA-Z' -]+","", sentence)

                # Make misspelling sentence
                corrupted = make_misspelling(sentence, self.threshold)

                # Format the inputs
                tokens = self.formatting(corrupted + self.special_character)
                tokens_list.append(tokens)
                original_list.append(corrupted)
                labels_list.append(sentence)

        sentences = [self.tokenizer.decode(tokens) for tokens in tokens_list]
        
        encodings = self.tokenizer(
            sentences, return_tensors='pt', truncation=True,
            padding='max_length', max_length=self.max_length)
        
        info["original"] = original_list
        info["labels"] = labels_list
        
        return encodings, info
    
    def formatting(self, input_text):
        input_tokens = self.tokenizer.encode(input_text)
        
        return input_tokens
    
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    special_tokens_dict = {'sep_token': '[SEP]'}
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    eval_dataset = EvalDataCollator(filename=args.data_path,
                                       tokenizer=tokenizer,
                                       max_length=args.max_length,
                                       threshold=args.threshold,
                                       special_character="[SEP]")
    eval_dataset, info = eval_dataset.load_dataset()
    eval_dataset = datasets.Dataset.from_dict(eval_dataset)
    eval_dataloader = DataLoader(
                eval_dataset, 
                collate_fn=default_data_collator,
                batch_size=args.batch_size, 
                pin_memory=True
            )
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(args.device)
    model.eval()

    with open(args.result_file, 'w', encoding="utf-8", newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(["input","prediction","reference"])

        with torch.no_grad():
            kwargs = {
                'num_return_sequences': args.num_return_sequences,
            }

            i = 0
            for batch in tqdm(eval_dataloader, position=0, desc="Inference", leave=False):
                originals = info["original"][i: i + len(batch["input_ids"])]
                labels = info["labels"][i: i + len(batch["input_ids"])]
                batch = {k: v.to(args.device) for k, v in batch.items()}
                output = model.generate(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"],
                                        pad_token_id=tokenizer.pad_token_id,
                                        **kwargs)
                
                output_text = tokenizer.batch_decode(output.detach().cpu().numpy(), skip_special_tokens=True)

                # print("Original:", originals)
                # print("Labels:", labels)
                # print("Output:", output_text)
                # logging.info(f'Output text: {output_text}')
                # logging.info(f'Label text: {label_text}')
                # logging.info("\n") # newline

                for original, output, label in zip(originals, output_text, labels):
                    writer.writerow([original, output, label])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./data/english.txt',
                        help='Dataset file')
    parser.add_argument('--result_file', type=str, 
                        default="output.csv",
                        help='Output file')

    parser.add_argument('--log', type=str,
                        default='./logs/inference_{datetime}.log',
                        help='Log filename')
    parser.add_argument('--device', type=str, default='cuda',
                        help='{cuda, cpu}')

    parser.add_argument('--model_name_or_path', type=str, default="gpt2-medium",
                        help='pretrained model name')
    parser.add_argument('--max_length', type=int, default=64,
                        help='Maximum number of tokens for each sequence')
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help='Maximum number of return sequences')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')

    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Threshold to add a misspelling character to a sentence')

    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2023)

    args = parser.parse_args()

    if os.path.exists("logs/") == False:
        os.mkdir("logs/")

    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = args.log

    logging.basicConfig(level=log_level, format=log_format,
                        filename=log_file.format(datetime=start_datetime.replace(':','-')))
    logging.getLogger().setLevel(log_level)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info(f'Parsed args: {json.dumps(dict(args.__dict__), indent=2)}')

    main(args)