import numpy as np
import torch
import os
import argparse
import json
from datetime import datetime
import random
import logging
import os
import datasets
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from collator import DataCollatorCustom
from transformers import default_data_collator, get_linear_schedule_with_warmup

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def main(args):
    device = args.device
    batch_size = args.batch_size

    # data preprocessing
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    special_tokens_dict = {'sep_token': '[SEP]'}
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = DataCollatorCustom(filename=args.data_path,
                                       tokenizer=tokenizer,
                                       max_length=args.max_length,
                                       threshold=args.threshold)
    train_dataset = datasets.Dataset.from_dict(train_dataset.load_dataset())
    print("Input:", train_dataset["input_ids"][0])
    print("Label:", train_dataset["labels"][0])
    print("Length:", len(train_dataset["input_ids"][0]), \
        len(train_dataset["labels"][0]))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # training
    logging.info("==========Start training==========")
    comparative_loss = -1

    for epoch in tqdm(range(args.num_epochs), position=0, desc="Epoch", leave=False):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, position=1, desc="Training", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            lr_scheduler.step()
            optimizer.zero_grad()
        
        print("train_loss:", train_loss, len(train_dataloader), train_loss / len(train_dataloader))

        cummulative_loss = train_loss / len(train_dataloader)

        logging.info(f"Epoch {epoch+1}: \
                    train_loss: {cummulative_loss}")

        if comparative_loss == -1 or cummulative_loss < comparative_loss:
            # Update comparative_loss for later comparison
            comparative_loss = cummulative_loss
            
            # Saving model
            logging.info(f"Epoch {epoch+1}: Saving model and tokenizer...")
            model.save_pretrained(args.model_name_or_path)
            tokenizer.save_pretrained(args.model_name_or_path)
            logging.info(f"Epoch {epoch+1}: Done.")
        
        # Generate new train dataset after each epoch to diversify the training set
        logging.info(f"Generate new train dataset after each epoch...")
        train_dataset = DataCollatorCustom(filename=args.data_path,
                                       tokenizer=tokenizer,
                                       max_length=args.max_length,
                                       threshold=args.threshold)
        train_dataset = datasets.Dataset.from_dict(train_dataset.load_dataset())
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=batch_size,
            pin_memory=True,
            )
        logging.info(f"Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./data/english.txt',
                        help='Train dataset file')

    parser.add_argument('--log', type=str,
                        default='./logs/train_{datetime}.log',
                        help='Log filename')
    parser.add_argument('--device', type=str, default='cuda',
                        help='{cuda, cpu}')

    parser.add_argument('--model_name_or_path', type=str, default="gpt2-medium",
                        help='pretrained model name')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum number of tokens for each sequence')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')

    parser.add_argument('--learning_rate', type=float, default=2.5e-5,
                        help='Learning rate of fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Threshold to add a misspelling character to a sentence')
    parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                        help="Max gradient norm.")

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