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

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers.integrations import TensorBoardCallback
from torch.utils.data import DataLoader
from collator import DataCollatorCustom
from transformers import default_data_collator, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import evaluate
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# def compute_metrics_custom(eval_preds):
#     metric = evaluate.load("glue", "mrpc")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

def main(args):
    device = args.device
    batch_size = args.batch_size

    # # Data preprocessing
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    special_tokens_dict = {'sep_token': '[SEP]'}
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dataset = DataCollatorCustom(filename=args.data_path,
                                       tokenizer=tokenizer,
                                       max_length=args.max_length,
                                       threshold=args.threshold)
    train_dataset = datasets.Dataset.from_dict(dataset.load_dataset())
    valid_dataset = datasets.Dataset.from_dict(dataset.load_dataset())

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )
    valid_dataloader = DataLoader(
        valid_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )
    print("Input:", train_dataset["input_ids"][:10])
    print("Label:", train_dataset["labels"][:10])
    print("Length:", len(train_dataset["input_ids"][0]), \
        len(train_dataset["labels"][0]))
    
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
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
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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
#             model_path = os.path.join(args.model_name_or_path + "_new", f'model_{epoch}.pt')
#             torch.save({"model_state_dict": lora.lora_state_dict(model)}, model_path)
#             tokenizer.save_pretrained(args.model_name_or_path + "_new")
            model.save_pretrained(args.model_name_or_path + "_new")
            tokenizer.save_pretrained(args.model_name_or_path + "_new")
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

#     special_tokens_dict = {'sep_token': '[SEP]'}
#     tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
#     tokenizer.add_special_tokens(special_tokens_dict)
#     tokenizer.pad_token = tokenizer.eos_token
    
#     gpt2_model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
#     gpt2_model.resize_token_embeddings(len(tokenizer))
#     gpt2_model = gpt2_model.to(args.device)
    
#     with open(args.data_path, "r", encoding="utf-8") as f:
#         sentence_list = []
#         for sentence in f.readlines():
#             sentence_list.append(sentence)

#         random.shuffle(sentence_list)
#         idx = int(len(sentence_list) * 0.8)
#         train_data = sentence_list[:idx]
#         valid_data = sentence_list[idx:]
    
#     with open("data/train.txt", "w", encoding="utf-8") as f:
#         for sentence in train_data:
#             f.write(sentence)
#     f.close()
    
#     with open("data/valid.txt", "w", encoding="utf-8") as f:
#         for sentence in valid_data:
#             f.write(sentence)
#     f.close()

# #     for epoch in range(args.num_epochs):
# #         print("Generating new data set at each epoch...")
#     train_dataset = DataCollatorCustom(filename="data/train.txt",
#                                        tokenizer=tokenizer,
#                                        max_length=args.max_length,
#                                        threshold=args.threshold)
#     valid_dataset = DataCollatorCustom(filename="data/valid.txt",
#                                        tokenizer=tokenizer,
#                                        max_length=args.max_length,
#                                        threshold=args.threshold)
# #         print("Finished generating data.")

#     if os.path.exists(args.summary_dir):
#         shutil.rmtree(args.summary_dir, ignore_errors=True)
        
#     if hasattr(args, 'summary_dir'):
#         summary_writer = SummaryWriter(args.summary_dir)

#     training_args = TrainingArguments(
#         output_dir=args.save_dir,
#         num_train_epochs=args.num_epochs,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.eval_batch_size,
#         gradient_accumulation_steps=args.gradient_accumulation,
#         learning_rate=args.learning_rate,
#         warmup_steps=300,  # warmup_steps=gpt_model.num_warmup_steps,
#         weight_decay=0.01,
#         save_total_limit=2,
# #         save_steps=args.save_steps,
# #         eval_steps=args.save_steps,
# #         logging_steps=1000,
#         save_strategy='epoch',
# #         evaluation_strategy='steps',
#         eval_accumulation_steps=32,
#         seed=args.seed,
#         dataloader_num_workers=8,
#         report_to='tensorboard',
#         logging_dir=args.summary_dir,
#     )

#     trainer = Trainer(
#         model=gpt2_model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         data_collator=default_data_collator,
#         callbacks=[TensorBoardCallback(tb_writer=summary_writer)],
#         compute_metrics=compute_metrics_custom,
#     )

#     trainer.train()
#     trainer.save_model()
#     trainer.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./data/english.txt',
                        help='Train dataset file')

    parser.add_argument('--log', type=str,
                        default='./logging_files/train_{datetime}.log',
                        help='Log filename')
    parser.add_argument('--device', type=str, default='cuda',
                        help='{cuda, cpu}')
#     parser.add_argument('--save_dir', type=str,
#                         help='Path to SAVE model checkpoint',
#                        default='model_weights/')
#     parser.add_argument('--summary_dir', type=str,
#                         help='Path to save tensorboard summary',
#                        default="logs/")

    parser.add_argument('--model_name_or_path', type=str, default="gpt2-medium",
                        help='pretrained model name')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum number of tokens for each sequence')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Number of update steps to accumulate the gradients')

    parser.add_argument('--learning_rate', type=float, default=2.5e-5,
                        help='Learning rate of fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Threshold to add a misspelling character to a sentence')
#     parser.add_argument("--max_grad_norm", default=1.0, type=float, 
#                         help="Max gradient norm.")
#     parser.add_argument('--save_steps', type=int, default=1000,
#                         help='Number of update steps before eval & save')

    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2023)
    
    # ========================= LoRA CONFIGURATION ==============================
#     parser.add_argument('--lora_dim', type=int, default=8, help='lora attn dimension')
#     parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')
#     parser.add_argument('--lora_dropout', default=0.1, type=float, help='dropout probability for lora layers')
#     parser.add_argument('--label_smooth', default=0.1, type=float, help='label smoothing')
#     parser.add_argument('--init_checkpoint', default="pretrained_checkpoints/pytorch_model.bin", help='pretrained checkpoint path')
    # ===========================================================================

    args = parser.parse_args()

    if os.path.exists("logging_files/") == False:
        os.mkdir("logging_files/")

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