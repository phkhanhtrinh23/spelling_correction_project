import torch
import re
import random
from misspelling import make_misspelling

class DataCollatorCustom:
    def __init__(self,
                filename,
                tokenizer,
                max_length=256,
                threshold=0.95,
#                 is_inference=False,
                device="cuda",
                ):
        self.tokenizer = tokenizer
        self.filename = filename
        self.max_length = max_length
        self.threshold = threshold
#         self.is_inference = bool(is_inference)
        self.device = device
        
#         self.load_dataset()
        
#     def __len__(self):
#         return len(self.input_ids)
            
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         input_ids = self.input_ids[idx, :]
#         samples = {
#             'input_ids': input_ids,
#         }
#         if self.is_inference is False:
#             samples['attention_mask'] = self.attention_mask[idx, :]
#             samples['labels'] = self.labels[idx, :]
#         return samples

    def load_dataset(self):
        tokens_list, labels_list = [], []
        attention_masks_list = []
        encodings = {}
        with open(self.filename, "r", encoding="utf-8") as f:
            self.sentence_list = []
            for sentence in f.readlines():
                # Remove unused character or punctuation
                sentence = re.sub(r"[^0-9a-zA-Z' -]+","", sentence)
                self.sentence_list.append(sentence)
                
        for i, sentence in enumerate(self.sentence_list):
            # Make misspelling sentence
            corrupted = make_misspelling(sentence, self.threshold)

            # Format the inputs
            tokens, masks, labels = self.formatting(corrupted, sentence)
#             labels = self.formatting(sentence)
            tokens_list.append(tokens)
            attention_masks_list.append(masks)
            labels_list.append(labels)
            
#             if i == 1:
#                 break

#         sentences = [self.tokenizer.decode(tokens) for tokens in tokens_list]
#         labels = [self.tokenizer.decode(labels) for labels in labels_list]
        
#         encodings = self.tokenizer(
#             sentences, return_tensors='pt', truncation=True,
#             padding='max_length', max_length=self.max_length)
        
#         label_encodings = self.tokenizer(
#             labels, return_tensors='pt', truncation=True,
#             padding='max_length', max_length=self.max_length)
        
        encodings["input_ids"] = torch.tensor(tokens_list)
        encodings["attention_mask"] = torch.tensor(attention_masks_list)

#         encodings = self.tokenizer(
#             tokens_list, return_tensors='pt', truncation=True,
#             padding='max_length', max_length=self.max_length)
        encodings["labels"] = torch.tensor(labels_list)
        
#         print("encodings:\n", encodings)

#         self.input_ids = encodings['input_ids']
#         self.attention_mask = encodings['attention_mask']

#         if self.is_inference is False:
#             self.labels = torch.tensor(labels_list, dtype=torch.long)

        return encodings

    def formatting(self, input_text, target_text):
        input_tokens = self.tokenizer(input_text)["input_ids"]
        target_tokens = self.tokenizer(target_text)["input_ids"]

#         tokens = [self.tokenizer.bos_token_id] + input_tokens \
#                 + [self.tokenizer.eos_token_id]

#         labels = target_tokens + [self.tokenizer.eos_token_id] \
#             + [-100] * (self.max_length - len(target_tokens) - 1)
#         labels = labels[:self.max_length]

#         input_tokens = self.tokenizer.encode(input_text)
#         target_tokens = self.tokenizer.encode(target_text)

        tokens = [self.tokenizer.bos_token_id] + input_tokens \
            + [self.tokenizer.sep_token_id] + target_tokens \
            + [self.tokenizer.eos_token_id]
        tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        tokens = tokens[:self.max_length]
        
        attention_masks = [1] + [1] * len(input_tokens) + [1] \
                        + [1] * len(target_tokens)  + [1] \
                        + [0] * self.max_length
        attention_masks = attention_masks[:self.max_length]

        labels = [-100] * (len(input_tokens) + 2) \
            + target_tokens + [self.tokenizer.eos_token_id] \
            + [-100] * self.max_length
        labels = labels[:self.max_length]
        
        return tokens, attention_masks, labels
        
#         labels = target_tokens + [self.tokenizer.eos_token_id] \
#             + [-100] * (self.max_length - len(target_tokens))
#         labels = labels[:self.max_length]

#         return labels