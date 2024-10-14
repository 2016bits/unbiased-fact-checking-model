import json
import argparse
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset

from utils import log

class ClaimDataset(Dataset):
    def __init__(self, evidence_list, refuted_claim_list, corrected_claim_list, tokenizer, max_length):
        self.evidence_list = evidence_list
        self.refuted_claim_list = refuted_claim_list
        self.corrected_claim_list = corrected_claim_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.evidence_list)

    def __getitem__(self, idx):
        evidence = self.evidence_list[idx]
        refuted_claim = self.refuted_claim_list[idx]
        corrected_claim = self.corrected_claim_list[idx]

        input_text = f"evidence: {evidence} refuted claim: {refuted_claim}"
        target_text = corrected_claim

        # encode input and target text
        inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

def main(args):
    logger = log.get_logger(args.log_path)

    tokenizer = T5Tokenizer.from_pretrained(args.cache_dir)

    logger.info("loading data...")
    train_data_path = args.data_path + "train.json"
    dev_data_path = args.data_path + "dev.json"

    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_dataset = json.load(f)
    with open(dev_data_path, "r", encoding='utf-8') as f:
        dev_dataset = json.load(f)
    
    train_data = ClaimDataset(
        [example['evidence'] for example in train_dataset],
        [example['claim_refuted'] for example in train_dataset],
        [example['claim'] for example in train_dataset],
        tokenizer,
        args.max_length
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_data = ClaimDataset(
        [example['evidence'] for example in dev_dataset],
        [example['claim_refuted'] for example in dev_dataset],
        [example['claim'] for example in dev_dataset],
        tokenizer,
        args.max_length
    )
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)

    logger.info("loading model...")
    model = T5ForConditionalGeneration.from_pretrained(args.cache_dir)
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    logger.info("start training...")
    best_val_loss = float('inf')
    for epoch in range(args.epoch):
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epoch}, Batch {i+1}/{len(train_loader)}, Loss: {total_loss/(i+1)}")

        with torch.no_grad():
            val_losses = []
            for batch in dev_loader:
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                labels = batch['labels'].to('cuda')

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_losses.append(outputs.loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"Validation Loss: {avg_val_loss}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save_pretrained(args.model_save_path)
                tokenizer.save_pretrained(args.model_save_path)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./logs/train_corrector.log")
    parser.add_argument("--data_path", type=str, default="./data/symmetric-CHEF/supported_negative_")

    parser.add_argument("--cache_dir", type=str, default="/data/yangjun/tools/google/t5-base")
    parser.add_argument("--model_save_path", type=str, default="./models/fine_tuned_t5")

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    main(args)

