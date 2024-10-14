import torch
import json
import argparse

from tqdm import tqdm, trange
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset

from utils import log

class ClaimDataset(Dataset):
    def __init__(self, id_list, evidence_list, claim_list, tokenizer, max_length):
        self.id_list = id_list
        self.evidence_list = evidence_list
        self.claim_list = claim_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.evidence_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]
        evidence = self.evidence_list[idx]
        claim = self.claim_list[idx]

        input_text = f"evidence: {evidence} refuted claim: {claim}"

        # encode input and target text
        inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        return {
            "id": id,
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
        }

def translate(text, tokenizer, model):
    batch = model.generate(**tokenizer(text, padding=True, max_length=512, truncation=True, return_tensors="pt").to('cuda'))
    translated_text = tokenizer.batch_decode(batch, skip_special_tokens=True)
    return translated_text

def main(args):
    logger = log.get_logger(args.log_path)

    logger.info("loading model...")
    # corrector model and tokenizer
    # corrector_tokenizer = T5Tokenizer.from_pretrained(args.corrector_cache_dir)
    # corrector_model = T5ForConditionalGeneration.from_pretrained(args.corrector_cache_dir)
    corrector_tokenizer = T5Tokenizer.from_pretrained(args.fine_tuned_corrector_path)
    corrector_model = T5ForConditionalGeneration.from_pretrained(args.fine_tuned_corrector_path)
    corrector_model = corrector_model.cuda()
    state_dict = torch.load(args.corrector_checkpoint)
    corrector_model.load_state_dict(state_dict)
    corrector_model.eval()

    # translater model and tokenizer
    translate_tokenizer = AutoTokenizer.from_pretrained(args.translate_cache_dir)
    translate_model = AutoModelForSeq2SeqLM.from_pretrained(args.translate_cache_dir)
    translate_model = translate_model.cuda()

    for datatype in ["train", "dev", "test"]:    
        logger.info("loading data...")
        data_path = args.data_path + datatype + ".json"
        
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        with open(args.supported_negative_data_path + datatype + ".json", 'r', encoding='utf-8') as f:
            supported_negative_data = json.load(f)
        supported_negative_dict = {d["id"]: d["claim_refuted"] for d in supported_negative_data[:args.num_sample]}

        logger.info("correcting claims in refuted examples...")
        refuted_examples = [d for d in dataset[:args.num_sample] if d['label'] == 1]
        refuted_data = ClaimDataset(
            [example['id'] for example in refuted_examples],
            [example['translated_evidence'] for example in refuted_examples],
            [example['translated_claim'] for example in refuted_examples],
            corrector_tokenizer,
            args.max_length
        )
        dataloader = DataLoader(refuted_data, batch_size=args.batch_size, shuffle=False)
        corrected_refuted_claim_dict = {}
        for batch in tqdm(dataloader):
            with torch.no_grad():
                ids = batch['id']
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                outputs = corrector_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=args.max_length)
                corrected_claims = corrector_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for id, claim in zip(ids, corrected_claims):
                    corrected_refuted_claim_dict[id] = claim
        
        logger.info("translating claims into Chinese...")
        translate_examples = []
        for data in tqdm(dataset[:args.num_sample]):
            if data['label'] == 1:
                translate_examples.append({
                    'id': data['id'],
                    'text': corrected_refuted_claim_dict[data['id']],
                })
            elif data['label'] == 0:
                translate_examples.append({
                    'id': data['id'],
                    'text': supported_negative_dict[data['id']],
                })
        translated_dict = {}
        for i in trange(0, len(translate_examples), args.batch_size):
            ids = [d['id'] for d in translate_examples[i:i+args.batch_size]]
            claims = [d['text'] for d in translate_examples[i:i+args.batch_size]]
            translated_claims = translate(claims, translate_tokenizer, translate_model)
            for id, text in zip(ids, translated_claims):
                translated_dict[id] = text
        
        logger.info("generate symmetric data...")
        results = []
        for data in tqdm(dataset[:args.num_sample]):
            if data['label'] == 1:
                # for refuted claims
                # original claim-evidence pair
                results.append({
                    'id': 100000 + data['id'],
                    'zh_claim': data['claim'],
                    'en_claim': data['translated_claim'],
                    'gold_evidence': data['evidence'],
                    'label': 0,
                })

                # corrected claim-evidence pair
                results.append({
                    'id': 200000 + data['id'],
                    'zh_claim': translated_claims[data['id']],
                    'en_claim': corrected_refuted_claim_dict[data['id']],
                    'gold_evidence': data['evidence'],
                    'label': 1,
                })

            elif data['label'] == 0:
                # for supported claims
                # original claim-evidence pair 
                results.append({
                    'id': 100000 + data['id'],
                    'zh_claim': data['claim'],
                    'en_claim': data['translated_claim'],
                    'gold_evidence': data['evidence'],
                    'label': 1
                })

                # corrected claim-evidence pair
                results.append({
                    'id': 200000 + data['id'],
                    'zh_claim': translated_claims[data['id']],
                    'en_claim': supported_negative_dict[data['id']],
                    'gold_evidence': data['evidence'],
                    'label': 1
                })

        logger.info("save data...")
        with open(args.data_path + datatype + ".json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./logs/generate_symmetric_chef.log")
    parser.add_argument("--data_path", type=str, default="./data/symmetric-CHEF/translated_")
    parser.add_argument("--supported_negative_data_path", type=str, default="./data/symmetric-CHEF/supported_negative_")
    parser.add_argument("--output_path", type=str, default="./data/symmetric-CHEF/symmetric_CHEF_")

    parser.add_argument("--corrector_cache_dir", type=str, default="/data/yangjun/tools/google/t5-base")
    parser.add_argument("--corrector_checkpoint", type=str, default="./models/best_corrector.pth")
    parser.add_argument("--fine_tuned_corrector_path", type=str, default="./model/fine_tuned_t5")

    parser.add_argument("--translate_cache_dir", type=str, default='/data/yangjun/tools/Helsinki-NLP/opus-mt-en-zh')

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_sample", type=int, default=-1)

    args = parser.parse_args()
    main(args)
