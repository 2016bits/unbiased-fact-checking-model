import json
import argparse
import time

from tqdm import trange, tqdm
from nltk import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from util import log

def generate_negative_claims(args, dataset):
    # for generating negative claims
    tokenizer = AutoTokenizer.from_pretrained(args.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.cache_dir)
    model = model.cuda()

    for i in trange(0, len(dataset), args.batch_size):
        claims = [d['translated_claim'] for d in dataset[i:i+args.batch_size]]
        batch = tokenizer(claims, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        out = model.generate(batch['input_ids'].to('cuda'), num_beams=5, max_length=1024)
        refuted_sents = tokenizer.batch_decode(out, skip_special_tokens=True)
        for j, refuted in enumerate(refuted_sents):
            dataset[i + j]['claim_refuted'] = refuted
    return dataset

def main(args):
    logger = log.get_logger(args.log_path)

    for datatype in ['train', 'dev', 'test']:
        logger.info("loading data...")
        in_path = args.in_path.replace('[DATA]', datatype)
        with open(in_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        logger.info("generating negative claims for supported examples...")
        negative_claim_dataset = generate_negative_claims(args, dataset)

        out_path = args.out_path.replace('[DATA]', datatype)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(negative_claim_dataset, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='./logs/generate_negative_data_for_supported.log')
    parser.add_argument('--in_path', type=str, default='./data/symmetric-CHEF/translated_supported_[DATA].json')
    parser.add_argument('--out_path', type=str, default='./data/symmetric-CHEF/supported_negative_[DATA].json')
    parser.add_argument('--datatype', type=str, default='train')

    parser.add_argument('--cache_dir', type=str, default='/data/yangjun/fact/debias/minwhoo/bart-base-negative-claim-generation')
    parser.add_argument('--zh_cache_dir', type=str, default='/data/yangjun/tools/Helsinki-NLP/opus-mt-zh-en')
    parser.add_argument('--en_cache_dir', type=str, default='/data/yangjun/tools/Helsinki-NLP/opus-mt-en-zh')

    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)
