import json
import argparse
import time

from tqdm import trange, tqdm
from nltk import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from util import log


def process_evidence(raws, tokenizer):
    evidence_list = ""
    for index in raws:
        evidence = raws[index]['text']
        if evidence and len(tokenizer(evidence_list + evidence)['input_ids']) < 512:
            evidence_list += evidence + ' '
    return evidence_list

def translate(text, tokenizer, model):
    batch = model.generate(**tokenizer(text, padding=True, max_length=512, truncation=True, return_tensors="pt").to('cuda'))
    translated_text = tokenizer.batch_decode(batch, skip_special_tokens=True)
    return translated_text

def main(args):
    logger = log.get_logger(args.log_path)

    # for Chinese to English translation
    logger.info("Loading Chinese to English Translation Model...")
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_cache_dir)
    zh_model = AutoModelForSeq2SeqLM.from_pretrained(args.zh_cache_dir)
    zh_model = zh_model.cuda()

    for datatype in ['train', 'dev', 'test']:
        logger.info("loading data...")
        in_path = args.in_path.replace('[DATA]', datatype)
        dataset = []
        with open(in_path, 'r', encoding='utf-8') as f:
            raws = json.load(f)

        for data in raws:
            label = data['label']
            # if label != 0:
            #     continue
            id = data['claimId']
            if type(id) == str:
                id = eval(id) 
            claim = data['claim']
            evidence = process_evidence(data['gold evidence'], zh_tokenizer)
            # if evidence == '':
            #     continue

            dataset.append({
                "id": id,
                "claim": claim,
                "evidence": evidence,
                "label": label,
            })
        
        logger.info("generating negative claims...")

        for i in trange(0, len(dataset), args.batch_size):
            claims = [d['claim'] for d in dataset[i:i+args.batch_size]]
            translated_claims = translate(claims, zh_tokenizer, zh_model)
            for j, trans in enumerate(translated_claims):
                dataset[i + j]['translated_claim'] = trans
            
            evidence = [d['evidence'] for d in dataset[i:i+args.batch_size]]
            translated_evidence = translate(evidence, zh_tokenizer, zh_model)
            for j, trans in enumerate(translated_evidence):
                dataset[i + j]['translated_evidence'] = trans

        out_path = args.out_path.replace('[DATA]', datatype)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='./logs/translated_CHEF_data.log')
    parser.add_argument('--in_path', type=str, default='./data/raw/[DATA].json')
    parser.add_argument('--out_path', type=str, default='./data/symmetric-CHEF/translated_[DATA].json')
    # parser.add_argument('--datatype', type=str, default='dev')

    parser.add_argument('--cache_dir', type=str, default='/data/yangjun/fact/debias/minwhoo/bart-base-negative-claim-generation')
    parser.add_argument('--zh_cache_dir', type=str, default='/data/yangjun/tools/Helsinki-NLP/opus-mt-zh-en')
    parser.add_argument('--en_cache_dir', type=str, default='/data/yangjun/tools/Helsinki-NLP/opus-mt-en-zh')

    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(args)
