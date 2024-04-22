# python environment: api
import json
import argparse
import random
import openai
from tqdm import tqdm

claim_prompt = """根据证据改写声明，使得改写后的新声明与证据之间表达主题和原声明与证据之间表达的主题相反
原声明: [CLAIM]
证据: [EVIDENCE]
新声明: 
"""
evidence_prompt = """根据声明改写证据，使得改写后的新证据与声明之间表达主题和原证据与声明之间表达主题相反
声明: [CLAIM]
原证据: [EVIDENCE]
新证据: 
"""

def define_gpt():
    # chatgpt api
    openai.api_type = 'azure'
    openai.api_base = 'https://1027gpt.openai.azure.com/'
    openai.api_version = '2023-03-15-preview'
    openai.api_key = '2d568b2bb17b4e02a2b424783e313176'

def llm(prompt, stop=["\n"]):
    response = openai.ChatCompletion.create(
            engine="ChatGPT",
            temperature=0,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
            messages=[{"role":"user","content":prompt}]
        )
    return response.choices[0].message.content

def main(args):
    f_tmp = open(args.tmp_path, 'w', encoding='utf-8')
    define_gpt()

    with open(args.in_path, 'r', encoding='utf-8') as f:
        raws = json.load(f)
    random.shuffle(raws)

    # control example number
    
    processed = []
    count = 0
    for data in tqdm(raws):
        
        original_id = data['id']
        if type(original_id) == str:
            original_id = eval(original_id)
        original_claim = data['claim']
        original_evidence = " ".join(data['gold_evidence'])
        if not original_evidence:
            # evidence is null
            continue
        original_label = data['label']

        if original_label == "SUPPORTS" or original_label == "REFUTES":
            continue

        elif original_label == "NOT ENOUGH INFO":
            original_label = 2
            rewritten_label = 2
        
        # prepare prompt
        claim_text = claim_prompt.replace("[CLAIM]", original_claim)
        claim_text = claim_text.replace("[EVIDENCE]", original_evidence)
        evidence_text = evidence_prompt.replace("[EVIDENCE]", original_evidence)
        evidence_text = evidence_text.replace("[CLAIM]", original_claim)
        
        # rewritte by ChatGPT
        write_claim_flag = True
        write_evidence_flag = True
        try:
            rewritten_claim = llm(claim_text)
        except:
            write_claim_flag = False
            print("generate claim {} failed".format(original_id))
        
        if not write_claim_flag or not write_evidence_flag:
            # failed
            continue

        # generate samples
        # original claim + original evidence
        data_oo = {
            'id': 100000 + original_id,
            'claim': original_claim,
            'label': original_label,
            'gold_evidence': original_evidence
        }
        print(data_oo, file=f_tmp)
        processed.append(data_oo)
        count += 1

        if write_claim_flag:
            # rewritten claim + original evidence
            data_ro = {
                'id': 200000 + original_id,
                'claim': rewritten_claim,
                'label': rewritten_label,
                'gold_evidence': original_evidence
            }
            print(data_ro, file=f_tmp)
            processed.append(data_ro)
            count += 1
    
    with open(args.out_path, 'w', encoding='utf-8') as fout:
        json.dump(processed, fout, indent=2, ensure_ascii=False)
    print("Finished!\nGenerate {} examples.".format(count))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default='/data/yangjun/fact/debias/data/improved_CHEF_3/test.json')
    parser.add_argument("--out_path", type=str, default='/data/yangjun/fact/debias/data/gpt/sysmmetric_test_3_nei.json')
    parser.add_argument("--tmp_path", type=str, default="./tmp/out_three_class_test_nei.txt")

    args = parser.parse_args()
    main(args)
