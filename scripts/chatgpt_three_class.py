import argparse
import json
import random
import openai
import requests
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from util import log

prompt = """根据证据判断给定声明的真实性：SUPPORTS表示证据支持声明，REFUTES表示证据反驳声明，NOT ENOUGH INFORMATION表示根据证据无法判断声明真实性
声明: [CLAIM]
证据: [EVIDENCE]
标签: 。
"""

label_map = {
    'true': 0, 'support': 0, 'supports': 0, 'yes': 0,
    'false': 1, 'refute': 1, 'refutes': 1, 'no': 1,
    'not enough information': 2, 'nei': 2, 'it is impossible to say': 2, 'unknown': 2, '无法判断': 2}

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

# url = "https://api.kksj.org/v1/chat/completions"
# headers = {
#     "Authorization": "sk-jXqmzArPu5eXz9UR9aA7F625F64d4eF7BcC084F8366b4c72",
#     "Content-Type": "application/json"
# }
# def llm(prompt):
#     data = {
#         "model": "gpt-3.5-turbo",
#         "messages": [{"role": "system", "content": "You are a fact-checker."},
#                      {"role": "user", "content": prompt}]
#     }
#     response = requests.post(url, headers=headers, json=data)
#     return response.json()['choices'][0]['message']['content']

def main(args):
    logger = log.get_logger(args.log_path)

    f_tmp = open(args.tmp_path, 'w', encoding='utf-8')

    # load data
    logger.info("loading data...")
    with open(args.data_path, 'r', encoding='utf-8') as fin:
        dataset = json.load(fin)
    
    # define llm
    define_gpt()
    
    predictions = []
    targets = []
    results = []
    # verify each sample
    for data in tqdm(dataset):
        claim = data['claim']
        evidence = " ".join(data['gold_evidence'])
        prompt_text = prompt.replace('[CLAIM]', claim)
        prompt_text = prompt_text.replace('[EVIDENCE]', evidence)

        try:
            output = llm(prompt_text)
            pred = output.lower().strip()
            print("id: {}, claim: {}, prediction: {}".format(data['id'], claim, pred), file=f_tmp)

            if pred in label_map:
                pred = label_map[pred]
            else:
                logger.info("Alert! Prediction failed! claim: {}, prediction: {}".format(claim, pred))
                pred = random.sample([0, 1, 2], 1)[0]
        except:
            logger.info("Generation failed! claim: {}".format(claim))
            pred = random.sample([0, 1, 2], 1)[0]
        
        label = 0 if data['label'] == "SUPPORTS" else 1

        predictions.append(pred)
        targets.append(label)
        
        results.append({
            'id': data['id'],
            'claim': claim,
            'gold_evidence': evidence,
            'gold_label': label,
            'pred_label': pred
        })

    f_tmp.close()
    with open(args.out_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=2, ensure_ascii=False)
    
    acc = accuracy_score(targets, predictions)
    _, _, micro_f1, _ = precision_recall_fscore_support(targets, predictions, average='micro')
    pre, recall, macro_f1, _ = precision_recall_fscore_support(targets, predictions, average='macro')
    print("         Accuracy: {:.3%}".format(acc))
    print("       F1 (micro): {:.3%}".format(micro_f1))
    print("Precision (macro): {:.3%}".format(pre))
    print("   Recall (macro): {:.3%}".format(recall))
    print("       F1 (macro): {:.3%}".format(macro_f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("使用ChatGPT测试在CHEF上三分类效果")
    parser.add_argument("--log_path", type=str, default="./logs/chatgpt_CHEF_three_class.log")
    parser.add_argument("--data_path", type=str, default="./data/processed/test_3.json")
    parser.add_argument("--out_path", type=str, default="./results/verify_CHEF_three_class_with_chatgpt.json")
    parser.add_argument("--tmp_path", type=str, default="./tmp/chatgpt_CHEF_three_class.txt")

    # parser.add_argument("--log_path", type=str, default="./logs/chatgpt_improved_CHEF_three_class.log")
    # parser.add_argument("--data_path", type=str, default="./data/improved_CHEF_2/test.json")
    # parser.add_argument("--out_path", type=str, default="./results/verify_improved_CHEF_two_class_with_chatgpt.json")
    # parser.add_argument("--tmp_path", type=str, default="./tmp/chatgpt_improved_CHEF_two_class2.txt")

    args = parser.parse_args()
    main(args)
