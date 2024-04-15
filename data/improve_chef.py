# 改进数据集思路：沿用Conv-FFD中all_data.txt的SUPPORT和REFUTE证据，对于NEI证据，从evidence中根据和claim相似度选择top-5作为gold_evidence
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 将all_data.txt划分为train dev和test
def split_all(lines):
    i = 0
    train_data = []
    dev_data = []
    test_data = []
    for line in lines:
        words = line.split('\t')[-1].replace('\n', '')
        text_list = words.split('[SEP]')
        claim = text_list[0]
        evidence = text_list[1:]
        label = line.split('\t')[0]

        if i in range(0, 7226):
            # train data
            train_data.append({
                'claim': claim,
                'label': label,
                'gold_evidence': evidence
            })

        elif i in range(7226, 7892):
            # dev data
            dev_data.append({
                'claim': claim,
                'label': label,
                'gold_evidence': evidence
            })

        elif i in range(7892, 8558):
            # test data
            test_data.append({
                'claim': claim,
                'label': label,
                'gold_evidence': evidence
            })
        
        i += 1
    
    print("finish splitting all_data.txt !")
    return train_data, dev_data, test_data

# 定义计算相似度的函数
def calc_similarity(tokenizer, model, s1, s2, max_len):
    # 对句子进行分词，并添加特殊标记
    inputs = tokenizer([s1, s2], return_tensors='pt', padding=True, truncation=True, max_length=max_len).to("cuda")

    # 将输入传递给BERT模型，并获取输出
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # 计算余弦相似度，并返回结果
    sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return sim

def process_evidence(raws):
    evidence_list = []
    for index in raws:
        evidence = raws[index]['text']
        if evidence:
            evidence_list.append(evidence)
    return evidence_list

# 对于NEI样例，将train、dev和test中的evidence根据相似度进行提取得到gold_evidence
def extract_gold_evidence(data_path, data_type, data_list, tokenizer, model, max_len):
    data_path = data_path.replace("[DATA]", data_type)
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    for data in dataset:
        if data['label'] == 2:
            # 仅针对NEI样例
            claim = data['claim']
            gold_evidence = process_evidence(data['gold evidence'])
            if gold_evidence:
                # 原始gold_evidence不为空
                data_list.append({
                    'claim': claim,
                    'label': data['label'],
                    'gold_evidence': gold_evidence
                })
            else:
                # 选择和claim语义最相似的top-5作为gold_evidence
                raw_evidence = process_evidence(data['evidence'])
                evidence = []
                for evi in raw_evidence:
                    evidence += evi.split('。')
                
                similarities = []
                for sent in evidence:
                    prob = calc_similarity(tokenizer, model, claim, sent, max_len)
                    similarities.append((sent, prob))
                
                # 按相似度排序并获取top-5
                top_similar_sentences = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

                data_list.append({
                    'claim': claim,
                    'label': data['label'],
                    'gold_evidence': [sent[0] for sent in top_similar_sentences]
                })
    
    return data_list

# 将json数据转化成文本数据
def convert_json2txt(data_list, f_all):
    lines = []
    for data in data_list:
        label = data['label']
        claim = data['claim']
        evidence = data['gold_evidence']

        text = str(label) + "\t" + claim + "[SEP]" + "[SEP]".join(evi for evi in evidence)
        lines.append(text)
        print(text, file=f_all)
    return lines

# 生成数据字典
def create_dict(data_path, dict_path):
    # 清空文件内容
    with open(dict_path, 'w') as f:
        f.seek(0)
        f.truncate() 

    dict_set = set()
    # 读取全部数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 把数据生成一个元组
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for s in content:
            dict_set.add(s)
    # 把元组转换成字典，一个字对应一个数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 添加未知字符
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    end_dict = {"<pad>": i+1}
    dict_txt.update(end_dict)
    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))
        
    print("数据字典生成完成！")

# 根据字典分别得到train、dev和test的列表形式
def get_list(data_path, data_type, lines, dict_txt):
    data_path = data_path.replace("[DATA]", data_type)
    with open(data_path, 'w', encoding='utf-8') as f:
        for line in lines:
            words = line.split('\t')[-1].replace('\n', '')
            label = line.split('\t')[0]

            labs = ""
            for s in words:
                lab = str(dict_txt[s])
                labs = labs + lab + ','
            labs = labs[:-1]
            labs = labs + '\t' + label + '\n'

            f.write(labs)

def load_vocab(dict_path):
    fr = open(dict_path, 'r', encoding='utf8')
    vocab = eval(fr.read())   #读取的str转换为字典
    fr.close()

    return vocab

def main(args):
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(args.model_name, cache_dir=args.cache_dir).cuda()
    
    # 读取all_data.txt
    print("loading all_data.txt ...")
    with open(args.convffd_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 将all_data.txt划分为train dev和test
    print("splitting all_data.txt ...")
    train_data, dev_data, test_data = split_all(lines)
    
    # 对于NEI样例，将train、dev和test中的evidence根据相似度进行提取得到gold_evidence
    print("extrating gold_evidence...")
    train_data = extract_gold_evidence(args.raw_data_path, "train", train_data, tokenizer, model, args.max_len)
    dev_data = extract_gold_evidence(args.raw_data_path, "dev", dev_data, tokenizer, model, args.max_len)
    test_data = extract_gold_evidence(args.raw_data_path, "test", test_data, tokenizer, model, args.max_len)

    # 将train、dev和test中的claim、evidence和label转换为all_data.txt形式
    print("Converting claim_evidence...")
    f_all = open(args.all_data_txt_path, 'w', encoding='utf-8')

    train_lines = convert_json2txt(train_data, f_all)
    dev_lines = convert_json2txt(dev_data, f_all)
    test_lines = convert_json2txt(test_data, f_all)

    f_all.close()

    # 根据all_data.txt得到字典
    print("Getting dict...")
    create_dict(args.all_data_txt_path, args.dict_path)
    
    # 根据字典分别得到train、dev和test的列表形式
    print("Getting data list...")
    dict_txt = load_vocab(args.dict_path)
    get_list(args.processed_data_path, "train", train_lines, dict_txt)
    get_list(args.processed_data_path, "dev", dev_lines, dict_txt)
    get_list(args.processed_data_path, "test", test_lines, dict_txt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="/data/yangjun/fact/debias/data/raw/[DATA].json")
    parser.add_argument("--convffd_data_path", type=str, default="/data/yangjun/fact/debias/data/raw/all_data.txt")
    parser.add_argument("--processed_data_path", type=str, default="/data/yangjun/fact/debias/data/improved_CHEF/[DATA].txt")
    parser.add_argument("--all_data_txt_path", type=str, default="/data/yangjun/fact/debias/data/improved_CHEF/all_data.txt")
    parser.add_argument("--dict_path", type=str, default="/data/yangjun/fact/debias/data/improved_CHEF/dict.txt")

    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--cache_dir", type=str, default="./bert-base-chinese")
    parser.add_argument("--max_len", type=int, default=512)

    args = parser.parse_args()

    main(args)
