import argparse
import json
import jieba.posseg as pseg

def process_evidence(raws):
    evidence_text = ""
    for index in raws:
        evidence = raws[index]['text']
        if evidence:
            evidence_text += evidence
    return evidence_text

# 常见中文否定词列表
neg_words = ['不', '没', '无', '非', '莫', '未', '别', '毋', '否', '未', '弗', '勿', '无', '甭', '毫', '决不', '没', '未曾', '不曾']

def has_negative_word(sentence):
    # 使用jieba进行分词和词性标注
    words = pseg.cut(sentence)
    
    # 检查分词后的词性，查找是否含有否定词
    for word, flag in words:
        if word in neg_words:
            return 1
    
    return 0

def calculate_nn_ratio(dataset):
    neg_claim = 0
    nn_claim = 0
    neg_evidence = 0
    nn_evidence = 0

    for data in dataset:
        claim = data['claim']
        evidence = process_evidence(data['evidence'])
        label = data['label']
        if label == 2:
            continue

        if has_negative_word(claim) == 1:
            neg_claim += 1
            if label == 1:
                nn_claim += 1
        if has_negative_word(evidence) == 1:
            neg_evidence += 1
            if label == 1:
                nn_evidence += 1

    print("neg_claim:", neg_claim)
    print("nn_claim:", nn_claim)
    print("neg_evidence:", neg_evidence)
    print("nn_evidence:", nn_evidence)
    print("neg_claim/nn_claim:", neg_claim / nn_claim)
    print("neg_evidence/nn_evidence:", neg_evidence / nn_evidence)

def main(args):
    print("**********************{}************************".format(args.data_type))
    in_path = args.in_path.replace('[DATA]', args.data_type)
    with open(in_path, 'r') as f:
        dataset = json.load(f)

    # analyze the sentiment of the claim and evidence, and analyze sentiment relationship between claim and label, evidence and label
    claim_sent = [0, 0]
    evidence_sent = [0, 0]
    claim_consistency = [0, 0]
    evidence_consistency = [0, 0]

    for data in dataset:
        claim = data['claim']
        evidence = process_evidence(data['evidence'])
        label = data['label']

        if label == 2:
            continue

        c_sent = has_negative_word(claim)
        e_sent = has_negative_word(evidence)
        claim_sent[c_sent] += 1
        evidence_sent[e_sent] += 1
        if c_sent == label:
            claim_consistency[0] += 1
        else:
            claim_consistency[1] += 1
        if e_sent == label:
            evidence_consistency[0] += 1
        else:
            evidence_consistency[1] += 1
    print("claim_sentiment:", claim_sent)
    print("evidence_sentiment:", evidence_sent)
    print("claim_consistency:", claim_consistency)
    print("evidence_consistency:", evidence_consistency)

    print("**********************")
    calculate_nn_ratio(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='./data/raw/[DATA].json')
    parser.add_argument('--data_type', type=str, default='train')
    args = parser.parse_args()
    main(args)
