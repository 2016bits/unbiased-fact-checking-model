import pandas as pd
import json
import jieba
from collections import defaultdict
from scipy.stats import chi2_contingency

def tokenize(sentence):
    return list(jieba.cut(sentence))

def process_evidence(raws):
    evidence_text = ""
    for index in raws:
        evidence = raws[index]['text']
        if evidence:
            evidence_text += evidence
    return evidence_text

# 计算词在不同标签中的分布差异
def compute_chi2(word_freq_table):
    chi2_scores = {}
    for word, freqs in word_freq_table.items():
        chi2, _, _, _ = chi2_contingency([freqs])
        chi2_scores[word] = chi2
    return chi2_scores

def analyze_word(data_path, data_type):
    data_path = data_path.replace("[DATA]", data_type)
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    claim_word_freq = defaultdict(lambda: defaultdict(int))
    evidence_word_freq = defaultdict(lambda: defaultdict(int))
    labels = set()
    max_token_len = 0
    claim_token_len_list = []
    evidence_token_len_list = []
    for data in  dataset:
        claim = data["claim"]
        evidence = process_evidence(data["evidence"])
        label = data["label"]
        labels.add(label)

        claim_tokens = tokenize(claim)
        evidence_tokens = tokenize(evidence)
        for token in claim_tokens:
            claim_word_freq[token][label] += 1
            token_len = len(token)
            claim_token_len_list.append(token_len)
            if token_len > max_token_len:
                max_token_len = token_len

        for token in evidence_tokens:
            evidence_word_freq[token][label] += 1
            token_len = len(token)
            evidence_token_len_list.append(token_len)
            if token_len > max_token_len:
                max_token_len = token_len

    print("max_token_len: {}".format(max_token_len))
    print("average_claim_token_len: {}".format(sum(claim_token_len_list) / len(claim_token_len_list)))
    print("average_evidence_token_len: {}".format(sum(evidence_token_len_list) / len(evidence_token_len_list)))

    claim_word_freq_table = {word: [claim_word_freq[word].get(label, 0) for label in labels] for word in claim_word_freq}
    evidence_word_freq_table = {word: [evidence_word_freq[word].get(label, 0) for label in labels] for word in evidence_word_freq}

    # 统计单词在所有单词中的的平均词频
    total_claim_word_freq = sum([sum(freqs) for freqs in claim_word_freq.values()])
    total_evidence_word_freq = sum([sum(freqs) for freqs in evidence_word_freq.values()])
    print("avg_claim_word_freq: {}".format(total_claim_word_freq/(3 * len(claim_word_freq))))
    print("avg_evidence_word_freq: {}".format(total_evidence_word_freq/(3 * len(evidence_word_freq))))

    claim_word_ratio = {}
    evidence_word_ratio = {}

    for word, freqs in claim_word_freq_table.items():
        total_freq = sum(freqs)
        if total_freq < 50:
            continue
        claim_word_ratio[word] = [freq / total_freq for freq in freqs]
    
    for word, freqs in evidence_word_freq_table.items():
        total_freq = sum(freqs)
        if total_freq < 50:
            continue
        evidence_word_ratio[word] = [freq / total_freq for freq in freqs]

    sorted_claim_sup = sorted(claim_word_ratio.items(), key=lambda x: x[1][0], reverse=True)
    sorted_claim_ref = sorted(claim_word_ratio.items(), key=lambda x: x[1][1], reverse=True)
    sorted_claim_nei = sorted(claim_word_ratio.items(), key=lambda x: x[1][2], reverse=True)
    sorted_evidence_sup = sorted(evidence_word_ratio.items(), key=lambda x: x[1][0], reverse=True)
    sorted_evidence_ref = sorted(evidence_word_ratio.items(), key=lambda x: x[1][1], reverse=True)
    sorted_evidence_nei = sorted(evidence_word_ratio.items(), key=lambda x: x[1][2], reverse=True)

    print("**************{}**************".format(data_type))
    print("claim_sup")
    for word, ratio in sorted_claim_sup[:10]:
        print(f"{word}: {ratio}")
    print("claim_ref")
    for word, ratio in sorted_claim_ref[:10]:
        print(f"{word}: {ratio}")
    print("claim_nei")
    for word, ratio in sorted_claim_nei[:10]:
        print(f"{word}: {ratio}")
    
    print("evidence_sup")
    for word, ratio in sorted_evidence_sup[:10]:
        print(f"{word}: {ratio}")
    print("evidence_ref")
    for word, ratio in sorted_evidence_ref[:10]:
        print(f"{word}: {ratio}")
    print("evidence_nei")
    for word, ratio in sorted_evidence_nei[:10]:
        print(f"{word}: {ratio}")


def main():
    data_path = "data/raw/[DATA].json"
    
    analyze_word(data_path, "train")
    analyze_word(data_path, "dev")
    analyze_word(data_path, "test")

if __name__ == "__main__":
    main()