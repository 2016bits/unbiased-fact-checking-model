import json
import jieba
import math
import csv
import numpy as np
from collections import defaultdict, Counter

MIN_FREQ = 5

def tokenize(sentence):
    return list(jieba.cut(sentence))

def process_evidence(raws):
    evidence_text = ""
    for index in raws:
        evidence = raws[index]['text']
        if evidence:
            evidence_text += evidence
    return evidence_text

def get_single_stopwords(data_path):
    # 从数据集中提取出 10 个最常见的停用词
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    global_word_counter = defaultdict(int)

    for data in dataset:
        claim = data['claim']
        words = tokenize(claim)

        for word in words:
            global_word_counter[word] += 1
    
    counter = Counter(global_word_counter)
    stop_words = counter.most_common(10)
    stop_words = [word[0] for word in stop_words]
    return stop_words

def get_counters(data_path, data_item):
    """
    函数的目的是:
        获取单个停用词(使用 get_single_stopwords 函数)
        统计数据集中所有词的出现频率(全局计数)
        统计每个标签(label)下所有词的出现频率
        统计每个标签(label)的出现次数
        返回上述三个统计结果以及总的词数量
    """
    stop_words = get_single_stopwords(data_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    global_word_counter = defaultdict(int)      # 统计全局词的出现频率
    global_label_counter = defaultdict(int)     # 统计每个标签(label)的出现次数
    label_word_counter = defaultdict(lambda: defaultdict(int))  # 统计每个标签(label)下所有词的出现频率
    words_num = 0                               # 统计数据集中所有词的数量

    for data in dataset:
        label = data['label']
        if data_item == 'claim':
            text = data['claim']
        elif data_item == 'gold evidence':
            text = process_evidence(data['gold evidence'])
        words = tokenize(text)
        words = [word for word in words if word not in stop_words]
    
        for word in words:
            global_word_counter[word] += 1
            global_label_counter[label] += 1
            label_word_counter[label][word] += 1
            words_num += 1
    
    print("Total words num: ", words_num)
    return global_word_counter, global_label_counter, label_word_counter, words_num

def analyze_lmi(data_path, data_type, data_item):
    data_path = data_path.replace("[DATA]", data_type)
    global_word_counter, global_label_counter, label_word_counter, words_num = get_counters(data_path, data_item)

    for label in label_word_counter.keys():
        if label == 2:
            continue
        words = []
        scores = []
        pmis = []
        freqs = []
        p_l = global_label_counter[label] / words_num

        word_counter = label_word_counter[label]
        for w in word_counter:
            if global_word_counter[w] < MIN_FREQ:
                continue

            # p(label | word)
            score = word_counter[w] / global_word_counter[w]
            pmi = math.log(score / p_l)     # 计算 PMI, 公式: log(p(label | word) / p(label))

            words.append(w)
            scores.append(score)
            pmis.append(pmi)
            freqs.append(word_counter[w])
        
        assert(len(words) == len(scores) == len(freqs) == len(pmis))

        pmis_x_freq = list(np.array(pmis) * freqs / words_num)
        pmis_x_freq, pmis, scores, freqs, words = (list(t) for t in zip(*sorted(zip(pmis_x_freq, pmis, scores, freqs, words), reverse=True)))


        # 将最重要的前 50 个词写入文件
        # filepath = './results/analyze/{}_{}_{}.csv'.format(data_type, data_item, label)
        filepath = './results/analyze/symmetric_{}_{}_{}.csv'.format(data_type, data_item, label)
        with open(filepath, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(min(50, len(words))):
                if not math.isnan(pmis_x_freq[i]): 
                    csv_writer.writerow([words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), freqs[i]])
                else:
                    csv_writer.writerow([words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), freqs[i]])

def main():
    # data_path = "data/raw/[DATA].json"
    # analyze_lmi(data_path, "train", "claim")
    # analyze_lmi(data_path, "train", "gold evidence")
    data_path = "data/gpt/symmetric_data/[DATA]_2.json"
    analyze_lmi(data_path, "dev", "claim")
    analyze_lmi(data_path, "dev", "gold evidence")
    analyze_lmi(data_path, "test", "claim")
    analyze_lmi(data_path, "test", "gold evidence")

if __name__ == '__main__':
    main()
