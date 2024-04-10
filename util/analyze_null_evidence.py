import json
import argparse

def process_evidence(raws):
    evidences = ""
    for index in raws:
        evidence = raws[index]['text']
        if evidence:
            evidences += " "
    return evidences

def count_null_evidence(data_path, data_type):
    data_path = data_path.replace("[DATA]", data_type)
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    total_len = len(dataset)
    # count numbers of support/refute/nei examples
    s_num = 0
    r_num = 0
    n_num = 0
    # count numbers of examples with null evidence
    s_count = 0
    r_count = 0
    n_count = 0
    for data in dataset:
        total_len += 1
        evidence = process_evidence(data['gold evidence'])
        if data['label'] == 0:
            s_num += 1
            if not evidence:
                s_count += 1
        elif data['label'] == 1:
            r_num += 1
            if not evidence:
                r_count += 1
        elif data['label'] == 2:
            n_num += 1
            if evidence:
                n_count += 1

    print(data_type)
    print("total_len: ", total_len)
    print("s_num: {}, s_count: {}, s_rate: {:.3%}".format(s_num, s_count, (s_count / s_num)))
    print("r_num: {}, r_count: {}, r_rate: {:.3%}".format(r_num, r_count, (r_count / r_num)))
    print("n_num: {}, n_count: {}, n_rate: {:.3%}".format(n_num, n_count, (n_count / n_num)))

def count_gpt_null_evidence(data_path, data_type):
    data_path = data_path.replace("[DATA]", data_type)
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    total_len = len(dataset)
    # count numbers of support/refute/nei examples
    s_num = 0
    r_num = 0
    n_num = 0
    # count numbers of examples with null evidence
    s_count = 0
    r_count = 0
    for data in dataset:
        total_len += 1
        evidence = " ".join(data['gold_evidence'])
        if data['label'] == 0:
            s_num += 1
            if not evidence:
                s_count += 1
        elif data['label'] == 1:
            r_num += 1
            if not evidence:
                r_count += 1
        elif data['label'] == 2:
            n_num += 1

    print(data_type)
    print("total_len: ", total_len)
    print("s_num: {}, s_count: {}, s_rate: {:.3%}".format(s_num, s_count, (s_count / s_num)))
    print("r_num: {}, r_count: {}, r_rate: {:.3%}".format(r_num, r_count, (r_count / r_num)))
    print("n_num: {}".format(n_num))

def main(args):
    count_null_evidence(args.data_path, "train")
    count_null_evidence(args.data_path, "dev")
    count_null_evidence(args.data_path, "test")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/yangjun/fact/debias/data/raw/[DATA].json")
    parser.add_argument("--gpt_data_path", type=str, default="/data/yangjun/fact/debias/data/gpt/[DATA].json")

    args = parser.parse_args()
    main(args)
