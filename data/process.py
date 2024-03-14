import argparse
import json

label_dict = {
    0: 'SUPPORTS',
    1: 'REFUTES',
    2: 'NOT ENOUGH INFO'
}

def process_evidence(raws):
    evidence_list = []
    for index in raws:
        evidence = raws[index]['text']
        if evidence:
            evidence_list.append(evidence)
    return evidence_list

def main(args):
    in_path = args.in_path.replace("[DATA]", args.data_type)
    out_path = args.out_path.replace("[DATA]", args.data_type)

    with open(in_path, 'r', encoding='utf-8') as f:
        raws = json.load(f)

    processed = []
    for data in raws:
        processed.append({
            'id': data['claimId'],
            'claim': data['claim'],
            'label': label_dict[data['label']],
            'evidence': process_evidence(data['evidence']),
            'gold_evidence': process_evidence(data['gold evidence'])
        })
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default='/data/yangjun/fact/debias/data/raw/[DATA].json')
    parser.add_argument("--out_path", type=str, default='/data/yangjun/fact/debias/data/processed/[DATA].json')
    parser.add_argument("--data_type", type=str, default='train')

    args = parser.parse_args()
    main(args)
