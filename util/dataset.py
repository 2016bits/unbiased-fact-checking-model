import json
import torch
from torch.utils.data import TensorDataset

label_dict = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}

def read_data(data_path, evidence_type):
    with open(data_path, 'r', encoding='utf-8') as fin:
        dataset = json.load(fin)

    data_list = []

    for data in dataset:
        claim = data['claim']
        if evidence_type == "gold_evidence":
            evidence = " [SEP] ".join(data['gold_evidence'])
        elif evidence_type == "selected_evidence":
            evidence = " [SEP] ".join(data['selected_evidence'])
        elif evidence_type == "only_claim":
            evidence = ""
        
        ce_pair = "[CLS] {} [SEP] {}".format(claim, evidence)

        data_list.append({
            "id": data['id'],
            "claim": claim,
            "ce_pair": ce_pair,
            "label": label_dict[data['label']]
        })

    return data_list

def batch_c_ce_data(data_loader, max_len, tokenizer):
    # batch claim, claim-evidence data, for debiased model
    claim_ids = []
    claim_msks = []
    ce_pair_ids = []
    ce_pair_msks = []
    labels = []

    # pre-processing sentenses to BERT pattern
    for data in data_loader:
        encoded_claim_dict = tokenizer.encode_plus(
            data["claim"],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        claim_ids.append(encoded_claim_dict['input_ids'])
        claim_msks.append(encoded_claim_dict['attention_mask'])

        encoded_ce_dict = tokenizer.encode_plus(
            data['ce_pair'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        ce_pair_ids.append(encoded_ce_dict['input_ids'])
        ce_pair_msks.append(encoded_ce_dict['attention_mask'])

        labels.append(data['label'])
    
    # convert the lists into tensors
    claim_ids = torch.cat(claim_ids, dim=0).cuda()
    claim_msks = torch.cat(claim_msks, dim=0).cuda()
    ce_pair_ids = torch.cat(ce_pair_ids, dim=0).cuda()
    ce_pair_msks = torch.cat(ce_pair_msks, dim=0).cuda()
    labels = torch.tensor(labels, device='cuda')

    batched_dataset = TensorDataset(claim_ids, claim_msks, ce_pair_ids, ce_pair_msks, labels)
    return batched_dataset

def batch_ce_data(data_loader, max_len, tokenizer):
    # batch claim-evidence data, for base model
    ids = []
    msks = []
    labels = []

    for data in data_loader:
        encoded_claim_dict = tokenizer.encode_plus(
            data["ce_pair"],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        ids.append(encoded_claim_dict['input_ids'])
        msks.append(encoded_claim_dict['attention_mask'])

        labels.append(data['label'])
    
    ids = torch.cat(ids, dim=0).cuda()
    msks = torch.cat(msks, dim=0).cuda()
    labels = torch.tensor(labels, device='cuda')

    batched_dataset = TensorDataset(ids, msks, labels)
    return batched_dataset
