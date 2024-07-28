import json
import random
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

        label = data['label']
        if not isinstance(label, int):
            label = label_dict[label]
        data_list.append({
            "id": data['id'],
            "claim": claim,
            "evidence": evidence,
            "ce_pair": ce_pair,
            "label": label
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

def read_c_e_ce_data(data_path, evidence_type):
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

        label = data['label']
        if not isinstance(label, int):
            label = label_dict[label]
        data_list.append({
            "id": data['id'],
            "claim": claim,
            "evidence": evidence,
            "ce_pair": ce_pair,
            "label": label
        })

    return data_list

def batch_c_e_ce_data(data_loader, max_len, tokenizer):
    # batch claim, evidence, claim-evidence data, for debiased model
    claim_ids = []
    claim_msks = []
    evidence_ids = []
    evidence_msks = []
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

        encoded_evidence_dict = tokenizer.encode_plus(
            data['evidence'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        evidence_ids.append(encoded_evidence_dict['input_ids'])
        evidence_msks.append(encoded_evidence_dict['attention_mask'])

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
    evidence_ids = torch.cat(evidence_ids, dim=0).cuda()
    evidence_msks = torch.cat(evidence_msks, dim=0).cuda()
    ce_pair_ids = torch.cat(ce_pair_ids, dim=0).cuda()
    ce_pair_msks = torch.cat(ce_pair_msks, dim=0).cuda()
    labels = torch.tensor(labels, device='cuda')

    batched_dataset = TensorDataset(claim_ids, claim_msks, evidence_ids, evidence_msks, ce_pair_ids, ce_pair_msks, labels)
    return batched_dataset

def batch_c_data(data_loader, max_len, tokenizer):
    # batch claim-evidence data, for base model
    ids = []
    msks = []
    labels = []

    for data in data_loader:
        encoded_claim_dict = tokenizer.encode_plus(
            data["claim"],
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

def batch_e_data(data_loader, max_len, tokenizer):
    # batch claim-evidence data, for base model
    ids = []
    msks = []
    labels = []

    for data in data_loader:
        encoded_claim_dict = tokenizer.encode_plus(
            data["evidence"],
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

# prepare data for self-supervised model
def read_ss_data(data_path, evidence_type):
    # read data and generate negeative claim-evidence pair by sampling randomly evidence
    with open(data_path, 'r', encoding='utf-8') as fin:
        dataset = json.load(fin)

    data_list = []

    for data in dataset:
        claim = data['claim']
        if evidence_type == "gold_evidence":
            evidence = " [SEP] ".join(data['gold_evidence'])
        elif evidence_type == "selected_evidence":
            evidence = " [SEP] ".join(data['selected_evidence'])
        
        pos_ce = "[CLS] {} [SEP] {}".format(claim, evidence)

        # randomly chose evidence
        random_data = random.choice(dataset)
        while True:
            if evidence_type == "gold_evidence":
                random_evidence = " [SEP] ".join(random_data['gold_evidence'])
            elif evidence_type == "selected_evidence":
                random_evidence = " [SEP] ".join(random_data['selected_evidence'])
            
            if random_evidence != evidence:
                # ensure that negative evidence is different from positive evidence
                break
            else:
                random_data = random.choice(dataset)
        neg_ce = "[CLS] {} [SEP] {}".format(claim, random_evidence)

        label = data['label']
        if not isinstance(label, int):
            label = label_dict[label]
        data_list.append({
            "id": data['id'],
            "claim": claim,
            "pos_ce": pos_ce,
            "neg_ce": neg_ce,
            "label": label
        })

    return data_list

# batch data for self-supervised model
def batch_ss_ce_data(data_loader, max_len, tokenizer):
    # batch claim, positive claim-evidence pair and negative claim-evidence pair
    pos_ce_ids = []
    pos_ce_msks = []
    neg_ce_ids = []
    neg_ce_msks = []
    labels = []

    # pre-processing sentenses to BERT pattern
    for data in data_loader:
        encoded_pos_ce_dict = tokenizer.encode_plus(
            data['pos_ce'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        pos_ce_ids.append(encoded_pos_ce_dict['input_ids'])
        pos_ce_msks.append(encoded_pos_ce_dict['attention_mask'])

        encoded_neg_ce_dict = tokenizer.encode_plus(
            data['neg_ce'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        neg_ce_ids.append(encoded_neg_ce_dict['input_ids'])
        neg_ce_msks.append(encoded_neg_ce_dict['attention_mask'])

        labels.append(data['label'])
    
    # convert the lists into tensors
    pos_ce_ids = torch.cat(pos_ce_ids, dim=0).cuda()
    pos_ce_msks = torch.cat(pos_ce_msks, dim=0).cuda()
    neg_ce_ids = torch.cat(neg_ce_ids, dim=0).cuda()
    neg_ce_msks = torch.cat(neg_ce_msks, dim=0).cuda()
    labels = torch.tensor(labels, device='cuda')

    batched_dataset = TensorDataset(pos_ce_ids, pos_ce_msks, neg_ce_ids, neg_ce_msks, labels)
    return batched_dataset

# batch data for self-supervised debiased model
def batch_ss_c_ce_data(data_loader, max_len, tokenizer):
    # batch claim, positive claim-evidence pair and negative claim-evidence pair
    claim_ids = []
    claim_msks = []
    pos_ce_ids = []
    pos_ce_msks = []
    neg_ce_ids = []
    neg_ce_msks = []
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

        encoded_pos_ce_dict = tokenizer.encode_plus(
            data['pos_ce'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        pos_ce_ids.append(encoded_pos_ce_dict['input_ids'])
        pos_ce_msks.append(encoded_pos_ce_dict['attention_mask'])

        encoded_neg_ce_dict = tokenizer.encode_plus(
            data['neg_ce'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        neg_ce_ids.append(encoded_neg_ce_dict['input_ids'])
        neg_ce_msks.append(encoded_neg_ce_dict['attention_mask'])

        labels.append(data['label'])
    
    # convert the lists into tensors
    claim_ids = torch.cat(claim_ids, dim=0).cuda()
    claim_msks = torch.cat(claim_msks, dim=0).cuda()
    pos_ce_ids = torch.cat(pos_ce_ids, dim=0).cuda()
    pos_ce_msks = torch.cat(pos_ce_msks, dim=0).cuda()
    neg_ce_ids = torch.cat(neg_ce_ids, dim=0).cuda()
    neg_ce_msks = torch.cat(neg_ce_msks, dim=0).cuda()
    labels = torch.tensor(labels, device='cuda')

    batched_dataset = TensorDataset(claim_ids, claim_msks, pos_ce_ids, pos_ce_msks, neg_ce_ids, neg_ce_msks, labels)
    return batched_dataset

def read_c_e_ce_data(data_path, evidence_type):
    with open(data_path, 'r', encoding='utf-8') as fin:
        dataset = json.load(fin)

    data_list = []
    for data in dataset:
        claim = data['claim']
        if evidence_type == "gold_evidence":
            evidence = " [SEP] ".join(data['gold_evidence'])
        elif evidence_type == "selected_evidence":
            evidence = " [SEP] ".join(data['selected_evidence'])
        ce_pair = "[CLS] {} [SEP] {}".format(claim, evidence)

        label = data['label']
        if not isinstance(label, int):
            label = label_dict[label]

        data_list.append({
            "id": data['id'],
            "claim": claim,
            "evidence": evidence,
            "ce_pair": ce_pair,
            "label": label
        })
    return data_list

def batch_c_e_ce_data(data_list, max_len, tokenizer):
    # pre-processing sentenses to BERT pattern
    claim_ids = []
    claim_msks = []
    evidence_ids = []
    evidence_msks = []
    ce_pair_ids = []
    ce_pair_msks = []
    labels = []
    for data in data_list:
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

        encoded_evidence_dict = tokenizer.encode_plus(
            data['evidence'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        evidence_ids.append(encoded_evidence_dict['input_ids'])
        evidence_msks.append(encoded_evidence_dict['attention_mask'])
        
        encoded_ce_pair_dict = tokenizer.encode_plus(
            data['ce_pair'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        ce_pair_ids.append(encoded_ce_pair_dict['input_ids'])
        ce_pair_msks.append(encoded_ce_pair_dict['attention_mask'])

        labels.append(data['label'])
    
    # convert the lists into tensors
    claim_ids = torch.cat(claim_ids, dim=0).cuda()
    claim_msks = torch.cat(claim_msks, dim=0).cuda()
    evidence_ids = torch.cat(evidence_ids, dim=0).cuda()
    evidence_msks = torch.cat(evidence_msks, dim=0).cuda()
    ce_pair_ids = torch.cat(ce_pair_ids, dim=0).cuda()
    ce_pair_msks = torch.cat(ce_pair_msks, dim=0).cuda()
    labels = torch.tensor(labels, device='cuda')

    batched_dataset = TensorDataset(claim_ids, claim_msks, evidence_ids, evidence_msks, ce_pair_ids, ce_pair_msks, labels)
    return batched_dataset
