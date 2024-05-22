import torch.nn as nn

from transformers import BertModel

class Claim_only_model(nn.Module):
    def __init__(self, args):
        super(Claim_only_model, self).__init__()
        self.encoder = BertModel.from_pretrained(args.cache_dir)
        self.classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
    
    def forward(self, ids, msks):
        hidden_states = self.encoder(ids, attention_mask=msks)[0]
        cls_hidden_states = hidden_states[:, 0, :]
        out = self.classifier(cls_hidden_states)
        return out

class Evidence_only_model(nn.Module):
    def __init__(self, args):
        super(Evidence_only_model, self).__init__()
        self.encoder = BertModel.from_pretrained(args.cache_dir)
        self.classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
    
    def forward(self, ids, msks):
        hidden_states = self.encoder(ids, attention_mask=msks)[0]
        cls_hidden_states = hidden_states[:, 0, :]
        out = self.classifier(cls_hidden_states)
        return out

class Claim_Evidence_model(nn.Module):
    def __init__(self, args):
        super(Claim_Evidence_model, self).__init__()
        self.encoder = BertModel.from_pretrained(args.cache_dir)
        self.classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)

    def forward(self, ids, msks):
        hidden_states = self.encoder(ids, attention_mask=msks)[0]
        cls_hidden_states = hidden_states[:, 0, :]
        out = self.classifier(cls_hidden_states)
        return out

    