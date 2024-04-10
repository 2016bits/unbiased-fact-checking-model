import torch.nn as nn

from transformers import BertModel

class Base_model(nn.Module):
    def __init__(self, args):
        super(Base_model, self).__init__()
        self.encoder = BertModel.from_pretrained(args.cache_dir)
        self.classifier = nn.Linear(args.bert_hidden_dim, 3)
        
    def forward(self, ids, msks):
        hidden_states = self.encoder(ids, attention_mask=msks)[0]
        cls_hidden_states = hidden_states[:, 0, :]
        out = self.classifier(cls_hidden_states)
        return out

class Unbiased_model(nn.Module):
    def __init__(self, args):
        super(Unbiased_model, self).__init__()
        self.claim_encoder = BertModel.from_pretrained(args.cache_dir)
        self.claim_classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        self.ce_encoder = BertModel.from_pretrained(args.cache_dir)
        self.ce_classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        
    def forward(self, claim_ids, claim_msks, ce_ids, ce_msks):
        claim_hidden_states = self.claim_encoder(claim_ids, attention_mask=claim_msks)[0]
        claim_cls_hidden_states = claim_hidden_states[:, 0, :]
        out_c = self.claim_classifier(claim_cls_hidden_states)

        ce_hidden_states = self.ce_encoder(ce_ids, attention_mask=ce_msks)[0]
        ce_cls_hidden_states = ce_hidden_states[:, 0, :]
        out_ce = self.ce_classifier(ce_cls_hidden_states)
        return out_c, out_ce