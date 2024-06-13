import random
import torch.nn as nn

from transformers import BertModel

class Base_model(nn.Module):
    def __init__(self, args):
        super(Base_model, self).__init__()
        self.encoder = BertModel.from_pretrained(args.cache_dir)
        self.classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        
    def forward(self, ids, msks):
        hidden_states = self.encoder(ids, attention_mask=msks)[0]
        cls_hidden_states = hidden_states[:, 0, :]
        out = self.classifier(cls_hidden_states)
        return out

class Dual_unbiased_model(nn.Module):
    def __init__(self, args):
        super(Dual_unbiased_model, self).__init__()
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

class Dual_unbiased_model(nn.Module):
    def __init__(self, args):
        super(Dual_unbiased_model, self).__init__()
        self.claim_encoder = BertModel.from_pretrained(args.cache_dir)
        self.claim_classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        self.evidence_encoder = BertModel.from_pretrained(args.cache_dir)
        self.evidence_classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        self.ce_encoder = BertModel.from_pretrained(args.cache_dir)
        self.ce_classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        
    def forward(self, claim_ids, claim_msks, evidence_ids, evidence_msks, ce_ids, ce_msks):
        claim_hidden_states = self.claim_encoder(claim_ids, attention_mask=claim_msks)[0]
        claim_cls_hidden_states = claim_hidden_states[:, 0, :]
        out_c = self.claim_classifier(claim_cls_hidden_states)

        evidence_hidden_states = self.evidence_encoder(evidence_ids, attention_mask=evidence_msks)[0]
        evidence_cls_hidden_states = evidence_hidden_states[:, 0, :]
        out_e = self.evidence_classifier(evidence_cls_hidden_states)

        ce_hidden_states = self.ce_encoder(ce_ids, attention_mask=ce_msks)[0]
        ce_cls_hidden_states = ce_hidden_states[:, 0, :]
        out_ce = self.ce_classifier(ce_cls_hidden_states)
        return out_c, out_e, out_ce
    
class SS_model(nn.Module):
    def __init__(self, args):
        super(SS_model, self).__init__()
        self.claim_encoder = BertModel.from_pretrained(args.cache_dir)
        self.claim_classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        self.ce_encoder = BertModel.from_pretrained(args.cache_dir)
        self.ce_classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        
    def forward(self, pos_ce_ids, pos_ce_msks, neg_ce_ids, neg_ce_msks, self_sup=True):
        ce_hidden_states = self.ce_encoder(pos_ce_ids, attention_mask=pos_ce_msks)[0]
        ce_cls_hidden_states = ce_hidden_states[:, 0, :]
        out_ce = self.ce_classifier(ce_cls_hidden_states)

        if self_sup:
            neg_hidden_states = self.ce_encoder(neg_ce_ids, attention_mask=neg_ce_msks)[0]
            neg_ce_cls_hidden_states = neg_hidden_states[:, 0, :]
            out_neg_ce = self.ce_classifier(neg_ce_cls_hidden_states)

            return out_ce, out_neg_ce
        else:
            return out_ce

class SS_unbiased_model(nn.Module):
    def __init__(self, args):
        super(SS_unbiased_model, self).__init__()
        self.claim_encoder = BertModel.from_pretrained(args.cache_dir)
        self.claim_classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        self.ce_encoder = BertModel.from_pretrained(args.cache_dir)
        self.ce_classifier = nn.Linear(args.bert_hidden_dim, args.num_classes)
        
    def forward(self, claim_ids, claim_msks, pos_ce_ids, pos_ce_msks, neg_ce_ids, neg_ce_msks, self_sup=True):
        claim_hidden_states = self.claim_encoder(claim_ids, attention_mask=claim_msks)[0]
        claim_cls_hidden_states = claim_hidden_states[:, 0, :]
        out_c = self.claim_classifier(claim_cls_hidden_states)

        ce_hidden_states = self.ce_encoder(pos_ce_ids, attention_mask=pos_ce_msks)[0]
        ce_cls_hidden_states = ce_hidden_states[:, 0, :]
        out_ce = self.ce_classifier(ce_cls_hidden_states)

        if self_sup:
            neg_hidden_states = self.ce_encoder(neg_ce_ids, attention_mask=neg_ce_msks)[0]
            neg_ce_cls_hidden_states = neg_hidden_states[:, 0, :]
            out_neg_ce = self.ce_classifier(neg_ce_cls_hidden_states)

            return out_c, out_ce, out_neg_ce
        else:
            return out_c, out_ce