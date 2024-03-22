import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support

try:
    from util import log, dataset
except:
    from ..util import log, dataset

class Unbiased_model(nn.Module):
    def __init__(self, args, bert_model):
        super(Unbiased_model, self).__init__()
        self.encoder = bert_model
        self.classifier_ce_pair = nn.Linear(args.hidden_size, args.num_classes)
        self.classifier_claim = nn.Linear(args.hidden_size, args.num_classes)
    
    def forward(self, claim_ids, claim_msks, ce_ids, ce_msks):
        encoded_c = self.encoder(
            claim_ids, 
            token_type_ids=None, 
            attention_mask=claim_msks,
        )
        out_c = self.classifier_claim(encoded_c.last_hidden_state[:, 0, :])

        encoded_ce = self.encoder(
            ce_ids,
            token_type_ids=None,
            attention_mask=ce_msks,
        )
        out_ce = self.classifier_ce_pair(encoded_ce.last_hidden_state[:, 0, :])
        return out_c, out_ce

def train(args, model, train_loader, dev_loader, logger):
    # train
    logger.info("start training......")
    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    total_steps = len(train_loader) * args.epoch_num

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    best_micro_f1 = 0.0
    best_macro_f1 = 0.0

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epoch_num):
        model.train()
        for i, (claim_ids, claim_msks, ce_ids, ce_msks, labels) in tqdm(enumerate(train_loader),
                                                                        ncols=100, total=len(train_loader),
                                                                        desc="Epoch %d" % (epoch + 1)):
            claim_ids = Variable(claim_ids).cuda()
            claim_msks = Variable(claim_msks).cuda()
            ce_ids = Variable(ce_ids).cuda()
            ce_msks = Variable(ce_msks).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            out_c, out_ce = model(
                claim_ids, claim_msks, ce_ids, ce_msks
            )
            loss_c = loss_fn(out_c, labels)
            loss_ce = loss_fn(out_ce, labels)
            
            # for calculating KL-divergence
            pce = F.softmax(out_ce, dim=0)
            pc = F.softmax(out_c, dim=0)
            zeros_tensor = torch.zeros(labels.size(0), 3).cuda()
            label_distribution = zeros_tensor.scatter_(1, labels.unsqueeze(1), 1)
            loss_constraint = F.kl_div(pce.log(), label_distribution, reduction='sum') + F.kl_div(pc.log(), label_distribution, reduction='sum')
            loss = loss_ce.sum().item() + args.claim_loss_weight * loss_c.sum().item() - args.constraint_loss_weight * loss_constraint

            loss.backward()

            optimizer.step()
            scheduler.step()
        
        logger.info("start validating......")
        all_prediction = np.array([])
        all_target = np.array([])
        model.eval()
        for i, (claim_ids, claim_msks, ce_ids, ce_msks, labels) in tqdm(enumerate(dev_loader),
                                                                        ncols=100, total=len(dev_loader),
                                                                        desc="Epoch %d" % (epoch + 1)):
            claim_ids = Variable(claim_ids).cuda()
            claim_msks = Variable(claim_msks).cuda()
            ce_ids = Variable(ce_ids).cuda()
            ce_msks = Variable(ce_msks).cuda()
            labels = Variable(labels).cuda()

            with torch.no_grad():
                out_c, out_ce = model(
                    claim_ids, claim_msks, ce_ids, ce_msks
                )
                prob_ce = F.softmax(out_ce, dim=-1)
                prob_c = F.softmax(out_c, dim=-1)
                logits = prob_ce - prob_c
                
                prob = F.softmax(logits, dim=-1)
                prediction = torch.argmax(prob, dim=-1)
                label_ids = labels.to('cpu').numpy()

                labels_flat = label_ids.flatten()
                prediction_flat = prediction.to('cpu').numpy().flatten()
                all_prediction = np.concatenate((all_prediction, prediction_flat), axis=None)
                all_target = np.concatenate((all_target, labels_flat), axis=None)
            
        # Measure how long the validation run took.
        logger.info("Epoch {}".format(epoch + 1))
        pre, recall, micro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='micro')
        logger.info("       F1 (micro): {:.3%}".format(micro_f1))
        pre, recall, macro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='macro')
        logger.info("Precision (macro): {:.3%}".format(pre))
        logger.info("   Recall (macro): {:.3%}".format(recall))
        logger.info("       F1 (macro): {:.3%}".format(macro_f1))

        if micro_f1 > best_micro_f1 or macro_f1 > best_macro_f1:
            torch.save(model.state_dict(), args.saved_model_path)
            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
            else:
                best_macro_f1 = macro_f1

def test(model, logger, dev_loader):
    all_prediction = np.array([])
    all_target = np.array([])
    for i, (claim_ids, claim_msks, ce_ids, ce_msks, labels) in tqdm(enumerate(dev_loader),
                                                                        ncols=100, total=len(dev_loader)):
        claim_ids = Variable(claim_ids).cuda()
        claim_msks = Variable(claim_msks).cuda()
        ce_ids = Variable(ce_ids).cuda()
        ce_msks = Variable(ce_msks).cuda()
        labels = Variable(labels).cuda()

        with torch.no_grad():
            out_c, out_ce = model(
                claim_ids, claim_msks, ce_ids, ce_msks
            )
            logits = out_ce - out_c
            prob = F.softmax(logits, dim=-1)
            prediction = torch.argmax(prob, dim=-1)
            label_ids = labels.to('cpu').numpy()

            labels_flat = label_ids.flatten()
            prediction_flat = prediction.to('cpu').numpy().flatten()
            all_prediction = np.concatenate((all_prediction, prediction_flat), axis=None)
            all_target = np.concatenate((all_target, labels_flat), axis=None)
        
    # Measure how long the validation run took.
    pre, recall, micro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='micro')
    logger.info("       F1 (micro): {:.3%}".format(micro_f1))
    pre, recall, macro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='macro')
    logger.info("Precision (macro): {:.3%}".format(pre))
    logger.info("   Recall (macro): {:.3%}".format(recall))
    logger.info("       F1 (macro): {:.3%}".format(macro_f1))

def main(args):
    # init logger
    log_path = args.log_path + "debias.log"
    logger = log.get_logger(log_path)

    # load data
    logger.info("loading dataset......")
    train_data_path = args.data_path.replace("[DATA]", "train")
    train_raw = dataset.read_data(train_data_path, "gold_evidence")
    dev_data_path = args.data_path.replace("[DATA]", "dev")
    dev_raw = dataset.read_data(dev_data_path, "gold_evidence")

    # tokenizer
    logger.info("loading tokenizer......")
    tokenizer = BertTokenizer.from_pretrained(args.cache_dir)

    # batch data
    logger.info("batching data")
    train_batched = dataset.BatchedData(train_raw[:args.num_sample], args.max_len, tokenizer)
    dev_batched = dataset.BatchedData(dev_raw[:args.num_sample], args.max_len, tokenizer)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader = DataLoader(
        train_batched,
        sampler=RandomSampler(train_batched),
        batch_size=args.batch_size
    )
    dev_loader = DataLoader(
        dev_batched,
        sampler=SequentialSampler(dev_batched),
        batch_size=args.batch_size
    )

    # load model
    bert_model = BertModel.from_pretrained(args.cache_dir)
    bert_model = bert_model.cuda()
    model = Unbiased_model(args, bert_model)
    model = model.cuda()

    # for test
    if args.mode == 'test':
        checkpoint = "/data/yangjun/fact/debias/models/unbiased_model.ptdebias_gold_evidence.pth"
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        test(model, logger, dev_loader)
    elif args.mode == 'train':
        train(args, model, train_loader, dev_loader, logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./logs/')
    parser.add_argument("--data_path", type=str, default="./data/processed/[DATA].json")
    parser.add_argument("--saved_model_path", type=str, default="./models/unbiased_model.pt")

    parser.add_argument("--cache_dir", type=str, default="./bert-base-chinese")

    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--mode", type=str, default="train")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=768)

    # hyperparameters
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--claim_loss_weight", type=float, default=1.0)
    parser.add_argument("--constraint_loss_weight", type=float, default=0.01)

    args = parser.parse_args()
    main(args)
