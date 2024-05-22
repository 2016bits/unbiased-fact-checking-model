import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from util import log, dataset
from util.debias_model import Claim_only_model, Evidence_only_model, Claim_Evidence_model

def train(args, claim_only_model, evidence_only_model, model, train_loader, dev_loader, logger):
    # train
    logger.info("start training......")
    claim_only_optimizer = AdamW(
        claim_only_model.parameters(),
        lr=args.initial_lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=args.initial_eps  # args.adam_epsilon  - default is 1e-8.
    )
    total_steps = len(train_loader) * args.epoch_num
    claim_only_scheduler = get_linear_schedule_with_warmup(
        claim_only_optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    evidence_only_optimizer = AdamW(
        evidence_only_model.parameters(),
        lr=args.initial_lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=args.initial_eps  # args.adam_epsilon  - default is 1e-8.
    )
    evidence_only_scheduler = get_linear_schedule_with_warmup(
        evidence_only_optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.initial_lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=args.initial_eps  # args.adam_epsilon  - default is 1e-8.
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    best_macro_f1 = 0.0

    for epoch in range(args.epoch_num):
        # optimize claim_only_model and evidence_only_model
        claim_only_model.train()
        evidence_only_model.train()
        model.eval()
        for _, (claim_ids, claim_msks, evidence_ids, evidence_msks, _, _, labels) in tqdm(enumerate(train_loader),
                                            ncols=100, total=len(train_loader),
                                            desc="Epoch %d" % (epoch + 1)):
            claim_ids = Variable(claim_ids).cuda()
            claim_msks = Variable(claim_msks).cuda()
            evidence_ids = Variable(evidence_ids).cuda()
            evidence_msks = Variable(evidence_msks).cuda()
            labels = Variable(labels).cuda()

            claim_only_optimizer.zero_grad()

            out_c = model(claim_ids, claim_msks)
            out_e = model(evidence_ids, evidence_msks)
            loss_c = F.cross_entropy(out_c, labels.long())
            loss_e = F.cross_entropy(out_e, labels.long())
            
            sepearte_loss = loss_c.sum() + loss_e.sum()

            sepearte_loss.backward()

            claim_only_optimizer.step()
            claim_only_scheduler.step()
            evidence_only_optimizer.step()
            evidence_only_scheduler.step()
        
        # freeze claim_only_model and evidence_only_model, only optimize model
        claim_only_model.eval()
        evidence_only_model.eval()
        model.train()
        for _, (claim_ids, claim_msks, evidence_ids, evidence_msks, ce_ids, ce_msks, labels) in tqdm(enumerate(train_loader),
                                            ncols=100, total=len(train_loader),
                                            desc="Epoch %d" % (epoch + 1)):
            claim_ids = Variable(claim_ids).cuda()
            claim_msks = Variable(claim_msks).cuda()
            evidence_ids = Variable(evidence_ids).cuda()
            evidence_msks = Variable(evidence_msks).cuda()
            ce_ids = Variable(ce_ids).cuda()
            ce_msks = Variable(ce_msks).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            out_c = claim_only_model(claim_ids, claim_msks)
            out_e = evidence_only_model(evidence_ids, evidence_msks)
            out_ce = model(ce_ids, ce_msks)
            loss_c = F.cross_entropy(out_c, labels.long())
            loss_e = F.cross_entropy(out_e, labels.long())
            loss_ce = F.cross_entropy(out_ce, labels.long())

            # calculate constraint loss
            pce = F.softmax(out_ce, dim=-1)
            pc = F.softmax(out_c, dim=-1)
            pe = F.softmax(out_e, dim=-1)
            zeros_tensor = torch.zeros(labels.size(0), args.num_classes).cuda()
            label_distribution = zeros_tensor.scatter_(1, labels.unsqueeze(1), 1)
            loss_constraint = F.kl_div(pce.log(), label_distribution, reduction='sum') + F.kl_div(pc.log(), label_distribution, reduction='sum') + F.kl_div(pe.log(), label_distribution, reduction='sum')

            loss_overall = loss_ce.sum() - args.claim_loss_weight * loss_c.sum() - args.evidence_loss_weight * loss_e.sum() - args.constraint_loss_weight * loss_constraint

            loss_overall.backward()
            optimizer.step()
            scheduler.step()
            
        # for validation
        logger.info("start validating......")
        all_prediction = np.array([])
        all_target = np.array([])
        claim_only_model.eval()
        evidence_only_model.eval()
        model.eval()
        for i, (claim_ids, claim_msks, evidence_ids, evidence_msks, ce_ids, ce_msks, labels) in tqdm(enumerate(dev_loader),
                                            ncols=100, total=len(dev_loader),
                                            desc="Epoch %d" % (epoch + 1)):
            claim_ids = Variable(claim_ids).cuda()
            claim_msks = Variable(claim_msks).cuda()
            evidence_ids = Variable(evidence_ids).cuda()
            evidence_msks = Variable(evidence_msks).cuda()
            ce_ids = Variable(ce_ids).cuda()
            ce_msks = Variable(ce_msks).cuda()
            labels = Variable(labels).cuda()

            with torch.no_grad():
                out_c = claim_only_model(claim_ids, claim_msks)
                out_e = evidence_only_model(evidence_ids, evidence_msks)
                out_ce = model(ce_ids, ce_msks)
                prob_c = F.softmax(out_c, dim=-1)
                prob_e = F.softmax(out_e, dim=-1)
                prob_ce = F.softmax(out_ce, dim=-1)
                scores = prob_ce - prob_c * args.claim_loss_weight - prob_e * args.evidence_loss_weight

                pred_prob, pred_label = torch.max(scores, dim=-1)
                
                label_ids = labels.to('cpu').numpy()

                labels_flat = label_ids.flatten()
                all_prediction = np.concatenate((all_prediction, np.array(pred_label.to('cpu'))), axis=None)
                all_target = np.concatenate((all_target, labels_flat), axis=None)
            
        # Measure how long the validation run took.
        logger.info("Epoch {}".format(epoch + 1))
        acc = accuracy_score(all_target, all_prediction)
        logger.info("         Accuracy: {:.3%}".format(acc))
        pre, recall, micro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='micro')
        logger.info("       F1 (micro): {:.3%}".format(micro_f1))
        pre, recall, macro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='macro')
        logger.info("Precision (macro): {:.3%}".format(pre))
        logger.info("   Recall (macro): {:.3%}".format(recall))
        logger.info("       F1 (macro): {:.3%}".format(macro_f1))

        if macro_f1 > best_macro_f1:
            model_path = args.saved_model_path.replace("[claim]", str(args.claim_loss_weight))
            model_path = model_path.replace("[evidence]", str(args.evidence_loss_weight))
            model_path = model_path.replace("[constraint]", str(args.constraint_loss_weight))
            torch.save(model.state_dict(), model_path)
            claim_only_model_path = model_path.replace("[model]", "claim")
            evidence_only_model_path = model_path.replace("[model]", "evidence")
            best_macro_f1 = macro_f1
            torch.save(claim_only_model.state_dict(), claim_only_model_path)
            torch.save(evidence_only_model.state_dict(), evidence_only_model_path)

def test(claim_only_model, evidence_only_model, model, logger, test_loader):
    claim_only_model.eval()
    evidence_only_model.eval()
    model.eval()

    logger.info("start testing......")
    all_prediction = np.array([])
    all_target = np.array([])
    for i, (claim_ids, claim_msks, evidence_ids, evidence_msks, ce_ids, ce_msks, labels) in tqdm(enumerate(test_loader),
                                        ncols=100, total=len(test_loader)):
        claim_ids = Variable(claim_ids).cuda()
        claim_msks = Variable(claim_msks).cuda()
        evidence_ids = Variable(evidence_ids).cuda()
        evidence_msks = Variable(evidence_msks).cuda()
        ce_ids = Variable(ce_ids).cuda()
        ce_msks = Variable(ce_msks).cuda()
        labels = Variable(labels).cuda()

        with torch.no_grad():
            out_c = claim_only_model(claim_ids, claim_msks)
            out_e = evidence_only_model(evidence_ids, evidence_msks)
            out_ce = model(ce_ids, ce_msks)
            prob_c = F.softmax(out_c, dim=-1)
            prob_e = F.softmax(out_e, dim=-1)
            prob_ce = F.softmax(out_ce, dim=-1)
            scores = prob_ce - prob_c * args.claim_loss_weight - prob_e * args.evidence_loss_weight

            pred_prob, pred_label = torch.max(scores, dim=-1)
            
            label_ids = labels.to('cpu').numpy()

            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, np.array(pred_label.to('cpu'))), axis=None)
            all_target = np.concatenate((all_target, labels_flat), axis=None)
        
    acc = accuracy_score(all_target, all_prediction)
    logger.info("         Accuracy: {:.3%}".format(acc))
    pre, recall, micro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='micro')
    logger.info("       F1 (micro): {:.3%}".format(micro_f1))
    pre, recall, macro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='macro')
    logger.info("Precision (macro): {:.3%}".format(pre))
    logger.info("   Recall (macro): {:.3%}".format(recall))
    logger.info("       F1 (macro): {:.3%}".format(macro_f1))

    return acc, micro_f1, pre, recall, macro_f1

def main(args):
    # init logger
    if args.mode == "train":
        log_path = args.log_path + "{}_class_CHEF_adverserial_unbiased_constraint_{}_claim_{}_evidence_{}.log".format(args.num_classes, args.constraint_loss_weight, args.claim_loss_weight, args.evidence_loss_weight)
    elif args.mode == "test":
        log_path = args.log_path + "test_two_class_CHEF_adverserial_unbiased_model.log"
    logger = log.get_logger(log_path)

    # load data
    logger.info("loading dataset......")
    train_data_path = args.data_path.replace("[DATA]", "train")
    train_raw = dataset.read_c_e_ce_data(train_data_path, "gold_evidence")
    dev_data_path = args.data_path.replace("[DATA]", "dev")
    dev_raw = dataset.read_c_e_ce_data(dev_data_path, "gold_evidence")
    test_data_path = args.data_path.replace("[DATA]", "test")
    test_raw = dataset.read_c_e_ce_data(test_data_path, "gold_evidence")

    # tokenizer
    logger.info("loading tokenizer......")
    tokenizer = BertTokenizer.from_pretrained(args.cache_dir)

    # batch data
    logger.info("batching data")
    train_batched = dataset.batch_c_e_ce_data(train_raw[:args.num_sample], args.max_len, tokenizer)
    dev_batched = dataset.batch_c_e_ce_data(dev_raw[:args.num_sample], args.max_len, tokenizer)
    test_batched = dataset.batch_c_e_ce_data(test_raw[:args.num_sample], args.max_len, tokenizer)

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
    test_loader = DataLoader(
        test_batched,
        sampler=SequentialSampler(test_batched),
        batch_size=args.batch_size
    )

    # load model
    claim_only_model = Claim_only_model(args)
    evidence_only_model = Evidence_only_model(args)
    model = Claim_Evidence_model(args)
    claim_only_model = claim_only_model.cuda()
    evidence_only_model = evidence_only_model.cuda()
    model = model.cuda()

    # for test
    if args.mode == 'test':
        claim_only_checkpoint = args.checkpoint.replace("[MODEL]", "claim_only")
        claim_only_state_dict = torch.load(claim_only_checkpoint)
        claim_only_model.load_state_dict(claim_only_state_dict)
        evidence_only_checkpoint = args.checkpoint.replace("[MODEL]", "evidence_only")
        evidence_only_state_dict = torch.load(evidence_only_checkpoint)
        evidence_only_model.load_state_dict(evidence_only_state_dict)
        checkpoint = args.checkpoint.replace("[MODEL]", "model")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
    elif args.mode == 'train':
        train(args, claim_only_model, evidence_only_model, model, train_loader, dev_loader, logger)
    acc, micro_f1, pre, recall, macro_f1 = test(claim_only_model, evidence_only_model, model, logger, test_loader)

    # with open(args.test_results, 'a+') as f:
    #     print("claim_loss_weight: {}, evidence_loss_weight: {}".format(args.claim_loss_weight, args.evidence_loss_weight), file=f)
    #     print("         Accuracy: {:.3%}".format(acc), file=f)
    #     print("       F1 (micro): {:.3%}".format(micro_f1), file=f)
    #     print("Precision (macro): {:.3%}".format(pre), file=f)
    #     print("   Recall (macro): {:.3%}".format(recall), file=f)
    #     print("       F1 (macro): {:.3%}".format(macro_f1), file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./logs/c_e_ce_logs/')
    parser.add_argument("--data_path", type=str, default="./data/processed/[DATA]_2.json")
    parser.add_argument("--saved_model_path", type=str, default="./models/c_e_ce_models/two_CHEF_adverserial_unbiased_[MODEL]_[constraint]_[claim]_[evidence].pth")
    parser.add_argument("--test_results", type=str, default="./para_results/two_class_CHEF_adverserial_unbiased.txt")

    parser.add_argument("--cache_dir", type=str, default="./bert-base-chinese")
    parser.add_argument("--checkpoint", type=str, default="./models/two_CHEF_adverserial_unbiased_[MODEL]_0.004_0.5.pth")

    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--mode", type=str, default="train")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--bert_hidden_dim", type=int, default=768)
    parser.add_argument('--initial_lr', type=float, default=5e-6, help='initial learning rate')
    parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')

    # hyperparameters
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--constraint_loss_weight", type=float, default=0.008)
    parser.add_argument("--claim_loss_weight", type=float, default=0.1)
    parser.add_argument("--evidence_loss_weight", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
