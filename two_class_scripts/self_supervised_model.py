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
from util.model import SS_model

def compute_self_loss(logits_neg, label):
    index = label.view(label.size(0), 1)
    neg_top_k = F.softmax(logits_neg, dim=-1).gather(1, index).sum(1)
    qice_loss = neg_top_k.mean()
    return qice_loss

def train(args, model, train_loader, dev_loader, logger):
    # train
    logger.info("start training......")
    optimizer = AdamW(
        model.parameters(),
        lr=args.initial_lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=args.initial_eps  # args.adam_epsilon  - default is 1e-8.
    )
    total_steps = len(train_loader) * args.epoch_num

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    best_macro_f1 = 0.0

    for epoch in range(args.epoch_num):
        model.train()
        for i, (pos_ce_ids, pos_ce_msks, neg_ce_ids, neg_ce_msks, labels) in tqdm(enumerate(train_loader),
                                            ncols=100, total=len(train_loader),
                                            desc="Epoch %d" % (epoch + 1)):
            pos_ce_ids = Variable(pos_ce_ids).cuda()
            pos_ce_msks = Variable(pos_ce_msks).cuda()
            neg_ce_ids = Variable(neg_ce_ids).cuda()
            neg_ce_msks = Variable(neg_ce_msks).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            # train with two-stages
            # if epoch < args.pretrain_epoches:
            #     out_pos_ce = model(pos_ce_ids, pos_ce_msks, None, None, False)
            #     loss_pos_ce = F.cross_entropy(out_pos_ce, labels.long())

            #     loss = loss_pos_ce.sum()
            # else:
            #     # add self-supervised module
            #     out_pos_ce, out_neg_ce = model(pos_ce_ids, pos_ce_msks, neg_ce_ids, neg_ce_msks, True)
            #     loss_pos_ce = F.cross_entropy(out_pos_ce, labels.long())

            #     self_loss = compute_self_loss(out_neg_ce, labels)

            #     loss = loss_pos_ce.sum() + args.self_loss_weight * self_loss

            # train with one-stage
            out_pos_ce, out_neg_ce = model(pos_ce_ids, pos_ce_msks, neg_ce_ids, neg_ce_msks, True)
            loss_pos_ce = F.cross_entropy(out_pos_ce, labels.long())

            self_loss = compute_self_loss(out_neg_ce, labels)

            loss = loss_pos_ce.sum() + args.self_loss_weight * self_loss

            loss.backward()

            optimizer.step()
            scheduler.step()
        
        logger.info("start validating......")
        all_prediction = np.array([])
        all_target = np.array([])
        model.eval()
        for i, (pos_ce_ids, pos_ce_msks, neg_ce_ids, neg_ce_msks, labels) in tqdm(enumerate(dev_loader),
                                            ncols=100, total=len(dev_loader),
                                            desc="Epoch %d" % (epoch + 1)):
            pos_ce_ids = Variable(pos_ce_ids).cuda()
            pos_ce_msks = Variable(pos_ce_msks).cuda()
            neg_ce_ids = Variable(neg_ce_ids).cuda()
            neg_ce_msks = Variable(neg_ce_msks).cuda()
            labels = Variable(labels).cuda()

            with torch.no_grad():
                out_pos_ce = model(pos_ce_ids, pos_ce_msks, None, None, False)
                prob_ce = F.softmax(out_pos_ce, dim=-1)
                pred_prob, pred_label = torch.max(prob_ce, dim=-1)
                
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
            model_path = args.saved_model_path
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), model_path)            

def test(model, logger, test_loader):
    logger.info("start testing......")
    all_prediction = np.array([])
    all_target = np.array([])
    for i, (pos_ce_ids, pos_ce_msks, neg_ce_ids, neg_ce_msks, labels) in tqdm(enumerate(test_loader),
                                            ncols=100, total=len(test_loader)):
        pos_ce_ids = Variable(pos_ce_ids).cuda()
        pos_ce_msks = Variable(pos_ce_msks).cuda()
        neg_ce_ids = Variable(neg_ce_ids).cuda()
        neg_ce_msks = Variable(neg_ce_msks).cuda()
        labels = Variable(labels).cuda()

        with torch.no_grad():
            out_pos_ce = model(pos_ce_ids, pos_ce_msks, None, None, False)
            prob_ce = F.softmax(out_pos_ce, dim=-1)
            pred_prob, pred_label = torch.max(prob_ce, dim=-1)
            
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
        log_path = args.log_path + "ss_{}_class_CHEF2.log".format(args.num_classes)
    elif args.mode == "test":
        log_path = args.log_path + "test_ss_two_class_CHEF.log"
    logger = log.get_logger(log_path)

    # load data
    logger.info("loading dataset......")
    train_data_path = args.data_path.replace("[DATA]", "train")
    train_raw = dataset.read_ss_data(train_data_path, "gold_evidence")
    dev_data_path = args.data_path.replace("[DATA]", "dev")
    dev_raw = dataset.read_ss_data(dev_data_path, "gold_evidence")
    test_data_path = args.data_path.replace("[DATA]", "test")
    test_raw = dataset.read_ss_data(test_data_path, "gold_evidence")

    # tokenizer
    logger.info("loading tokenizer......")
    tokenizer = BertTokenizer.from_pretrained(args.cache_dir)

    # batch data
    logger.info("batching data")
    train_batched = dataset.batch_ss_ce_data(train_raw[:args.num_sample], args.max_len, tokenizer)
    dev_batched = dataset.batch_ss_ce_data(dev_raw[:args.num_sample], args.max_len, tokenizer)
    test_batched = dataset.batch_ss_ce_data(test_raw[:args.num_sample], args.max_len, tokenizer)

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
    model = SS_model(args)
    model = model.cuda()

    # for test
    if args.mode == 'test':
        checkpoint = args.checkpoint
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
    elif args.mode == 'train':
        train(args, model, train_loader, dev_loader, logger)
    acc, micro_f1, pre, recall, macro_f1 = test(model, logger, test_loader)

    # with open(args.test_results, 'a+') as f:
    #     print("constraint_loss_weight: {}, claim_loss_weight: {}".format(args.constraint_loss_weight, args.claim_loss_weight), file=f)
    #     print("         Accuracy: {:.3%}".format(acc))
    #     print("       F1 (micro): {:.3%}".format(micro_f1), file=f)
    #     print("Precision (macro): {:.3%}".format(pre), file=f)
    #     print("   Recall (macro): {:.3%}".format(recall), file=f)
    #     print("       F1 (macro): {:.3%}".format(macro_f1), file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./logs/')
    parser.add_argument("--data_path", type=str, default="./data/processed/[DATA]_2.json")
    parser.add_argument("--saved_model_path", type=str, default="./models/two_ss_CHEF2.pth")
    # parser.add_argument("--test_results", type=str, default="./logs/test_result_unbiased_2_class.txt")

    parser.add_argument("--cache_dir", type=str, default="./bert-base-chinese")
    parser.add_argument("--checkpoint", type=str, default="./models/two_ss_CHEF.pth")

    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--mode", type=str, default="train")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--pretrain_epoches", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--bert_hidden_dim", type=int, default=768)
    parser.add_argument('--initial_lr', type=float, default=5e-6, help='initial learning rate')
    parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')

    # hyperparameters
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--self_loss_weight", type=float, default=3.0)
    
    args = parser.parse_args()
    main(args)
