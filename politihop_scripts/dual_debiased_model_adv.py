import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from util import log, dataset
from util.model import Unbiased_model

def scale_probability(original_probabilities, scaled_rate=1.2):
    # 较大的概率值增大 scaled_rate 倍，较小的概率值缩小 scaled_rate 倍
        # 找到每行最大概率值和对应的索引
    max_probabilities, max_indices = torch.max(original_probabilities, dim=1)
    min_probabilities, min_indices = torch.min(original_probabilities, dim=1)

    # 将较大的概率值增大 scaled_rate 倍，较小的概率值缩小 scaled_rate 倍
    scaled_max_probabilities = max_probabilities * scaled_rate
    scaled_min_probabilities = min_probabilities / scaled_rate

    # 更新概率值
    scaled_probabilities = original_probabilities.clone()  # 克隆原始概率，以防修改原始张量
    scaled_probabilities[torch.arange(original_probabilities.size(0)), max_indices] = scaled_max_probabilities
    scaled_probabilities[torch.arange(original_probabilities.size(0)), min_indices] = scaled_min_probabilities

    # 确保每个样本概率之和为 1
    scaled_probabilities /= torch.sum(scaled_probabilities, dim=1, keepdim=True)
    
    return scaled_probabilities

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
        for i, (claim_ids, claim_msks, ce_ids, ce_msks, labels) in tqdm(enumerate(train_loader),
                                            ncols=100, total=len(train_loader),
                                            desc="Epoch %d" % (epoch + 1)):
            claim_ids = Variable(claim_ids).cuda()
            claim_msks = Variable(claim_msks).cuda()
            ce_ids = Variable(ce_ids).cuda()
            ce_msks = Variable(ce_msks).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            out_c, out_ce = model(claim_ids, claim_msks, ce_ids, ce_msks)
            loss_c = F.cross_entropy(out_c, labels.long())
            loss_ce = F.cross_entropy(out_ce, labels.long())
            
            # for calculating KL-divergence
            pce = F.softmax(out_ce, dim=-1)
            pc = F.softmax(out_c, dim=-1)
            zeros_tensor = torch.zeros(labels.size(0), 2).cuda()
            label_distribution = zeros_tensor.scatter_(1, labels.unsqueeze(1), 1)
            loss_constraint = F.kl_div(pce.log(), label_distribution, reduction='sum') + F.kl_div(pc.log(), label_distribution, reduction='sum')

            loss = loss_ce.sum() + args.claim_loss_weight * loss_c.sum() - args.constraint_loss_weight * loss_constraint

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
                out_c, out_ce = model(claim_ids, claim_msks, ce_ids, ce_msks)
                prob_c = F.softmax(out_c, dim=-1)
                prob_ce = F.softmax(out_ce, dim=-1)

                # scale probability
                scaled_prob_c = scale_probability(prob_c, args.scaled_rate)

                scores = prob_ce - scaled_prob_c * args.claim_loss_weight

                pred_prob, pred_label = torch.max(scores, dim=-1)
                
                label_ids = labels.to('cpu').numpy()

                labels_flat = label_ids.flatten()
                all_prediction = np.concatenate((all_prediction, np.array(pred_label.to('cpu'))), axis=None)
                all_target = np.concatenate((all_target, labels_flat), axis=None)
            
        # Measure how long the validation run took.
        logger.info("Epoch {}".format(epoch + 1))
        logger.info("constraint_loss_weight: {}, claim_loss_weight: {}, scaled_rate: {}".format(
            args.constraint_loss_weight, args.claim_loss_weight, args.scaled_rate))
        acc = accuracy_score(all_target, all_prediction)
        logger.info("         Accuracy: {:.3%}".format(acc))
        pre, recall, micro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='micro')
        logger.info("       F1 (micro): {:.3%}".format(micro_f1))
        pre, recall, macro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='macro')
        logger.info("Precision (macro): {:.3%}".format(pre))
        logger.info("   Recall (macro): {:.3%}".format(recall))
        logger.info("       F1 (macro): {:.3%}".format(macro_f1))

        if macro_f1 > best_macro_f1:
            model_path = args.saved_model_path.replace("[DATASET]", args.dataset)
            model_path = model_path.replace("[constraint]", str(args.constraint_loss_weight))
            model_path = model_path.replace("[claim]", str(args.claim_loss_weight))
            model_path = model_path.replace("[scaled]", str(args.scaled_rate))
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), model_path)            

def test(model, logger, test_loader):
    logger.info("start testing......")
    all_prediction = np.array([])
    all_target = np.array([])
    for i, (claim_ids, claim_msks, ce_ids, ce_msks, labels) in tqdm(enumerate(test_loader),
                                        ncols=100, total=len(test_loader)):
        claim_ids = Variable(claim_ids).cuda()
        claim_msks = Variable(claim_msks).cuda()
        ce_ids = Variable(ce_ids).cuda()
        ce_msks = Variable(ce_msks).cuda()
        labels = Variable(labels).cuda()

        with torch.no_grad():
            out_c, out_ce = model(claim_ids, claim_msks, ce_ids, ce_msks)
            prob_c = F.softmax(out_c, dim=-1)
            prob_ce = F.softmax(out_ce, dim=-1)
            
            # scale probability
            scaled_prob_c = scale_probability(prob_c)

            scores = prob_ce - scaled_prob_c * args.claim_loss_weight

            pred_prob, pred_label = torch.max(scores, dim=-1)
            
            label_ids = labels.to('cpu').numpy()

            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, np.array(pred_label.to('cpu'))), axis=None)
            all_target = np.concatenate((all_target, labels_flat), axis=None)
        
    # Measure how long the validation run took.
    report = classification_report(all_target, all_prediction, output_dict=True)
    f1_score_class_0 = report['0.0']['f1-score']
    f1_score_class_1 = report['1.0']['f1-score']
    logger.info("F1 score for class 0: {:.3%}".format(f1_score_class_0))
    logger.info("F1 score for class 1: {:.3%}".format(f1_score_class_1))
    acc = accuracy_score(all_target, all_prediction)
    logger.info("         Accuracy: {:.3%}".format(acc))
    pre, recall, micro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='micro')
    logger.info("       F1 (micro): {:.3%}".format(micro_f1))
    pre, recall, macro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='macro')
    logger.info("Precision (macro): {:.3%}".format(pre))
    logger.info("   Recall (macro): {:.3%}".format(recall))
    logger.info("       F1 (macro): {:.3%}".format(macro_f1))

    return micro_f1, pre, recall, macro_f1

def main(args):
    # init logger
    log_path = args.log_path.replace("[DATASET]", args.dataset)
    if args.mode == "train":
        # train in FEVER, test in symmetric-FEVER
        log_path = log_path + "{}_unbiased_constraint_{}_claim_{}_scaled_{}.log".format(args.num_classes, args.constraint_loss_weight, args.claim_loss_weight, args.scaled_rate)
    elif args.mode == "test":
        # for train in FEVER, test in symmetric-FEVER
        log_path = log_path + "test_{}_unbiased.log".format(args.num_classes)
    logger = log.get_logger(log_path)

    # load data
    logger.info("loading dataset......")
    
    train_data_path = args.train_data_path.replace("[DATASET]", args.dataset)
    dev_data_path = args.dev_data_path.replace("[DATASET]", args.dataset)
    test_data_path = args.test_data_path.replace("[DATASET]", args.dataset)
    train_raw = dataset.read_fever_data(train_data_path, "gold_evidence")
    dev_raw = dataset.read_fever_data(dev_data_path, "gold_evidence")
    test_raw = dataset.read_fever_data(test_data_path, "gold_evidence")

    # tokenizer
    logger.info("loading tokenizer......")
    tokenizer = BertTokenizer.from_pretrained(args.cache_dir)

    # batch data
    logger.info("batching data")
    train_batched = dataset.batch_c_ce_data(train_raw[:args.num_sample], args.max_len, tokenizer)
    dev_batched = dataset.batch_c_ce_data(dev_raw[:args.num_sample], args.max_len, tokenizer)
    test_batched = dataset.batch_c_ce_data(test_raw[:args.num_sample], args.max_len, tokenizer)

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
    model = Unbiased_model(args)
    model = model.cuda()

    # for test
    if args.mode == 'test':
        checkpoint = args.checkpoint
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
    elif args.mode == 'train':
        train(args, model, train_loader, dev_loader, logger)
    micro_f1, pre, recall, macro_f1 = test(model, logger, test_loader)

    # with open(args.test_results, 'a+') as f:
    #     print("constraint_loss_weight: {}, claim_loss_weight: {}".format(args.constraint_loss_weight, args.claim_loss_weight), file=f)
    #     print("       F1 (micro): {:.3%}".format(micro_f1), file=f)
    #     print("Precision (macro): {:.3%}".format(pre), file=f)
    #     print("   Recall (macro): {:.3%}".format(recall), file=f)
    #     print("       F1 (macro): {:.3%}".format(macro_f1), file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("最完整的双向去偏模型，包括动态约束损失和扩大偏见影响")
    parser.add_argument("--log_path", type=str, default='./save_logs/[DATASET]/')
    
    parser.add_argument("--dataset", type=str, default="adversarial-PolitiHop")
    # for FEVER
    parser.add_argument("--train_data_path", type=str, default="./data/[DATASET]/converted_data/train_2.json")
    parser.add_argument("--dev_data_path", type=str, default="./data/[DATASET]/converted_data/dev_2.json")
    parser.add_argument("--test_data_path", type=str, default="./data/[DATASET]/converted_data/test_2.json")
    parser.add_argument("--saved_model_path", type=str, default="./save_models/[DATASET]/two_unbiased_[constraint]_[claim]_[scaled].pth")

    parser.add_argument("--checkpoint", type=str, default="./save_models/fever/two_unbiased_FEVER_0.007_0.2_1.5.pth")
    # parser.add_argument("--cache_dir", type=str, default="./bert-base-chinese")
    parser.add_argument("--cache_dir", type=str, default="./bert-base-uncased")

    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--mode", type=str, default="train")
    # parser.add_argument("--mode", type=str, default="test")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--bert_hidden_dim", type=int, default=768)
    parser.add_argument('--initial_lr', type=float, default=5e-6, help='initial learning rate')
    parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')

    # hyperparameters
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--claim_loss_weight", type=float, default=0.2)
    # best for improved_CHEF
    # parser.add_argument("--constraint_loss_weight", type=float, default=0.004)
    # best for CHEF
    parser.add_argument("--constraint_loss_weight", type=float, default=0.007)
    parser.add_argument("--scaled_rate", type=float, default=1.2)

    args = parser.parse_args()
    main(args)
