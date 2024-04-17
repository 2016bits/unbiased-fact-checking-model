import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from util import log, dataset
from util.model import Base_model

def test(model, logger, test_loader):
    logger.info("start testing......")
    all_prediction = np.array([])
    all_target = np.array([])
    for i, (ids, msks, labels) in tqdm(enumerate(test_loader),
                                        ncols=100, total=len(test_loader)):
        ids = Variable(ids).cuda()
        msks = Variable(msks).cuda()
        labels = Variable(labels).cuda()

        with torch.no_grad():
            out = model(ids, msks)
            scores = F.softmax(out, dim=-1)
            pred_prob, pred_label = torch.max(scores, dim=-1)
            
            label_ids = labels.to('cpu').numpy()

            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, np.array(pred_label.to('cpu'))), axis=None)
            all_target = np.concatenate((all_target, labels_flat), axis=None)
        
    # Measure how long the validation run took.
    acc = accuracy_score(all_target, all_prediction)
    logger.info("         Accuracy: {:.3%}".format(acc))
    pre, recall, micro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='micro')
    logger.info("       F1 (micro): {:.3%}".format(micro_f1))
    pre, recall, macro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='macro')
    logger.info("Precision (macro): {:.3%}".format(pre))
    logger.info("   Recall (macro): {:.3%}".format(recall))
    logger.info("       F1 (macro): {:.3%}".format(macro_f1))

def main(args):
    # init logger
    log_path = args.log_path + "test_two_class_base_symmetric_data.log"
    logger = log.get_logger(log_path)

    # load data
    logger.info("loading dataset......")
    test_raw = dataset.read_data(args.data_path, "gold_evidence")

    # tokenizer
    logger.info("loading tokenizer......")
    tokenizer = BertTokenizer.from_pretrained(args.cache_dir)

    # batch data
    logger.info("batching data")
    test_batched = dataset.batch_ce_data(test_raw[:args.num_sample], args.max_len, tokenizer)

    test_loader = DataLoader(
        test_batched,
        sampler=SequentialSampler(test_batched),
        batch_size=args.batch_size
    )

    # load model
    model = Base_model(args)
    model = model.cuda()

    # for test
    checkpoint = args.checkpoint
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    test(model, logger, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./logs/')
    parser.add_argument("--data_path", type=str, default="./data/gpt/symmetric_test_2_all.json")

    parser.add_argument("--cache_dir", type=str, default="./bert-base-chinese")
    parser.add_argument("--checkpoint", type=str, default="./models/two_class_base_model.pth")

    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=2)

    # train parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--bert_hidden_dim", type=int, default=768)
    parser.add_argument('--initial_lr', type=float, default=5e-6, help='initial learning rate')
    parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')

    # hyperparameters
    parser.add_argument("--seed", type=int, default=1111)

    args = parser.parse_args()
    main(args)
