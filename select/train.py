import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

from util import log, dataset

class inference_model(nn.Module):
    def __init__(self, bert_model, bert_hidden_dim, dropout, num_labels):
        self.bert_hidden_dim = bert_hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.encoder = bert_model
        self.classifier = nn.Linear(bert_hidden_dim, 1)
    
    def forward(self, inp_tensor, msk_tensor, seg_tensor):
        encoded_inputs = self.encoder(inp_tensor, msk_tensor, seg_tensor)
        dropouted_inputs = self.dropout(encoded_inputs)
        outputs = self.classifier(dropouted_inputs).squeeze(-1)
        score = torch.tanh(outputs)
        return score

def train_model(args, model, logger):
    best_acc = 0.0
    running_loss = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    global_step = 0

def main(args):
    # init logger
    log_path = args.log_path + "train_select_module.log"
    logger = log.get_logger(log_path)

    # load data
    logger.info("loading dataset......")
    train_data_path = args.data_path.replace("[DATA]", "train")
    train_loader = dataset.read_data(train_data_path, "train")
    dev_data_path = args.data_path.replace("[DATA]", "dev")
    dev_loader = dataset.read_data(dev_data_path, "dev")

    # tokenizer
    logger.info("loading tokenizer......")
    tokenizer = AutoTokenizer.from_pretrained(args.cache_dir)

    # batch data
    logger.info("batching data")
    train_dataset = dataset.BatchedData(train_loader, args.batch_size)
    dev_dataset = dataset.BatchedData(dev_loader, args.batch_size)

    # load model
    bert_model = AutoModelForMaskedLM.from_pretrained(args.cache_dir)
    bert_model = bert_model.cuda()
    model = inference_model(bert_model, args.bert_hidden_dim, args.dropout, args.num_labels)
    model = model.cuda()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--data_path", type=str, default="./data/processed/[DATA].json")
    parser.add_argument("--saved_model_path", type=str, default="./models/roberta.pt")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epoch_num", type=int, default=10)

    # roberta parameters
    parser.add_argument("--cache_dir", type=str, default="./hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument("--bert_hidden_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--num_labels", type=int, default=3)

    args = parser.parse_args()
    main(args)
