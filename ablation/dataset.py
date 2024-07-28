import numpy as np
import json
import torch
from torch.autograd import Variable

class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, batch_size=64):
        self.device = args.device

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.threshold = args.threshold
        self.data_path = data_path
        
        examples = self.read_file(data_path)
        self.examples = examples
        self.total_num = len(examples)
        self.total_step = self.total_num / batch_size
        self.shuffle()
        self.step = 0

    def read_file(self, data_path):
        examples = list()
        with open(data_path, 'r', encoding='utf-8') as fin:
            dataset = json.load(fin)
        for data in dataset:
            # fin: [claim, positive_title, positive_evidence, negative_title, negative_evidence]
            sublines = line.strip().split("\t")
            examples.append([self.process_sent(sublines[0]), self.process_sent(sublines[2]), self.process_sent(sublines[4])])
        return examples

    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            examples = self.examples[self.step * self.batch_size : (self.step+1)*self.batch_size]
            pos_inputs = list()
            neg_inputs = list()
            for example in examples:
                # example: [claim, positive_evidence, negative_evidence]
                pos_inputs.append([example[0], example[1]])
                neg_inputs.append([example[0], example[2]])
            
            inp_pos, msk_pos, seg_pos = text2tensor(pos_inputs, self.tokenizer, self.max_len)
            inp_neg, msk_neg, seg_neg = text2tensor(neg_inputs, self.tokenizer, self.max_len)

            inp_tensor_pos = Variable(torch.LongTensor(inp_pos)).to(self.device)
            msk_tensor_pos = Variable(torch.LongTensor(msk_pos)).to(self.device)
            seg_tensor_pos = Variable(torch.LongTensor(seg_pos)).to(self.device)
            inp_tensor_neg = Variable(torch.LongTensor(inp_neg)).to(self.device)
            msk_tensor_neg = Variable(torch.LongTensor(msk_neg)).to(self.device)
            seg_tensor_neg = Variable(torch.LongTensor(seg_neg)).to(self.device)

            self.step += 1
            return inp_tensor_pos, msk_tensor_pos, seg_tensor_pos, inp_tensor_neg, msk_tensor_neg, seg_tensor_neg
        else:
            self.step = 0
            self.shuffle()
            raise StopIteration()