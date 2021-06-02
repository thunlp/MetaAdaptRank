import os
import logging
import torch
import random
from torch.utils.data import Dataset

from . import loader_utils
from .t5_utils import t5_train_pair_converter, t5_train_single_converter
logger = logging.getLogger()


train_generate_feature_converter = {
    "contrastqg":t5_train_pair_converter,
    "qg":t5_train_single_converter
}


class train_generate_dataset(Dataset):
    def __init__(
        self, 
        args,
        train_file,
        tokenizer, 
    ):
        """
        :param intput_dir: examples.jsonl ("pos_docid"/"neg_docid"); docid2doc.jsonl
        :param tokenizer: T5Tokenizer or None
        """
        examples = loader_utils.load_csv(train_file)
        self.args = args
        self.tokenizer = tokenizer
        self.examples = examples
        
                
    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index):
        return train_generate_feature_converter[self.args.generator_mode](
            index,
            ex=self.examples[index],
            tokenizer=self.tokenizer,
        )