import os
import logging
import torch
from torch.utils.data import Dataset
from . import loader_utils
from .bert_utils import bert_eval_converter

logger = logging.getLogger()

class eval_dataset(Dataset):
    def __init__(
        self, 
        args, 
        data_dir, 
        tokenizer, 
        mode,
        extra_dataset=None
    ):
        """
        :param args: args parameters
        :param tokenizer: BertTokenizer or None
        :param data_dir: data folder path
        :param mode: train, dev/eval
        """
        # load pairs -> qid, docid
        self.examples = loader_utils.load_json2list(
            file_path=os.path.join(data_dir, "%s.jsonl"%mode)
        )
        logger.info('[%s] success load data = %d'%(mode, len(self.examples)))
        
        # load qid2query, docid2doc
        if extra_dataset is not None:
            self.dataset = extra_dataset.dataset
        else:
            dataset = loader_utils.load_corpus(
                args, 
                data_dir=data_dir,
                tokenizer=tokenizer,
                mode=mode,
            )
            self.dataset = dataset
        
        self.args = args
        self.mode = mode
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return bert_eval_converter(
            index,
            ex=self.examples[index],
            dataset=self.dataset, 
            tokenizer=self.tokenizer,
        )