import os
import logging
import torch
from torch.utils.data import Dataset

from . import loader_utils
from . import bert_utils

logger = logging.getLogger()
from .bert_utils import bert_triple_converter


class triple_dataset(Dataset):
    def __init__(
        self, 
        args, 
        data_dir, 
        tokenizer, 
        mode, 
        extra_dataset = None
    ):
        """
        :param args: args parameters
        :param tokenizer: BertTokenizer or None
        :param data_dir: data folder path
        :param mode: train, target
        """
        # load triples -> qid, pos_docid, neg_docid
        if args.mode_name == "meta" and mode == "train":
            example_filename = "examples.jsonl"
        else:
            example_filename = "train.jsonl"
            
        examples = loader_utils.load_json2list(
            file_path=os.path.join(data_dir, example_filename)
        )
        logger.info('[%s dataset] success load triples = %d'%(mode, len(examples)))
        
        # load qid2query, docid2doc
        dataset = loader_utils.load_corpus(
            args, 
            data_dir=data_dir,
            tokenizer=tokenizer,
            mode=mode,
        )
        self.examples = examples
        self.dataset = dataset

        self.args = args
        self.mode = mode
        self.tokenizer = tokenizer
        
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return bert_triple_converter(
            index,
            ex=self.examples[index],
            dataset=self.dataset, 
            tokenizer=self.tokenizer,
        )
