import os
import re
import torch
import json
import logging
from tqdm import tqdm
from ..transformers import AutoTokenizer
        
logger = logging.getLogger()

BERT_MAX_LEN = 510 # shrink max_len can enlarge batch_size with the limite GPU memory.

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class BertTokenizer:
    def __init__(self, args):
        self.args = args
        self.do_lower_case = args.do_lower_case
        self.subtokenizer = AutoTokenizer.from_pretrained(
            self.args.cache_pretrain_dir, 
            do_lower_case=self.do_lower_case
        )
        logger.info("success loaded Bert-Tokenizer !")
        
        self.cls_token = self.subtokenizer.cls_token
        self.sep_token = self.subtokenizer.sep_token
        self.unk_token = self.subtokenizer.unk_token
        
    def convert_tokens_to_ids(self, tokens):
        return self.subtokenizer.convert_tokens_to_ids(tokens)
        
    def tokenize(self, text, max_len):
        # lower case
        text = text.lower() if self.do_lower_case else text
        # drop char
        regex_drop_char = re.compile('[^a-zA-Z0-9\s]+')
        drop_char_text = regex_drop_char.sub(' ', text)
        # del multi spaces
        regex_multi_space = re.compile('\s+')
        tokens = regex_multi_space.sub(' ', drop_char_text).strip().split()
        
        # convert token to subtoken
        all_sub_tokens = []
        for num, token in enumerate(tokens):
            sub_tokens = self.subtokenizer.tokenize(token)
            if len(sub_tokens) < 1:
                sub_tokens = [self.unk_tokezn]
            for sub_token in sub_tokens:
                all_sub_tokens.append(sub_token)
            if len(all_sub_tokens) >= max_len:
                break
        return all_sub_tokens


## ----------------------------------------------------------------------
## ----------------------------------------------------------------------
## pairwise training

def bert_triple_converter(
    index, 
    ex, 
    dataset, 
    tokenizer, 
    max_len=BERT_MAX_LEN
):
    """
    :param index: training examples list index
    :param ex: one example
    :param dataset: qid2query, docid2doc
    :param tokenizer: Bert Tokenzier
    :param max_len: max input tensor length
    :return index, pos_intput_tensor, 
            neg_input_tensor, pos_segment_tensor, neg_segment_tensor
    """
    # cls sep toks
    cls_tok = tokenizer.cls_token
    sep_tok = tokenizer.sep_token
    
    # query toks
    query_toks = dataset["qid2query"][ex["qid"]]
    
    # pos_doc, neg_doc toks (limit len)
    max_doc_len = (max_len - len(query_toks) - 3)
    pos_doc_toks = dataset["docid2doc"][ex["pos_docid"]][:max_doc_len]
    neg_doc_toks = dataset["docid2doc"][ex["neg_docid"]][:max_doc_len]
    
    # pos & neg input tok
    pos_input_toks = [cls_tok] + query_toks + \
                     [sep_tok] + pos_doc_toks + [sep_tok]
    
    neg_input_toks = [cls_tok] + query_toks + \
                     [sep_tok] + neg_doc_toks + [sep_tok]
    
    # pos & neg input ids
    pos_intput_ids = tokenizer.convert_tokens_to_ids(pos_input_toks)
    neg_intput_ids = tokenizer.convert_tokens_to_ids(neg_input_toks)
    
    # pos & neg seg ids
    pos_segment_ids = [0] * (len(query_toks) + 2) + [1] * (len(pos_doc_toks) + 1)
    neg_segment_ids = [0] * (len(query_toks) + 2) + [1] * (len(neg_doc_toks) + 1)
    
    return {
        "index":index, 
        "pos_intput_tensor":torch.LongTensor(pos_intput_ids), 
        "neg_input_tensor":torch.LongTensor(neg_intput_ids), 
        "pos_segment_tensor":torch.LongTensor(pos_segment_ids), 
        "neg_segment_tensor":torch.LongTensor(neg_segment_ids)
    }

def bert_triple_batchify_for_train(batch):
    
    indexs = [ex["index"] for ex in batch]
    pos_inputs = [ex["pos_intput_tensor"] for ex in batch]
    neg_inputs = [ex["neg_input_tensor"] for ex in batch]
    pos_segments = [ex["pos_segment_tensor"] for ex in batch]
    neg_segments = [ex["neg_segment_tensor"] for ex in batch]
    
    # pack batch tensor
    pos_batch_tensor = pack_batch_tensor(
        inputs=pos_inputs, 
        segments=pos_segments
    )
    neg_batch_tensor = pack_batch_tensor(
        inputs=neg_inputs, 
        segments=neg_segments
    )
    return {
        "indexs":indexs,
        "pos_input_ids":pos_batch_tensor["input_ids"], 
        "pos_input_mask":pos_batch_tensor["input_mask"], 
        "pos_segment_ids":pos_batch_tensor["segment_ids"], 
        "neg_input_ids":neg_batch_tensor["input_ids"], 
        "neg_input_mask":neg_batch_tensor["input_mask"], 
        "neg_segment_ids":neg_batch_tensor["segment_ids"]
    }



## ----------------------------------------------------------------------
## ----------------------------------------------------------------------
## pointwise training
def bert_pair_converter(
    index, 
    ex, 
    dataset, 
    tokenizer, 
    max_len=BERT_MAX_LEN
):
    """
    :param index: training examples list index
    :param ex: one example
    :param dataset: qid2query, docid2doc
    :param tokenizer: Bert Tokenzier
    :param max_len: max input tensor length
    :return index, intput_tensor, segment_tensor
    """
    # cls sep toks
    cls_tok = tokenizer.cls_token
    sep_tok = tokenizer.sep_token
    
    label = int(ex["label"])
    
    # query toks
    query_toks = dataset["qid2query"][ex["qid"]]
    
    # doc toks (limit len)
    max_doc_len = (max_len - len(query_toks) - 3)
    doc_toks = dataset["docid2doc"][ex["docid"]][:max_doc_len]
    
    # pos input tok
    input_toks = [cls_tok] + query_toks + \
                    [sep_tok] + doc_toks + [sep_tok]
    
    # pos input ids
    intput_ids = tokenizer.convert_tokens_to_ids(input_toks)
    
    # pos seg ids
    segment_ids = [0] * (len(query_toks) + 2) + [1] * (len(doc_toks) + 1)
    
    return {
        "index":index, 
        "intput_tensor":torch.LongTensor(intput_ids), 
        "segment_tensor":torch.LongTensor(segment_ids), 
        "label":label,
    }

def bert_pair_batchify_for_train(batch):
    
    indexs = [ex["index"] for ex in batch]
    inputs = [ex["intput_tensor"] for ex in batch]
    segments = [ex["segment_tensor"] for ex in batch]
    labels = [ex["label"] for ex in batch]
    
    # pack batch tensor
    batch_tensor = pack_batch_tensor(
        inputs=inputs, 
        segments=segments
    )

    return {
        "indexs":indexs,
        "input_ids":batch_tensor["input_ids"], 
        "input_mask":batch_tensor["input_mask"], 
        "segment_ids":batch_tensor["segment_ids"], 
        "labels":torch.LongTensor(labels),
    }

## ----------------------------------------------------------------------
## ----------------------------------------------------------------------
## Evaluation
def bert_eval_converter(
    index, 
    ex, 
    dataset, 
    tokenizer, 
    max_len=BERT_MAX_LEN
):
    """
    :param index: training examples list index
    :param ex: one example
    :param dataset: qid2query, docid2doc
    :param tokenizer: Bert Tokenzier
    :param max_len: max input tensor length
    :return index, intput_tensor, segment_tensor
    """
    # cls sep toks
    cls_tok = tokenizer.cls_token
    sep_tok = tokenizer.sep_token
    
    # query toks
    query_toks = dataset["qid2query"][ex["qid"]]
    
    # doc toks (limit len)
    max_doc_len = (max_len - len(query_toks) - 3)
    doc_toks = dataset["docid2doc"][ex["docid"]][:max_doc_len]
    
    # input tok
    input_toks = [cls_tok] + query_toks + \
                 [sep_tok] + doc_toks + [sep_tok]

    # input ids
    intput_ids = tokenizer.convert_tokens_to_ids(input_toks)
    
    # seg ids
    segment_ids = [0] * (len(query_toks) + 2) + [1] * (len(doc_toks) + 1)
    
    return {
        "index":index, 
        "intput_tensor":torch.LongTensor(intput_ids),
        "segment_tensor":torch.LongTensor(segment_ids),
        "qd_score":ex["score"]   
    }


def bert_batchify_for_eval(batch):
    
    indexs = [ex["index"] for ex in batch]
    inputs = [ex["intput_tensor"] for ex in batch]
    segments = [ex["segment_tensor"] for ex in batch]
    qd_scores = [ex["qd_score"] for ex in batch]
    
    # pack batch tensor
    batch_tensor = pack_batch_tensor(
        inputs=inputs, 
        segments=segments
    )

    return {
        "indexs":indexs,
        "input_ids":batch_tensor["input_ids"], 
        "input_mask":batch_tensor["input_mask"], 
        "segment_ids":batch_tensor["segment_ids"],
        "qd_scores":qd_scores,
    }


## ----------------------------------------------------------------------
## ----------------------------------------------------------------------
def pack_batch_tensor(inputs, segments):
    """default pad_ids = 0
    """
    input_max_length = max([d.size(0) for d in inputs])
    # prepare batch tensor
    input_ids = torch.LongTensor(len(inputs), input_max_length).zero_()
    input_mask = torch.LongTensor(len(inputs), input_max_length).zero_()
    segment_ids = torch.LongTensor(len(inputs), input_max_length).zero_()
    for i, d in enumerate(inputs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)
    for i, s in enumerate(segments):
        segment_ids[i, :s.size(0)].copy_(s)
    return {
        "input_ids":input_ids, 
        "input_mask":input_mask, 
        "segment_ids":segment_ids
    }

