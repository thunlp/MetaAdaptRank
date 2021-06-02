import os
import time
import json
import torch
import random
import argparse
import logging
import traceback
import numpy as np
from typing import List, Dict

import pytrec_eval

logger = logging.getLogger()


# ------------------------------------------------------------
# ------------------------------------------------------------
class TotalDataLoader(object):
    def __init__(
        self,
        args,
        train_loader,
        target_loader,
        dev_loader,
        dev_dataset,
        batchify_train_inputs,
        batchify_eval_inputs,  
    ):
        self.args = args
        self.train_loader = train_loader
        self.target_loader = target_loader
        self.dev_loader = dev_loader
        self.dev_dataset = dev_dataset
        self.batchify_train_inputs = batchify_train_inputs
        self.batchify_eval_inputs = batchify_eval_inputs
        
        # init train iterator
        self.reset_train_iter()
        if args.mode_name == "meta":
            self.reset_target_iter()
        
    def reset_train_iter(self):
        self.train_iterator = iter(self.train_loader)

    def reset_target_iter(self):
        self.target_iterator = iter(self.target_loader)
            
    def generate_train_inputs(self):
        try:
            train_batch = next(self.train_iterator)
        except StopIteration:
            self.reset_train_iter()
            train_batch = next(self.train_iterator)
        train_inputs, train_indexs = self.batchify_train_inputs(train_batch, device=self.args.device)
        return train_inputs, train_indexs
            
    def generate_target_inputs(self):
        try:
            target_batch = next(self.target_iterator)
        except StopIteration:
            self.reset_target_iter()
            target_batch = next(self.target_iterator)
        target_inputs, target_indexs = self.batchify_train_inputs(target_batch, device=self.args.device)
        return target_inputs, target_indexs
    
# ------------------------------------------------------------
# ------------------------------------------------------------
def select_input_refactor(args):
    if args.loss_class == "pairwise":
        return bert_triple_refactor_for_train, bert_refactor_for_eval
    elif args.loss_class == "pointwise":
        return bert_pair_refactor_for_train, bert_refactor_for_eval
    else:
        raise ValueError("invalid loss_class %s" %loss_class)

## ------------------------------------------------------------
## ------------------------------------------------------------
## Bert Triple Refactor
def bert_triple_refactor_for_train(batch, device):
    inputs = {
        "pos_input_ids":batch["pos_input_ids"].to(device), 
        "pos_input_mask":batch["pos_input_mask"].to(device), 
        "pos_segment_ids":batch["pos_segment_ids"].to(device), 
        "neg_input_ids":batch["neg_input_ids"].to(device), 
        "neg_input_mask":batch["neg_input_mask"].to(device), 
        "neg_segment_ids":batch["neg_segment_ids"].to(device), 
    }
    return inputs, batch["indexs"]


def bert_pair_refactor_for_train(batch, device):
    inputs = {
        "pos_input_ids":batch["input_ids"].to(device), 
        "pos_input_mask":batch["input_mask"].to(device), 
        "pos_segment_ids":batch["segment_ids"].to(device), 
        "labels":batch["labels"].to(device), 
    }
    return inputs, batch["indexs"]


def bert_refactor_for_eval(batch, device):
    inputs = {
        "pos_input_ids":batch["input_ids"].to(device), 
        "pos_input_mask":batch["input_mask"].to(device), 
        "pos_segment_ids":batch["segment_ids"].to(device),
    }
    return inputs, batch["indexs"], batch["qd_scores"]

# ------------------------------------------------------------
# ------------------------------------------------------------
def get_metrics(rst_dict, qrels_path, save_filename, trec_mark):
    save_trec_result(rst_dict, save_filename, trec_mark)
    metrics = compute_trec_metrics(trec_path=save_filename, qrels_path=qrels_path)
    return metrics


def save_trec_result(rst_dict, save_filename, trec_mark):
    # save best
    with open(save_filename, "w", encoding="utf-8") as fo:
        for qid, item in rst_dict.items():
            res = sorted(item, key=lambda x: x[1], reverse=True)
            rank = 1
            docid_set = set()
            for docid, d_score in res:
                if docid in docid_set:
                    continue
                fo.write(" ".join([qid, "Q0", str(docid), str(rank), str(d_score), trec_mark]) + "\n")
                docid_set.add(docid)
                rank += 1
        fo.close()
        

def compute_trec_metrics(trec_path, qrels_path):
    with open(trec_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    
    with open(qrels_path, 'r') as f_qrel:
        qrels = pytrec_eval.parse_qrel(f_qrel)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)
    for query_id, query_measures in sorted(results.items()):
        pass
    mes = {}
    for measure in sorted(query_measures.keys()):
        mes[measure] = pytrec_eval.compute_aggregated_measure(
            measure, [query_measures[measure] for query_measures in results.values()]
        )
    return mes



def save_features(data_list, save_filename):
    if os.path.exists(save_filename):
        os.remove(save_filename)
    with open(file=save_filename, mode="w", encoding="utf-8") as fw:
        for data in data_list:
            fw.write("{}\n".format(json.dumps(data)))
        fw.close()

# ------------------------------------------------------------
# ------------------------------------------------------------
def set_seed(args):
    """
    Set random seed.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
        
        
def override_args(old_args, new_args):
    ''' cover old args to new args,
    log which args has been changed.
    '''
    old_args, new_args = vars(old_args), vars(new_args)
    for k in new_args.keys():
        if k in old_args and not (type(old_args[k]) is np.ndarray):
            if old_args[k] != new_args[k]:
                logger.info('Overriding saved %s: %s -> %s' 
                            %(k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
        else:
            old_args[k] = new_args[k]
    return argparse.Namespace(**old_args) 


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
    
