import os
import sys
import time
import tqdm
import json
import torch
import logging
import argparse
import traceback
import numpy as np
from tqdm import tqdm

sys.path.append("..")
import config
import utils
from utils import TotalDataLoader
from model import MetaRanker
from metaranker import dataloaders

torch.backends.cudnn.benchmark=True

logger = logging.getLogger()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# eval
def evaluate(args, model, eval_loader, eval_dataset, batchify_eval_inputs, mode):
    # empty cuda cache
    torch.cuda.empty_cache()
    
    rst_dict = {}
    combine_features = []
    for step, eval_batch in enumerate(tqdm(eval_loader)):
        eval_inputs, indexs, retrival_scores = batchify_eval_inputs(eval_batch, device=args.device)
        d_scores, d_features = model.predict(eval_inputs)
        
        for index, d_score, d_feature, r_score in zip(indexs, d_scores, d_features, retrival_scores):
            example = eval_dataset.examples[index]
            qid = example["qid"]
            docid = example["docid"]
            if qid not in rst_dict:
                rst_dict[qid] = [(docid, d_score)]
            else:
                rst_dict[qid].append((docid, d_score))
            
            combine_features.append({"qid":qid, 
                                     "docid":docid, 
                                     "neural_feature":d_feature, 
                                     "retrival_score":r_score, 
                                     "neural_score":d_score})
    # get full metrics
    metrics = utils.get_metrics(
        rst_dict=rst_dict,
        qrels_path=os.path.join(args.target_dir, "qrels"),
        save_filename=os.path.join(args.save_folder, 'best_%s.trec'%mode),
        trec_mark=args.mode_name,
    )

    # empty cuda cache
    torch.cuda.empty_cache()
    return metrics, combine_features






# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # setting args
    parser = argparse.ArgumentParser(
        'MetaAdaptRank', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    config.add_default_args(parser)

    args = parser.parse_args()
    config.init_args_config(args)
    
    # Setup CUDA, GPU
    assert torch.cuda.is_available() is True
    args.cuda = True
    args.device = torch.device("cuda")
    args.n_gpu = 1
    
    # random seed
    utils.set_seed(args)
    
    # load tokenizer
    tokenizer = dataloaders.select_tokenizer(args)
    
    # select dataloader
    dataloder_dict = dataloaders.select_data_loader(args)
    
    # input batchify
    _, batchify_eval_inputs = utils.select_input_refactor(args)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    ## test data loader
    test_dataset = dataloder_dict["build_eval_dataset"](
        args=args, 
        data_dir=args.target_dir,
        tokenizer=tokenizer,
        mode="test",
    )

    args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=args.data_workers,
        collate_fn=dataloder_dict["eval_batchify"],
        pin_memory=args.cuda,
    )
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    ## load testing model
    logger.info("*"*20 + "Loading checkpoint" + "*"*20)
    try:
        model, checkpoint_updates = MetaRanker.load_checkpoint(
            args, 
            args.load_checkpoint_folder,
            checkpoint_name=args.checkpoint_name
        )        
    except ValueError:
        logger.info("Could't load checkpoint from %s" % args.load_checkpoint_folder)
    
    # set model device
    model.set_device()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    ## start testing !
    test_metrics, test_features = evaluate(
        args,
        model,
        eval_loader=test_data_loader,
        eval_dataset=test_dataset, 
        batchify_eval_inputs=batchify_eval_inputs, 
        mode="test"
    )
    logger.info('[Test] NDCG_20 = %.4f | P_20 = %.4f'%(test_metrics["ndcg_cut_20"], test_metrics["P_20"]))

    # save features
    utils.save_features(test_features, os.path.join(args.save_folder, "best_test_features.jsonl"))
    logger.info("Success saved testing features !")