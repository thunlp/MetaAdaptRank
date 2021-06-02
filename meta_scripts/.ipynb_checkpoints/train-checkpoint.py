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
from tensorboardX import SummaryWriter

logger = logging.getLogger()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# meta_train
def meta_train(
    args, 
    tot_steps,
    tot_loader,
    writer):
    logger.info("Start meta-learning to rank ...")
    
    train_loss = utils.AverageMeter()
    target_loss = utils.AverageMeter()
    stats = {"best_%s"%args.main_metric: 0, "best_updates":0}
    
    for step in tqdm(range(tot_steps)):
        # ---------------------------------------------
        # ---------------------------------------------
        # generate train & target inputs
        train_inputs, _ = tot_loader.generate_train_inputs()
        target_inputs, _ = tot_loader.generate_target_inputs()
        # ---------------------------------------------
        # ---------------------------------------------
        # initilize optimizer update
        if not model.meta_initialize():
            loss = model.init_update(train_inputs)
            torch.cuda.empty_cache()
            logger.info("Init loss = {}".format(loss))
        # ---------------------------------------------
        # ---------------------------------------------
        # meta update
        try:
            l_f_, l_g_meta_ = model.meta_update(
                step=step, 
                train_inputs=train_inputs,
                target_inputs=target_inputs,
            )
        except:
            logging.error(str(traceback.format_exc()))
            break
            
        train_loss.update(l_f_)
        target_loss.update(l_g_meta_)

        # ---------------------------------------------
        # ---------------------------------------------
        # log tensorboard
        if (step + 1) % int(args.display_iter * args.gradient_accumulation_steps) == 0:
            writer.add_scalar('meta_train/loss', train_loss.avg, model.updates)
            writer.add_scalar('meta_target/loss', target_loss.avg, model.updates)
            writer.add_scalar('meta_train/lr', model.scheduler.get_last_lr()[0], model.updates)
            train_loss.reset()
            target_loss.reset()
            
        # ---------------------------------------------
        # ---------------------------------------------
        # eval & save checkpoint
        if args.eval_during_train and (((step + 1) % int(args.eval_step * args.gradient_accumulation_steps) == 0) or step == 0):
            stats = update_eval_scores(
                args=args, 
                model=model, 
                tot_loader=tot_loader, 
                stats=stats, 
                writer=writer,
                mode="dev"
            )
        # ---------------------------------------------
        # ---------------------------------------------
        # early stop
        if int(model.updates - stats["best_updates"]) > args.early_stop_step:
            logger.info("Early Stop ! | Now Updates = {}, Best Updates = {}".format(model.updates, stats["best_updates"]))
            break

            

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# finetune_train
def finetune_train(
    args, 
    tot_steps,
    tot_loader,
    writer):
    logger.info("Start Fine-tune training ...")
    
    train_loss = utils.AverageMeter()
    stats = {"best_%s"%args.main_metric: 0, "best_updates":0}
    
    for step in tqdm(range(tot_steps)):
        train_inputs, _ = tot_loader.generate_train_inputs()
        
        try:
            l_f_ = model.common_update(
                step=step, 
                train_inputs=train_inputs,
            )
        except:
            logging.error(str(traceback.format_exc()))
            break
            
        train_loss.update(l_f_)

        # ---------------------------------------------
        # ---------------------------------------------
        if (step + 1) % int(args.display_iter * args.gradient_accumulation_steps) == 0:
            writer.add_scalar('finetune_train/loss', train_loss.avg, model.updates)
            writer.add_scalar('finetune_train/lr', model.scheduler.get_last_lr()[0], model.updates)
            train_loss.reset()
            
        # ---------------------------------------------
        # ---------------------------------------------
        # eval & save checkpoint
        if args.eval_during_train and (((step + 1) % int(args.eval_step * args.gradient_accumulation_steps) == 0) or step == 0):
            stats = update_eval_scores(
                args=args, 
                model=model, 
                tot_loader=tot_loader, 
                stats=stats, 
                writer=writer,
                mode="dev"
            )
        # ---------------------------------------------
        # ---------------------------------------------
        # early stop
        if int(model.updates - stats["best_updates"]) > args.early_stop_step:
            logger.info("Early Stop ! | Now Updates = {}, Best Updates = {}".format(model.updates, stats["best_updates"]))
            break


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# update eval scores
def update_eval_scores(args, model, tot_loader, stats, writer, mode):
    eval_metrics, combine_features = evaluate(
        args,
        model,
        eval_loader=tot_loader.dev_loader,
        eval_dataset=tot_loader.dev_dataset, 
        batchify_eval_inputs=tot_loader.batchify_eval_inputs, 
        mode=mode
    )
    # log tensorboard
    writer.add_scalar('%s/%s'%(mode, args.main_metric), eval_metrics[args.main_metric], model.updates)
    
    # compare previous scores
    if eval_metrics[args.main_metric] >= stats['best_%s'%args.main_metric]:
        
        stats['best_%s'%args.main_metric] = eval_metrics[args.main_metric]
        stats['best_updates'] = model.updates
        
        logger.info("\n [%s] Update ! Best_NDCG@20 = %.4f | Best_P@20 = %.4f (updates=%d)"
                    %(mode, stats['best_%s'%args.main_metric], eval_metrics["P_20"], model.updates))

        # rename latest.trec to best.trec
        os.rename(os.path.join(args.save_folder, 'latest_%s.trec'%mode), os.path.join(args.save_folder, 'best_%s.trec'%mode))

        # save features
        utils.save_features(combine_features, os.path.join(args.save_folder, "best_%s_features.jsonl"%mode))
        
        # save checkpoint
        if args.save_checkpoint:
            model.save_checkpoint(args.checkpoint_folder, model.updates, if_best=True)
    return stats


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
        save_filename=os.path.join(args.save_folder, 'latest_%s.trec'%mode),
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
    args.n_gpu = 1
    args.device = torch.device("cuda")
    
    # random seed
    utils.set_seed(args)
    
    # set tensorboard
    tb_writer = SummaryWriter(args.viso_folder)
    
    # load tokenizer
    tokenizer = dataloaders.select_tokenizer(args)
    
    # select dataloader
    dataloder_dict = dataloaders.select_data_loader(args)
    # input batchify
    batchify_train_target_inputs, batchify_eval_inputs = utils.select_input_refactor(args)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    ## train data loader
    train_dataset = dataloder_dict["build_train_target_dataset"](
        args=args, 
        data_dir=args.train_dir,
        tokenizer=tokenizer,
        mode="train",
    )
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=dataloder_dict["train_target_batchify"],
        pin_memory=args.cuda,
    )
    
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    ## target data loader
    if args.mode_name == "meta":
        logger.info("[Use Meta-learning]")
        target_dataset = dataloder_dict["build_train_target_dataset"](
            args=args, 
            data_dir=args.target_dir,
            tokenizer=tokenizer,
            mode="target",
        )

        args.target_batch_size = args.train_batch_size
        target_sampler = torch.utils.data.sampler.RandomSampler(target_dataset)
        target_data_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=args.target_batch_size,
            sampler=target_sampler,
            num_workers=args.data_workers,
            collate_fn=dataloder_dict["train_target_batchify"],
            pin_memory=args.cuda,
        )
    else:
        target_data_loader = None
        
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    ## dev data loader
    dev_dataset = dataloder_dict["build_eval_dataset"](
        args=args, 
        data_dir=args.target_dir,
        tokenizer=tokenizer,
        mode="dev",
        extra_dataset=target_dataset if args.mode_name == "meta" else None
    )

    args.dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.dev_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=dataloder_dict["eval_batchify"],
        pin_memory=args.cuda,
    )
    
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    ## total data loader   
    tot_data_loader = TotalDataLoader(
        args, 
        train_loader=train_data_loader, 
        target_loader=target_data_loader, 
        dev_loader=dev_data_loader,
        dev_dataset=dev_dataset,
        batchify_train_inputs=batchify_train_target_inputs,
        batchify_eval_inputs=batchify_eval_inputs,
    )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Set training total steps
    if args.max_train_steps > 0:
        t_total = args.max_train_steps
        args.max_train_epochs = \
        args.max_train_steps // (len(train_data_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.max_train_epochs
    

    logger.info("*"*20 + "Initilize Model & Optimizer" + "*"*20)
    # Preprare Model & Optimizer
    if args.load_checkpoint:
        model, checkpoint_updates = MetaRanker.load_checkpoint(
            args, 
            args.load_checkpoint_folder,
            checkpoint_name=args.checkpoint_name
        )        
    else:
        logger.info('Training model from scratch...')
        model = MetaRanker(args)
    
    # initial optimizer
    model.init_optimizer(num_total_steps=t_total)
    if args.load_optimizer:
        model.load_optimizer(args.load_checkpoint_folder, checkpoint_name=args.checkpoint_name)
    
    # set model device
    model.set_device()
    
    # clear grad
    model.zero_grad()

    logger.info("Training/evaluation parameters %s", args)
    logger.info("*"*50)
    logger.info("  Num Train examples = %d", len(train_dataset))
    logger.info("  Num Train Epochs = %d", args.max_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info(
        "  Total train batch size (w. accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("*"*50)
    
    
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    ## start training !
    if args.mode_name == "meta":
        meta_train(
            args, 
            tot_steps=int(t_total),
            tot_loader=tot_data_loader,
            writer=tb_writer,
        )
    else:
        finetune_train(
            args, 
            tot_steps=int(t_total),
            tot_loader=tot_data_loader,
            writer=tb_writer,
        )