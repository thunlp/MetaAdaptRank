import os
import math
import torch
from torch import nn, optim
import logging
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import utils
from contrastqg import (T5ForConditionalGeneration, AdamW)

logger = logging.getLogger()

class QGenerator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.updates = 0
        self.network = T5ForConditionalGeneration.from_pretrained(args.pretrain_generator_type)
        self.network.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.batchify_inputs = utils.select_gen_input_refactor(args)
        
        if args.run_mode == "inference":
            self.network.load_state_dict(torch.load(args.generator_load_dir + '/models.pkl'))
            logger.info("sccuess load checkpoint from {} !".format(args.generator_load_dir))
            
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Initial Optimizer
    def init_optimizer(self):      
        self.no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.network.named_parameters() if not any(nd in n for nd in self.no_decay)],
             "weight_decay": self.args.weight_decay,
             "names": [n for n, p in self.network.named_parameters() if not any(nd in n for nd in self.no_decay)],
            },
            {"params": [p for n, p in self.network.named_parameters() if any(nd in n for nd in self.no_decay)], 
             "weight_decay": 0.0,
             "names": [n for n, p in self.network.named_parameters() if any(nd in n for nd in self.no_decay)], 
            },
        ]

        # init optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        
        
    # train
    def update(self, step, inputs):
        self.network.train()
        
        outputs = self.network(**inputs)
        loss = outputs[0]
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        loss.backward()
        
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1

        return loss.item()
        
        
    def predict(self, inputs):        
        self.network.eval()
        outputs = self.network.generate(**inputs)
        pred_tokens = self.tokenizer.convert_outputs_to_tokens(outputs)
        return pred_tokens
    
    
    def zero_grad(self):
        self.optimizer.zero_grad() 
        self.network.zero_grad()

        
    def set_device(self, device):
        self.device = device
        self.network.to(self.device)
        
        
    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
        
    def reset_train_iter(self, train_loader=None):
        if train_loader is not None:
            self.train_loader = train_loader
        self.train_iterator = iter(self.train_loader)
        
    def generate_train_inputs(self):
        try:
            train_batch = next(self.train_iterator)
        except StopIteration:
            self.reset_train_iter()
            train_batch = next(self.train_iterator)
        train_inputs, train_indexs = self.batchify_inputs(train_batch, device=self.args.device)
        return train_inputs, train_indexs
        
    
    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def save_checkpoint(self, checkpoint_folder, step):
        network = self.network.module if hasattr(self.network, 'module') else self.network
        try:
            torch.save(network.state_dict(), checkpoint_folder + '/models.pkl')
            self.tokenizer.subtokenizer.save_pretrained(checkpoint_folder)
            logger.info('success save step_%d checkpoints !' % step)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')