import os
import math
import torch
from torch import nn, optim
import logging
import numpy as np
import torch.nn.functional as F
import utils
from utils import override_args
from metaranker import networks, losses, stepoptims, dataloaders
from metaranker import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    MagicModule,
    
)

logger = logging.getLogger()



class MetaRanker(object):
    
    def __init__(self, args, state_dict=None):
        
        self.args = args
        self.updates = 0
        self.parallel = False
        self.distributed = False
        # select ranker
        args.num_labels = 1 if args.loss_class != 'pointwise' else 2
        self.network = networks.get_class(args)
        self.convert_grad2delta = stepoptims.get_class(args)

        # load checkpoint
        if state_dict is not None:
            self.network.load_state_dict(state_dict)
            logger.info('loaded checkpoint state_dict')
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Initial Optimizer
    def init_optimizer(self, num_total_steps):
        warmup_proportion = (self.args.num_warmup_steps / num_total_steps) * 100
        logger.info('warmup step = %d | warm up proportion = %.2f'
                    %(self.args.num_warmup_steps, warmup_proportion) + "%")
        
      
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
        self.optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        # init scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps, 
            num_training_steps=num_total_steps)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Initial Meta-Optimizer
    def meta_initialize(self):
        if self.parallel:
            return ("exp_avg" in self.optimizer.state[next(self.network.module.parameters())])
        else:
            return ("exp_avg" in self.optimizer.state[next(self.network.parameters())])
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Initial Update for Meta-Learning
    def init_update(self, inputs):        
        self.network.train()
        cost = self.network(**inputs)
        eps = torch.zeros(cost.size(), requires_grad=True).to(self.args.device)
        loss = torch.sum(cost * eps)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # create meta model
    def create_meta_model(self):
        meta_model = MagicModule(self.network)
        
        # load self.network new parameter
        meta_model.to(self.args.device)
        
        # parallel or distributed
        if self.parallel:
            meta_model = torch.nn.DataParallel(meta_model)
        return meta_model
    
    
    @staticmethod
    def get_name2grad(grads, named_buffers):
        j = 0
        name2grad = {}
        for n, p in named_buffers:
            if p.requires_grad:
                name2grad[n] = grads[j]
                j += 1
        return name2grad

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    # meta learning 
    def meta_update(self, step, train_inputs, target_inputs):
        # Train mode

        # ------------------------------------------------------------------------------
        # initialize a dummy network for the meta learning of the weights
        meta_model = self.create_meta_model()

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        meta_model.train()
        cost = meta_model(**train_inputs)
        
        # ------------------------------------------------------------------------------
        # disturbance
        eps = torch.zeros(cost.size(), requires_grad=True).to(self.args.device)
        l_f_meta = torch.sum(cost * eps)
        meta_model.zero_grad()
        
        # ------------------------------------------------------------------------------
        # Line 6 perform a parameter update            
        if self.parallel:
            grads = torch.autograd.grad(
                l_f_meta, 
                [p for n, p in meta_model.module.named_buffers() if p.requires_grad], 
                create_graph=True
            )
            grads = self.get_name2grad(grads, meta_model.module.named_buffers())
            deltas = self.convert_grad2delta(grads, self.optimizer)
            meta_model.module.update_params(deltas)
        else:
            grads = torch.autograd.grad(
                l_f_meta, 
                [p for n, p in meta_model.named_buffers() if p.requires_grad], 
                create_graph=True
            )
            grads = self.get_name2grad(grads, meta_model.named_buffers())
            deltas = self.convert_grad2delta(grads, self.optimizer)
            meta_model.update_params(deltas)

        # ------------------------------------------------------------------------------
        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        l_g_meta = meta_model(**target_inputs)
        l_g_meta = torch.mean(l_g_meta)
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        # ------------------------------------------------------------------------------
        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # ------------------------------------------------------------------------------
        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        self.network.train()
        cost = self.network(**train_inputs)
        l_f = torch.sum(cost * w)
        
        if self.args.gradient_accumulation_steps > 1:
            l_f = l_f / self.args.gradient_accumulation_steps
        
        l_f.backward()
            
        # ------------------------------------------------------------------------------
        # update
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()

            self.optimizer.zero_grad()
            self.network.zero_grad()
            self.updates += 1
        
        return l_f.item(), l_g_meta.item()
    
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # common update for fine-tuning meta-checkpoints
    def common_update(self, step, train_inputs):
        self.network.train()

        cost = self.network(**train_inputs)
        l_f = torch.mean(cost)
        
        l_f.backward()
        
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()

            self.optimizer.zero_grad()
            self.network.zero_grad()
            self.updates += 1

        return l_f.item()
    
    # predict
    def predict(self, test_inputs):
        self.network.eval()
        with torch.no_grad():
            doc_scores, doc_features = self.network(**test_inputs)
        d_scores = doc_scores.detach().cpu().tolist()
        d_features = doc_features.detach().cpu().tolist()
        return d_scores, d_features
    
    
    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def save_checkpoint(self, checkpoint_folder, step, if_best=False):
        network = self.network.module if hasattr(self.network, 'module') else self.network
        params = {
            'args': self.args,
            'step': step,
            'state_dict': network.state_dict(),
        }
        checkpoint_name = "step_best" if if_best else "step_%d"%step
        try:
            torch.save(params, os.path.join(checkpoint_folder, '%s.checkpoint'%checkpoint_name))
            logger.info('success save step_%d checkpoints !' % step)
            if self.args.save_optimizer:
                torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_folder, "%s.optimizer"%checkpoint_name))
                torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_folder, "%s.scheduler"%checkpoint_name))
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')
            

    @staticmethod
    def load_checkpoint(new_args, checkpoint_folder, checkpoint_name):
        assert os.path.isfile(os.path.join(checkpoint_folder, '%s.checkpoint'%checkpoint_name))
        
        logger.info('Loading Checkpoint ...')
        saved_params = torch.load(
            os.path.join(checkpoint_folder, '%s.checkpoint'%checkpoint_name), 
            map_location=lambda storage, loc:storage
        )
        args = saved_params['args']
        step = saved_params['step']
        state_dict = saved_params['state_dict']
        if new_args:
            args = override_args(args, new_args)
            
        model = MetaRanker(args, state_dict)
        logger.info('Success Loaded step_%d checkpoints ! From : %s' % (step, checkpoint_folder))
        return model, step
    
    
    def load_optimizer(self, checkpoint_folder, checkpoint_name):
        logger.info('Loading Optimizer & Scheduler ...')
        
        assert os.path.isfile(os.path.join(checkpoint_folder, "%s.optimizer"%checkpoint_name)) and \
        os.path.isfile(os.path.join(checkpoint_folder, "%s.scheduler"%checkpoint_name))
        
        # Load in optimizer and scheduler states
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_folder, "%s.optimizer"%checkpoint_name)))
        self.scheduler.load_state_dict(torch.load(os.path.join(checkpoint_folder, "%s.scheduler"%checkpoint_name)))
        logger.info('Success Loaded Optimizer & Scheduler !')

    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    def zero_grad(self):
        self.optimizer.zero_grad() 
        self.network.zero_grad()
        
    def set_device(self):
        self.network.to(self.args.device)