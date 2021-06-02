import torch
import torch.nn as nn
import logging
from .. import losses
from ..transformers import BertPreTrainedModel, BertModel

from typing import Tuple
logger = logging.getLogger()



class BertRanker(BertPreTrainedModel):
    def __init__(
        self, 
        config,
        loss_class,
    ):
        super().__init__(config)
        self._loss_class = loss_class
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config)
        self.linear_layer = nn.Linear(config.hidden_size, config.num_labels)            
        self.loss_fct = losses.get_class(self._loss_class)
        
        self.init_weights()
        
    def forward(
        self, 
        pos_input_ids, 
        pos_input_mask, 
        pos_segment_ids, 
        neg_input_ids=None, 
        neg_input_mask=None, 
        neg_segment_ids=None,
        labels=None,
    ):
        # pos input
        _, pos_output = self.bert(
            pos_input_ids, 
            attention_mask = pos_input_mask, 
            token_type_ids = pos_segment_ids
        )
        
        pos_score = self.linear_layer(pos_output).squeeze(-1)
        
        if self._loss_class == "pointwise":
            if labels is not None:
                loss = self.loss_fct(pos_score, labels)
                return loss
            else:
                return pos_score.softmax(dim=-1)[:, 1].squeeze(-1), pos_output
            
        elif self._loss_class == "pairwise":
            # pairwise loss
            if neg_input_ids is not None:
                _, neg_output = self.bert(
                    neg_input_ids, 
                    attention_mask = neg_input_mask, 
                    token_type_ids = neg_segment_ids
                )
                # pick cls token
                neg_score = self.linear_layer(neg_output).squeeze(-1)
                
                # compute loss
                loss = self.loss_fct(pos_score, neg_score)
                return loss
        
            # inference
            else:
                return pos_score, pos_output
