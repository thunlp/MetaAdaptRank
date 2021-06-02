import torch
from torch import nn
from torch.autograd import Variable

class PairWise(nn.Module):
    def __init__(
        self, 
        margin=1
    ):
        super(PairWise, self).__init__()
        self.tanh = nn.Tanh()
        self.loss_fct = nn.MarginRankingLoss(
            margin=margin, 
            reduction='none'
        )
    def forward(
        self, 
        pos_score, 
        neg_score,
    ):
        pos_score = self.tanh(pos_score)
        neg_score = self.tanh(neg_score)
        # compute loss
        batch_loss = self.loss_fct(
            pos_score, 
            neg_score, 
            target=torch.ones(pos_score.size()).to(pos_score.device)
        )
        return batch_loss