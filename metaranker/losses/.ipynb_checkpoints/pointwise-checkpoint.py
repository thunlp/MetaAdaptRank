import torch
from torch import nn
import torch.nn.functional as F


class PointWise(nn.Module):
    def __init__(
        self,
    ):
        super(PointWise, self).__init__()
    def forward(
        self, 
        pos_score, 
        labels,
    ):
        # compute loss
        batch_loss = F.cross_entropy(pos_score, labels, reduction="none")
        return batch_loss