import math
import torch

def get_adam_delta(grads, optimizer):
    deltas = {}
    for group in optimizer.param_groups:
        for n, p in zip(group['names'], group['params']):
            grad = grads[n]
            state = optimizer.state[p]

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            step = state['step'] + 1
            
            if group['weight_decay'] != 0:
                grad = grad + group['weight_decay'] * p.data

            bias_correction1 = 1. - beta1 ** step
            bias_correction2 = 1. - beta2 ** step

            step_size = group['lr'] / bias_correction1

            _exp_avg = exp_avg * beta1 + (1. - beta1) * grad
            _exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad

            denom = (torch.sqrt(_exp_avg_sq + group['eps']) / math.sqrt(bias_correction2)).add_(group['eps'])
            deltas[n] = -step_size * _exp_avg / denom
    return deltas