import copy
import torch
import operator
import torch.nn as nn
from torch import Tensor, device, dtype
from typing import Callable, Dict, List, Optional, Tuple
import torch.nn.functional as F
from ..transformers import ModuleUtilsMixin, BertPreTrainedModel

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class MagicModule(nn.Module):
    def __init__(self, module):
        nn.Module.__init__(self)
        if isinstance(module, torch.nn.DataParallel):
            module = module.module
        
        self._type = type(module)

        for key, value in module._parameters.items():
            if value.requires_grad:
                self.register_buffer(key, nn.Parameter(value.data))
            else:
                 self.register_buffer(key, nn.Parameter(value.data, requires_grad=False))

        for key, value in module._modules.items():
            self.add_module(key, MagicModule(value))

        for key, value in module.__dict__.items():
            if (not key in self.__dict__) and\
                    (not key in self._buffers) and\
                    (not key in self._modules):
                self.__setattr__(key, value)

    def forward(self, *args, **kwargs):
        return self._type.forward(self, *args, **kwargs)
    
    def update_params(self, deltas):
        sub_params = {}
        for key, delta in deltas.items():
            if not ('.' in key):
                self._buffers[key] = self._buffers[key] + delta
            else:
                attr = key.split('.')[0]
                if not (attr in sub_params):
                    sub_params[attr] = {}
                sub_params[attr]['.'.join(key.split('.')[1:])] = delta            
        for key, value in sub_params.items():
            self._modules[key].update_params(value)
        

    def check_forward_args(self, *args, **kwargs):
        assert issubclass(self._type, nn.RNNBase)
        return nn.RNNBase.check_forward_args(self, *args, **kwargs)

    @property
    def _flat_weights(self):
        assert issubclass(self._type, nn.RNNBase)
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):
        assert issubclass(self._type, nn.RNNBase)
        return [[getattr(self, weight) for weight in weights] for weights in
                self._all_weights]

    def _get_abs_string_index(self, idx):
        assert issubclass(self._type, nn.ModuleList)
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if not issubclass(self._type, nn.ModuleList):
            print(self._type)
        assert issubclass(self._type, nn.ModuleList)
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __len__(self):
        assert issubclass(self._type, nn.ModuleList)
        return len(self._modules)

    ## -----------------------------------------------------------------
    ## -----------------------------------------------------------------
    ## Bert modules
    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple, device: device) -> Tensor:
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.
        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device
        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        
#         extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    
    def get_head_mask(self, head_mask: Tensor, num_hidden_layers: int, is_attention_chunked: bool = False) -> Tensor:
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
            num_hidden_layers: int
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
    
    
    def transpose_for_scores(self, x):
        """BertSelfAttention."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)