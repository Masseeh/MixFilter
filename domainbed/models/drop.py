""" DropBlock, DropPath

PyTorch implementations of DropBlock and DropPath (Stochastic Depth) regularization layers.

Papers:
DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)

Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)

Code:
DropBlock impl inspired by two Tensorflow impl that I liked:
 - https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
 - https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)

def drop_block_2d(
        x,
        fixed_x: torch.Tensor,
        drop_prob: float = 0.1,
        block_size: int = 7,
        gamma_scale: float = 1.0,
        batchwise: bool = False
):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = ndgrid(torch.arange(W, device=x.device), torch.arange(H, device=x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)

    # out = (x * block_mask + fixed_x * (1 - block_mask) - gamma * fixed_x) * normalize_scale
    out = (x * block_mask) * normalize_scale
    return out


def drop_block_fast_2d(
        x: torch.Tensor,
        fixed_x: torch.Tensor,
        drop_prob: float = 0.0,
        block_size: int = 7,
        gamma_scale: float = 1.0
):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    if drop_prob == 0.:
        return x

    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    block_mask = torch.empty_like(x).bernoulli_(gamma)
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    block_mask = 1 - block_mask

    normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
    out = (x * block_mask + fixed_x * (1 - block_mask) - gamma * fixed_x) * normalize_scale

    return out

def drop_point(x, fixed_x, drop_prob: float = 0., scale_by_keep: bool = True):
    if drop_prob == 0.:
        return x
    
    keep_prob = 1 - drop_prob
    shape = x.shape
    mask = x.new_empty(shape).bernoulli_(keep_prob)

    # one mask for all channels
    # shape = (x.shape[0], 1) + x.shape[2:] 
    # mask = x.new_empty(shape).bernoulli_(keep_prob)
    # mask = mask.repeat(1, x.shape[1], *([1] * (x.ndim - 2)))

    out = (x * mask + fixed_x * (1 - mask) - drop_prob * fixed_x)

    if keep_prob > 0.0 and scale_by_keep:
        return out.div_(keep_prob)
    else:
         return out

def drop_spatial(x, fixed_x, drop_prob: float = 0., scale_by_keep: bool = True):
    if drop_prob == 0.:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], x.shape[1],) + (1,) * (x.ndim - 2)  # work with diff dim tensors, not just 2D ConvNets
    mask = x.new_empty(shape).bernoulli_(keep_prob)

    out = (x * mask + fixed_x * (1 - mask) - drop_prob * fixed_x)

    if keep_prob > 0.0 and scale_by_keep:
        return out.div_(keep_prob)
    else:
         return out

def general_drop_out(x, fixed_x, p: float = 0., scale_by_keep: bool = True, drop_mode='point'):
    if drop_mode == 'point':
        return drop_point(x, fixed_x, p, scale_by_keep)
    elif drop_mode == 'filter':
        return drop_spatial(x, fixed_x, p, scale_by_keep)
    elif drop_mode == 'block':
        return drop_block_fast_2d(x, fixed_x, p, block_size=7)
    else:
        raise ValueError(f"Unknown drop_mode: {drop_mode}")