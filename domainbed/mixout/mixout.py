##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Cheolhyoung Lee
## Department of Mathematical Sciences, KAIST
## Email: cheolhyoung.lee@kaist.ac.kr
## Implementation of mixout from https://arxiv.org/abs/1909.11299
## "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch.autograd.function import InplaceFunction
from typing import Optional
from collections import OrderedDict


class Mixout(InplaceFunction):
    # target: a weight tensor mixes with a input tensor
    # A forward method returns 
    # [(1 - Bernoulli(1 - p) mask) * target + (Bernoulli(1 - p) mask) * input - p * target]/(1 - p) 
    # where p is a mix probability of mixout.
    # A backward returns the gradient of the forward method.
    # Dropout is equivalent to the case of target=None. 
    # I modified the code of dropout in PyTorch. 
    @staticmethod
    def _make_noise(input:torch.Tensor, drop_mode:str='point') -> torch.Tensor:
        if drop_mode == 'filter':
            shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        elif drop_mode == 'point': 
            shape = input.shape
        else:
            raise ValueError(f"drop_mode should be either 'point' or 'filter', but got {drop_mode}")
        
        return input.new_empty(shape)

    @classmethod
    def forward(cls, 
                ctx, 
                input:torch.Tensor, 
                target:Optional["OrderedDict[str, torch.Tensor]"]=None, 
                p:float=0.0, 
                training:bool=False, 
                inplace:bool=False,
                scale_by_keep:bool=True,
                drop_mode:str='point') -> torch.Tensor:

        if p < 0 or p > 1:
            raise ValueError(f"A mix probability of mixout has to be between 0 and 1,  but got {p}")

        if target is not None and input.size() != target.size():
            raise ValueError(f"A target tensor size must match with a input tensor size {input.size()}, but got {target.size()}")
        
        ctx.p = p    
        ctx.training = training
        ctx.drop_mode = drop_mode
        ctx.scale_by_keep = scale_by_keep

        keep_prob = 1 - ctx.p

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
            target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p == 0:
            return output
        
        if not ctx.training:
            if not ctx.scale_by_keep:
                output.mul_(keep_prob).add_(ctx.p * target)
            return output
        
        ctx.noise = cls._make_noise(input, ctx.drop_mode)
        ctx.noise.bernoulli_(keep_prob)

        # one mask for all channels
        # if len(ctx.noise.size()) == 1 or ctx.drop_mode == 'filter':
        #     ctx.noise.bernoulli_(keep_prob)
        # else:
        #     ctx.noise[0].bernoulli_(keep_prob)
        #     ctx.noise = ctx.noise[0].repeat(input.size()[0], *([1] * (len(input.size())-1)))
        # ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target.clone()
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output)
            if ctx.scale_by_keep:
                output.add_(ctx.p * target, alpha=-1).div_(keep_prob)
        
        return output


    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> Optional[torch.Tensor]:
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None, None, None
        else:
            return grad_output, None, None, None, None, None, None