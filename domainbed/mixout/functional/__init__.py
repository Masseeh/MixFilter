import torch
from ..mixout import Mixout
from typing import Optional
from collections import OrderedDict


def mixout(input:torch.Tensor, 
           target:Optional["OrderedDict[str, torch.Tensor]"]=None, 
           p:float=0.0, 
           training:bool=False, 
           inplace:bool=False,
           scale_by_keep:bool=True,
           noise_lambda:float=0.0,
           drop_mode:str='point') -> torch.Tensor:
       
       noisy_target = target + (torch.rand(target.size(), device=target.device) - 0.5)*noise_lambda*torch.std(target)
       return Mixout.apply(input, noisy_target, p, training, inplace, scale_by_keep, drop_mode)