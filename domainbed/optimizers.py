import torch
from torch.cuda.amp import GradScaler

class AMPOptimizerWrapper:
    def __init__(self, optimizer, scheduler=None, disable=False):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = sum([group["params"] for group in optimizer.param_groups], [])
        self.disable = disable
        if not disable:
            self.scaler = GradScaler()
            self.last_scale = None

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def backward(self, loss, retain_graph=False):
        if self.disable:
            loss.backward(retain_graph=retain_graph)
        else:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)

    def clip_grad_norm(self, max_norm):
        if not self.disable:
            self.scaler.unscale_(self.optimizer)
        return torch.nn.utils.clip_grad_norm_(self.params, max_norm)

    def step(self):
        if self.disable:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.last_scale = self.scaler.get_scale()
            self.scaler.update()

    def lr_step(self):
        if self.scheduler is None:
            return
        
        if self.disable:
            self.scheduler.step()
        else:
            skip_lr_sched = (self.last_scale > self.scaler.get_scale())
            if not skip_lr_sched:
                self.scheduler.step()

def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optim_cls = optimizers[name]
    if name != "sgd":
        kwargs.pop("momentum")

    return optim_cls(params, **kwargs)
