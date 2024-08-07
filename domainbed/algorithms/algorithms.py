# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
from functools import partial

from domainbed import networks
from domainbed.optimizers import get_optimizer, AMPOptimizerWrapper

import domainbed.models.loralib as lora
from domainbed.mixout.mixlinear import MixConv2d

def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches

def linear_combination(alpha, state_dict1, beta, state_dict2):
    state_dict_now = {}
    for k in state_dict1.keys():
        state_dict_now[k] = alpha * state_dict1[k].cuda() + beta * state_dict2[k].cuda()    
    return state_dict_now

def to_item(x):
    return x.item() if isinstance(x, torch.Tensor) else x

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams
    
    def load_weights(self, ckp):
        self.load_state_dict(ckp)

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def update_parameters(self, *args, **kwargs):
        pass

    def post_load(self, *args, **kwargs):
        pass

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.g_dp = hparams["group_dropout"]
        self.r_dp = hparams["resnet_dropout"]

        if not hparams["pretrained"]:
            hparams["freeze_bn"] = False

        self.warmup = hparams["warmup"]
        self.l2 = hparams["l2"]

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optim = get_optimizer(
            hparams["optimizer"],
            [
                {'params': self.featurizer.parameters(), 'weight_decay': hparams["weight_decay"] if hparams["l2"] == 0.0 else 0.0},
                {'params': self.classifier.parameters(), 'weight_decay': hparams["weight_decay"]}
            ],
            lr=self.hparams["lr"],
            momentum=0.9,
        )

        if hparams["drop_zero"]:
            print("Zeroing out target Weights")

        for mod in self.featurizer.modules():
            if isinstance(mod, MixConv2d):
                if hparams["drop_zero"]:
                    mod.target_w.data.zero_()
                    if mod.bias:
                        mod.target_b.data.zero_()
                else:
                    mod.target_w.data.copy_(mod.weight.data)

                    if mod.bias:
                        mod.target_b.data.copy_(mod.bias.data)
                
                mod.activation = hparams["drop_activation"]
                mod.drop_mode = hparams["drop_mode"]
                mod.scale_by_keep = hparams["drop_scale"]
                mod.noise_lambda = hparams["drop_noise"]

        if self.warmup >= 0:
            self.dropout_reg(r_dp=0.0, g_dp=0.0)

        if hparams["n_steps"] <= 10000:
            milestones = [5000]
        elif hparams["n_steps"] == 30000:
            milestones = [15000, 25000]
        else:
            milestones = [25000, 40000]

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=0.1)

        self.optimizer = AMPOptimizerWrapper(self.optim, self.scheduler, disable=not self.hparams["use_amp"])
        
        self.lp = hparams["lp"]
        if self.lp:
            print("LP ON")
            self.freeze_params(self.featurizer, freeze=True)
        
    def freeze_params(self, module, freeze=True):
        for param in module.parameters():
            param.requires_grad = not freeze

        for mod in module.modules():
            if isinstance(mod, MixConv2d):
                mod.target_w.requires_grad = False
                if mod.target_b:
                    mod.target_b.requires_grad = False      

    def dropout_reg(self, r_dp, g_dp):
        self.featurizer.dropout.p = r_dp

        for mode_name, mod in self.featurizer.named_modules():
            if isinstance(mod, MixConv2d):
                mod.p = g_dp
        
        print(f"Changed to Dropout: {r_dp}, Group Dropout: {g_dp}")
        
    def l2_reg(self):
        l2 = 0.0
        for mod in self.featurizer.modules():
            if isinstance(mod, MixConv2d):
                l2 += F.mse_loss(mod.weight, mod.target_w, reduction='mean')

        return l2/2

    def update(self, x, y, step, **kwargs):

        if self.warmup > 0 and step == self.warmup and self.lp:
            self.freeze_params(self.featurizer, freeze=False)
            self.lp = False
            print("LP OFF")

        if step == self.warmup and (self.r_dp > 0.0 or self.g_dp > 0.0):
            self.dropout_reg(r_dp=self.r_dp, g_dp=self.g_dp)

        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=self.hparams["use_amp"]):

            all_x = torch.cat(x)
            all_y = torch.cat(y)

            if self.lp:
                with torch.no_grad():
                    features = self.featurizer(all_x)
                preds = self.classifier(features)
            else:
                preds = self.network(all_x)

            loss = F.cross_entropy(preds, all_y)
            acc = (preds.argmax(1) == all_y).float().mean()

            if step >= self.warmup and self.l2 > 0.0:
                if step == self.warmup:
                    print("L2 REGULARIZATION ON")

                loss += self.l2 * self.l2_reg()

        self.optimizer.backward(loss)
        self.optimizer.step()
        self.optimizer.lr_step()

        return {"loss": to_item(loss), 'acc': to_item(acc)}

    def clone(self):
        clone = super().clone()
        clone.optim = clone.optimizer
        clone.optimizer = AMPOptimizerWrapper(clone.optim, disable=not clone.hparams["use_amp"])

        return clone

    def train(self, mode=True):
        super().train(mode)

    def predict(self, x):
        return self.network(x)

class MA(ERM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network_ma = copy.deepcopy(self.network)
        self.network_ma.eval()
        self.ma_start_iter = 300
        self.global_iter = 0
        self.ma_count = 0
    
    def load_weights(self, ckp):
        mising_keys, _ = self.load_state_dict(ckp, strict=False)
        for k in mising_keys:
            if "network_ma" in k:
                continue
            else:
                raise ValueError(f"Missing key: {k}")

        self.network_ma = copy.deepcopy(self.network)
        self.network_ma.eval()

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        self.update_ma()
        return result

    def predict(self, x):
        self.network_ma.eval()
        return self.network_ma(x)

    def update_ma(self):
        self.global_iter += 1
        if self.global_iter>= self.ma_start_iter:
            self.ma_count += 1
            for param_q, param_k in zip(self.network.parameters(), self.network_ma.parameters()):
                param_k.data = (param_k.data * self.ma_count + param_q.data)/(1.+self.ma_count)
        else:
            for param_q, param_k in zip(self.network.parameters(), self.network_ma.parameters()):
                param_k.data = param_q.data

class DiWA(Algorithm):
    def __init__(self, *args, **kwargs):
        self.ens_size = kwargs.pop('ens_size', 1)

        super().__init__(*args, **kwargs)

        self.average_weights = self.hparams["average_weights"]
        self.ens_size = self.hparams["ens_size"]
        self.networks = nn.ModuleList(ERM(*args, **kwargs) for _ in range(self.ens_size))

    def post_load(self, **kwargs):
        if self.average_weights:
            sd0 = self.networks[0].state_dict()
            for exp in self.networks[1:]:
                sd0 = linear_combination(1.0, sd0, 1.0, exp.state_dict())
            sd0 = {k: v/len(self.networks) for k, v in sd0.items()}
            self.networks[0].load_state_dict(sd0)

    def predict(self, x):
        if self.average_weights:
            return self.networks[0].predict(x)
        else:
            preds = [F.log_softmax(network.predict(x), dim=1) for network in self.networks]
            return torch.stack(preds, dim=1)

class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.hparams["use_amp"]):

            features = [self.featurizer(xi) for xi, _ in minibatches]
            classifs = [self.classifier(fi) for fi in features]
            targets = [yi for _, yi in minibatches]

            for i in range(nmb):
                objective += F.cross_entropy(classifs[i], targets[i])
                for j in range(i + 1, nmb):
                    penalty += self.mmd(features[i], features[j])

            objective /= nmb
            if nmb > 1:
                penalty /= nmb * (nmb - 1) / 2

            loss = objective + self.hparams["mmd_gamma"] * penalty

        self.optimizer.backward(loss)
        self.optimizer.step()
        self.optimizer.lr_step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)
