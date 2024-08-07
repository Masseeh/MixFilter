# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}

    hparams["use_amp"] = (True, True)
    hparams["device"] = (0, 0)
    hparams["ffcv"] = (True, True)

    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet18"] = (False, False)
    hparams["resnet_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    hparams["optimizer"] = ("adam", "adam")

    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    hparams["rx"] = (True, True) # resnext

    hparams["group_dropout"] = (0.0, 0.0)
    hparams["warmup"] = (0, 0)
    hparams["l2"] = (0.0, 0.0)

    hparams["lp"] = (False, False)
    hparams["drop_mode"] = ("filter", ["point", "filter", "block"])
    hparams["drop_activation"] = (False, False)
    hparams["drop_zero"] = (False, False)
    hparams["drop_scale"] = (True, True)
    hparams["drop_noise"] = (0.0, 0.0)
    hparams["average_weights"] = (False, False)
    hparams["ens_size"] = (20, 20)

    if dataset not in SMALL_IMAGES:
        hparams["lr"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        if dataset == "DomainNet":
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
        else:
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5.5)))
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in SMALL_IMAGES:
        hparams["weight_decay"] = (0.0, 0.0)
    else:
        hparams["weight_decay"] = (0.0, 10 ** random_state.uniform(-6, -2))

    if algorithm in ("MMD", "CORAL"):
        hparams["mmd_gamma"] = (1.0, 10 ** random_state.uniform(-1, 1))

    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
