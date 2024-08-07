import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from ffcv.loader import Loader, OrderOption

from domainbed.datasets import get_dataset, split_dataset, set_transfroms
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import SmartZip, InfiniteDataLoader
from domainbed.datasets.transforms import ffcv_add_center_crop


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    hparams["n_steps"] = n_steps

    #######################################################
    # setup dataset & loader
    #######################################################

    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []
    if hparams.indomain_test > 0.0:
        logger.info("!!! In-domain test mode On !!!")
        assert hparams["val_augment"] is False, (
            "indomain_test split the val set into val/test sets. "
            "Therefore, the val set should be not augmented."
        )
        val_splits = []
        for env_i, (out_split, _weights) in enumerate(out_splits):
            n = len(out_split) // 2
            seed = misc.seed_hash(args.trial_seed, env_i)
            val_split, test_split = split_dataset(out_split, n, seed=seed)

            if hparams["ffcv"]:
                out_split.transforms = {}
                val_split.direct_return = False
                val_split.underlying_dataset = out_split.underlying_dataset
                test_split.direct_return = False
                test_split.underlying_dataset = out_split.underlying_dataset

                set_transfroms(val_split, "valid", hparams, algorithm_class)
                set_transfroms(test_split, "test", hparams, algorithm_class)

            val_splits.append((val_split, None))
            test_splits.append((test_split, None))
            logger.info(
                "env %d: out (#%d) -> val (#%d) / test (#%d)"
                % (env_i, len(out_split), len(val_split), len(test_split))
            )
        out_splits = val_splits

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()
    batch_multiplier = 1

    logger.info(f"Batch sizes for each domain: {[b*batch_multiplier for b in batch_sizes]} (total={sum([b*batch_multiplier for b in batch_sizes])})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / (batch_size*batch_multiplier)
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

    # setup loaders
    if hparams["ffcv"]:
        train_loaders = [
            Loader(
                fname=env.beton,
                os_cache=False,
                batch_size=batch_size,
                num_workers=1,
                order=OrderOption.QUASI_RANDOM,
                drop_last=True,
                indices=env.keys,
                seed=args.seed,
                pipelines={'image': env.transforms["x"],
                            'label': env.transforms["y"]})
            for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
        ]
    else:
        train_loaders = [
            InfiniteDataLoader(
                dataset=env,
                weights=env_weights,
                batch_size=batch_size,
                num_workers=dataset.N_WORKERS,
            )
            for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
        ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        if hparams["ffcv"]:
            loader_kwargs = {"fname": env.beton, "batch_size": batchsize, "num_workers": 1, "drop_last": False, "os_cache": False,
                            "order": OrderOption.SEQUENTIAL, "indices": env.keys, "pipelines": {'image': ffcv_add_center_crop(env.transforms["x"]), 'label': env.transforms["y"]}}

            loader_kwargs = Loader(**loader_kwargs)
            eval_loaders_kwargs.append(loader_kwargs)
        else:
            loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
            eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )

    if args.init_model:
        logger.info(f"Load model from {args.init_model}")
        test_env_str = ",".join(map(str, test_envs))
        ckp = Path(args.init_model)/"checkpoints"/f"TE{test_env_str}.pth"
        ckp = torch.load(ckp, map_location="cpu")
        algorithm.load_weights(ckp["model_dict"])

    algorithm.cuda()

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    train_minibatches_iterator = iter(SmartZip(*train_loaders))
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    best_results = {
        "step": 0,
        "score": 0,
    }

    for step in range(n_steps):
        step_start_time = time.time()
        
        batches_list = []
        batches_list = next(train_minibatches_iterator)
        batches = misc.merge_list(batches_list)
        
        if not hparams["ffcv"]:
            # to device
            batches = {
                key: [tensor.cuda() for tensor in tensorlist] for key, tensorlist in batches.items()
            }

        inputs = {**batches, "step": step}
        step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if step % checkpoint_freq == 0 or step == n_steps - 1:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            algorithm.update_parameters(step)

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
            writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            score = results["train_out"]
            if args.model_save and score > best_results["score"]:

                best_results["score"] = score
                best_results["step"] = step

                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)

                test_env_str = ",".join(map(str, test_envs))
                filename = f"TE{test_env_str}.pth"
                if len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}.pth"
                path = ckpt_dir / filename

                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }

                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

        if step % args.tb_freq == 0:
            # add step values only for tb log
            writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")

    # find best
    logger.info("---")
    records = Q(records)
    oracle_best = records.argmax("test_out")["test_in"]
    oracle_best_step = records.argmax("test_out")["step"]

    iid_best = records.argmax("train_out")["test_in"]
    iid_best_step = records.argmax("train_out")["step"]

    last = records[-1]["test_in"]
    last_step = records[-1]["step"]

    if hparams.indomain_test:
        # if test set exist, use test set for indomain results
        in_key = "train_inTE"
    else:
        in_key = "train_out"

    iid_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    ret = {
        "oracle": [oracle_best, oracle_best_step],
        "iid": [iid_best, iid_best_step],
        "last": [last, last_step],
        "last (inD)": [last_indomain, last_step],
        "iid (inD)": [iid_best_indomain, iid_best_step],
    }

    for k, acc in ret.items():
        logger.info(f"{k} = {acc[0]:.3%} (step={acc[1]})")

    return ret, records

def test(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################

    hparams["n_steps"] = n_steps

    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        if hparams["ffcv"]:
            loader_kwargs = {"fname": env.beton, "batch_size": batchsize, "num_workers": 1, "drop_last": False,
                            "order": OrderOption.SEQUENTIAL, "indices": env.keys, "pipelines": {'image': ffcv_add_center_crop(env.transforms["x"]), 'label': env.transforms["y"]}}

            loader_kwargs = Loader(**loader_kwargs)
            eval_loaders_kwargs.append(loader_kwargs)
        else:
            loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
            eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    np_model_test = np.array(args.model_test)
    if len(np_model_test) > hparams["ens_size"]:
        np_model_test.sort()
        rng = np.random.default_rng(seed=args.trial_seed)
        np_model_test = rng.choice(np_model_test, hparams["ens_size"], replace=False)
    else:
        hparams["ens_size"] = len(np_model_test)

    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams
    )

    test_env_str = ",".join(map(str, test_envs))
    for i, model_test in enumerate(np_model_test):
        ckp = Path(model_test)/"checkpoints"/f"TE{test_env_str}.pth"
        ckp = torch.load(ckp, map_location="cpu")
        algorithm.networks[i].load_state_dict(ckp["model_dict"], strict=True)
        algorithm.networks[i].hp = ckp['model_hparams']
    
    algorithm.post_load()
    algorithm.cuda()

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    step = 0
    results = {
        "step": step,
    }

    for key, val in checkpoint_vals.items():
        results[key] = np.mean(val)

    eval_start_time = time.time()
    accuracies, summaries = evaluator.evaluate(algorithm)
    results["eval_time"] = time.time() - eval_start_time

    # results = (epochs, loss, step, step_time)
    results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
    # merge results
    results.update(summaries)
    results.update(accuracies)

    # print
    if results_keys != last_results_keys:
        logger.info(misc.to_row(results_keys))
        last_results_keys = results_keys
    logger.info(misc.to_row([results[key] for key in results_keys]))
    records.append(copy.deepcopy(results))

    # update results to record
    results.update({"hparams": dict(hparams), "args": vars(args)})

    with open(epochs_path, "a") as f:
        f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

    checkpoint_vals = collections.defaultdict(lambda: [])

    writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
    writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

    # find best
    logger.info("---")
    records = Q(records)
    oracle_best = records.argmax("test_out")["test_in"]
    oracle_best_step = records.argmax("test_out")["step"]

    iid_best = records.argmax("train_out")["test_in"]
    iid_best_step = records.argmax("train_out")["step"]

    last = records[-1]["test_in"]
    last_step = records[-1]["step"]

    if hparams.indomain_test:
        # if test set exist, use test set for indomain results
        in_key = "train_inTE"
    else:
        in_key = "train_out"

    iid_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    ret = {
        "oracle": [oracle_best, oracle_best_step],
        "iid": [iid_best, iid_best_step],
        "last": [last, last_step],
        "last (inD)": [last_indomain, last_step],
        "iid (inD)": [iid_best_indomain, iid_best_step],
    }

    for k, acc in ret.items():
        logger.info(f"{k} = {acc[0]:.3%} (step={acc[1]})")

    return ret, records