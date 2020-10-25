#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# original code from https://github.com/facebookresearch/detectron2/blob/master/tools/plain_train_net.py
import logging
import os
from collections import OrderedDict
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from src.imgcls.config import get_cfg
import src.imgcls.modeling  # need this import to initialize modeling package
from src.imgcls.data import DatasetMapper
from src.imgcls.data.imagenet import register_imagenet_instances
from src.imgcls.evaluation.imagenet_evaluation import ImageNetEvaluator
from src.imgcls.evaluation.loss_eval_hook import LossEvalHook

logger = logging.getLogger("detectron2")


def build_test_loader(cfg, dataset_name):
    return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))


def build_train_loader(cfg):
    return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return ImageNetEvaluator(dataset_name, True, output_folder)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)  # change directory name
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]

    return results


def compute_val_loss(cfg, model):
    data_loader = build_test_loader(cfg, cfg.DATASETS.TEST[0])  # only compute loss on first test dataset
    running_loss = 0.0
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            result = model(data)
            target = torch.as_tensor(data[0]['label'], dtype=torch.float).to(model.device)
            loss = criterion(result[0]["pred"], target)
            running_loss += loss.item()
    model.train()
    return running_loss / len(data_loader)


def get_es_result(mode, current, best_so_far):
    """Returns true if monitored metric has been improved"""
    if mode == 'max':
        return current > best_so_far
    elif mode == 'min':
        return current < best_so_far


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # init best monitor metric
    best_monitor_metric = None

    # init early stopping count
    es_count = 0

    # init val loss
    val_loss = compute_val_loss(cfg, model)

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.step()

            _, losses, losses_reduced = get_loss(data, model)
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, validation_loss=val_loss)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                results = do_test(cfg, model)
                storage.put_scalars(**results)
                val_loss = compute_val_loss(cfg, model)

                if cfg.EARLY_STOPPING.ENABLE:
                    curr = None
                    if cfg.EARLY_STOPPING.MONITOR in results['metrics'].keys():
                        curr = results['metrics'][cfg.EARLY_STOPPING.MONITOR]
                    elif cfg.EARLY_STOPPING.MONITOR == 'validation_loss':
                        curr = val_loss

                    if curr is None:
                        logger.warning("Early stopping enabled but cannot find metric: %s" %
                                       cfg.EARLY_STOPPING.MONITOR)
                    elif best_monitor_metric is None:
                        best_monitor_metric = curr
                    elif get_es_result(cfg.EARLY_STOPPING.MODE,
                                       curr, best_monitor_metric):
                        best_monitor_metric = curr
                        es_count = 0
                        logger.info("Best metric %s improved to %0.4f" %
                                    (cfg.EARLY_STOPPING.MONITOR, curr))
                        # update best model
                        periodic_checkpointer.save(name="model_best",
                                                   **{**results['metrics'], 'validation_loss': val_loss})
                    else:
                        logger.info("Early stopping metric %s did not improve, current %.04f, best %.04f" %
                                    (cfg.EARLY_STOPPING.MONITOR, curr, best_monitor_metric))
                        es_count += 1

                comm.synchronize()

            if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            if es_count >= cfg.EARLY_STOPPING.PATIENCE:
                logger.info("Early stopping triggered, metric %s has not improved for %s validation steps" %
                            (cfg.EARLY_STOPPING.MONITOR, cfg.EARLY_STOPPING.PATIENCE))
                break


def get_loss(data, model):
    loss_dict = model(data)
    losses = sum(loss_dict.values())
    assert torch.isfinite(losses).all(), loss_dict
    loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
    return loss_dict_reduced, losses, losses_reduced


def setup(args):
    """
    Create configs, perform basic setups, and register dataset instances
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if cfg.EVAL_ONLY:
        cfg.DATASETS.TEST = ("smokenet_test",)
        test_annotation_file = os.path.join(cfg.DATA_DIR_PATH, "test.json")
        # Register into DatasetCatalog
        register_imagenet_instances(cfg.DATASETS.TEST[0], {}, test_annotation_file)

    else:
        train_annotation_file = os.path.join(cfg.DATA_DIR_PATH, "train.json")
        val_annotation_file = os.path.join(cfg.DATA_DIR_PATH, "val.json")
        # Register into DatasetCatalog
        register_imagenet_instances("smokenet_train", {}, train_annotation_file)
        register_imagenet_instances("smokenet_val", {}, val_annotation_file)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def run(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if cfg.EVAL_ONLY:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


def main(num_gpus=1, resume=False):
    args = default_argument_parser().parse_args()
    args.config_file = "src/config.yaml"
    args.resume = resume
    print("Command Line Args: ", args)
    # launch multi-gpu or distributed training
    launch(
        run,
        num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url="auto",
        args=(args,),
    )


if __name__ == "__main__":
    main()
