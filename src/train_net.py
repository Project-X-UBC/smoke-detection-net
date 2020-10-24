# original file from https://github.com/facebookresearch/detectron2/blob/master/tools/plain_train_net.py
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    verify_results,
)

from src.imgcls.config import get_cfg
import src.imgcls.modeling  # need this import to initialize modeling package
from src.imgcls.data import DatasetMapper
from src.imgcls.data.imagenet import register_imagenet_instances
from src.imgcls.evaluation.imagenet_evaluation import ImageNetEvaluator
from src.imgcls.evaluation.loss_eval_hook import LossEvalHook


class Trainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ImageNetEvaluator(dataset_name, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model,
                                      self.build_test_loader(self.cfg, 'smokenet_val')))
        return hooks


def setup(args):
    """
    Create configs, perform basic setups, and register dataset instances
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if cfg.EVAL_ONLY:
        cfg.DATASETS.TEST = ("smokenet_test", )
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

    if cfg.EVAL_ONLY:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)  # TEST.EXPECTED_RESULTS is an empty list
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


def main(num_gpus=1):
    args = default_argument_parser().parse_args()
    args.config_file = "src/config.yaml"
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
