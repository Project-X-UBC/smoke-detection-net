"""
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 19:42:05
@FilePath       : /ImageCls.detectron2/imgcls/evaluation/imagenet_evaluation.py
@Description    :
"""

import os
import itertools
import json
import logging
from collections import OrderedDict
import torch
from torch import nn
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
import pandas as pd


class ImageNetEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("detectron2.evaluation.imagenet_evaluation")

        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._gt = json.load(open(json_file))

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"file_path": input["file_name"], "image_id": input["image_id"], "label": input["label"],
                          "pred": nn.Sigmoid()(output["pred"].to(self._cpu_device))}
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning(
                "[ImageNetEvaluator] Did not receive valid predictions.")
            return {}

        target = []
        pred = []
        file_path = []
        image_id = []
        for p in self._predictions:
            target.append(p["label"])
            pred.append(p["pred"])
            file_path.append(p["file_path"])
            image_id.append(p["image_id"])

        pred = torch.stack(pred, dim=0)
        target = torch.as_tensor(target, dtype=pred.dtype)

        # save results as csv in output directory
        data = {"file_path": file_path, "image_id": image_id, "label": list(target.numpy()),
                "pred": list(pred.round().numpy()), "pred_raw": list(pred.numpy())}
        df = pd.DataFrame(data=data)
        path = os.path.join(self._output_dir, "results.csv")
        os.makedirs(self._output_dir, exist_ok=True)
        df.to_csv(path, index=False)
        self._logger.info("Saved results with file path: %s" % os.path.abspath(path))

        pred = pred.round().view(1, -1)
        target = target.view(1, -1)
        correct = pred.eq(target).squeeze()

        # FIXME naive metrics
        accuracy = correct.sum().true_divide(torch.tensor(correct.size(0)))
        confusion_vector = (pred // target).squeeze()
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        true_pos = torch.sum(confusion_vector == 1).item()
        false_pos = torch.sum(confusion_vector == float('inf')).item()
        true_neg = torch.sum(torch.isnan(confusion_vector)).item()
        false_neg = torch.sum(confusion_vector == 0).item()

        assert accuracy == ((true_pos + true_neg) / (true_pos + false_pos + true_neg + false_neg))
        # handle possibility of denominator being 0
        try:
            recall = true_pos / (true_pos + false_neg)
        except ZeroDivisionError:
            recall = float('NaN')
        try:
            precision = true_pos / (true_pos + false_pos)
        except ZeroDivisionError:
            precision = float('Nan')

        self._logger.info("Accuracy %.4f: " % accuracy)
        self._logger.info("Recall %.4f: " % recall)
        self._logger.info("Precision %.4f: " % precision)

        result = OrderedDict(metrics={"accuracy": accuracy.item(), "recall": recall, "precision": precision})
        return result

