import os
import itertools
import json
import logging
from collections import OrderedDict
import torch
from torch import nn
from fvcore.common.file_io import PathManager
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
                          "pred": output["pred"].to(self._cpu_device)}
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

        running_loss = 0.0
        criterion = nn.BCELoss()
        target = []
        pred = []
        file_path = []
        image_id = []
        for p in self._predictions:
            target.append(p["label"])
            pred.append(p["pred"])
            file_path.append(p["file_path"])
            image_id.append(p["image_id"])

            loss = criterion(p["pred"], torch.as_tensor(p["label"], dtype=torch.float))
            running_loss += loss.item()

        val_loss = running_loss / len(pred)
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

        # reshape arrays to single dimension
        pred = pred.view(1, -1).squeeze()
        target = target.view(1, -1).squeeze()

        # compute ROC area-under-curve
        fpr, tpr, _ = roc_curve(target, pred)
        roc_auc = auc(fpr, tpr)

        # save ROC curve results as a plot
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        path = os.path.join(self._output_dir, "roc_curve.png")
        plt.savefig(path)
        plt.close()
        self._logger.info("Saved ROC plot with file path: %s" % os.path.abspath(path))

        # get number of correct samples with threshold == 0.5
        pred = pred.round()
        correct = pred.eq(target)

        # compute Matthews correlation coefficient (MCC)
        mcc = matthews_corrcoef(target, pred)

        accuracy = correct.sum().true_divide(torch.tensor(correct.size(0)))
        confusion_vector = (pred // target)
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

        self._logger.info("True pos %.0f" % true_pos)
        self._logger.info("False pos %.0f" % false_pos)
        self._logger.info("True neg %.0f" % true_neg)
        self._logger.info("False neg %.0f" % false_neg)

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

        f1_score = float('Nan')
        if not (np.isnan(precision) or np.isnan(recall)):
            f1_score = 2 * precision * recall / (precision + recall)

        self._logger.info("Accuracy %.4f" % accuracy)
        self._logger.info("Recall %.4f" % recall)
        self._logger.info("Precision %.4f" % precision)
        self._logger.info("F1-score %.4f" % f1_score)
        self._logger.info("MCC %.4f" % mcc)
        self._logger.info("ROC AUC %.4f" % roc_auc)
        self._logger.info("Validation loss %.4f" % val_loss)

        result = OrderedDict(metrics={"true_pos": true_pos, "false_pos": false_pos,
                                      "true_neg": true_neg, "false_neg": false_neg,
                                      "accuracy": accuracy.item(), "recall": recall,
                                      "precision": precision, "f1_score": f1_score,
                                      "roc_auc": roc_auc, "val_loss": val_loss, "mcc": mcc})
        return result
