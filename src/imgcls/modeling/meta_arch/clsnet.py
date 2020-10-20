"""
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 16:20:01
@FilePath       : /ImageCls.detectron2/imgcls/modeling/meta_arch/clsnet.py
@Description    :
"""

import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from torch import nn
import logging


@META_ARCH_REGISTRY.register()
class ClsNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.MODEL.CLSNET.NUM_CLASSES
        self.in_features = cfg.MODEL.CLSNET.IN_FEATURES
        self.bottom_up = build_backbone(cfg)

        self._logger = logging.getLogger("detectron2.loss")
        pos_weight = torch.as_tensor(cfg.MODEL.POS_WEIGHT, dtype=torch.float)
        self._logger.info('pos_weight: ' + str(pos_weight))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)  # Do not need size_divisibility
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.bottom_up(images.tensor)
        features = [features[f] for f in self.in_features]

        if self.training:
            gt_labels = [x['label'] for x in batched_inputs]
            gt_labels = torch.as_tensor(gt_labels, dtype=torch.float).to(self.device)
            losses = self.losses(gt_labels, features)
            return losses
        else:
            results = nn.Sigmoid()(features[0])
            processed_results = []
            for results_per_image in results:
                processed_results.append({"pred": results_per_image})
            return processed_results

    def losses(self, gt_labels, features):
        loss = self.criterion(features[0], gt_labels)
        self._logger.info('loss: %.4f' % loss) # FIXME weirdest bug
        return {"loss_cls": loss}

