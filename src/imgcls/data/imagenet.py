"""
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 17:02:07
@FilePath       : /ImageCls.detectron2/imgcls/data/imagenet.py
@Description    :
"""

import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog


def register_imagenet_instances(name, metadata, json_file):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file

    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: json.load(open(json_file)))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(json_file=json_file, evaluator_type="imagenet", **metadata)


# TODO could improve this by removing hardcoded stuff
# Update info
_root = os.getenv("DETECTRON2_DATASETS", "../data")
imagenet_train_annotation_file = os.path.join(_root, "train.json")
imagenet_val_annotation_file = os.path.join(_root, "val.json")

# Register into DatasetCatalog
register_imagenet_instances("smokenet_train", {}, imagenet_train_annotation_file)
register_imagenet_instances("smokenet_val", {}, imagenet_val_annotation_file)
# TODO: need to register a third for test!
