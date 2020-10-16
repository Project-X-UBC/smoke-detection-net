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
