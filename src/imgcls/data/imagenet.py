import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
import logging


def register_imagenet_instances(name, metadata, json_file):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    logger = logging.getLogger("detectron2.data.imagenet")

    # check if dataset already exists
    if name not in DatasetCatalog.list():
        logger.info("Adding dataset '%s' to dataset catalog" % name)

        # 1. register a function which returns dicts
        DatasetCatalog.register(name, lambda: json.load(open(json_file)))

        # 2. Optionally, add metadata about this dataset,
        # since they might be useful in evaluation, visualization or logging
        MetadataCatalog.get(name).set(json_file=json_file, evaluator_type="imagenet", **metadata)

    else:
        logger.warning("Dataset with name '%s' is already registered, ignoring...")

