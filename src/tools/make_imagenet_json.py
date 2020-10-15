"""
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 17:27:16
@FilePath       : /ImageCls.detectron2/tools/make_imagenet_json.py
@Description    :
"""

import re
import os
import argparse
import json
from tqdm import tqdm
import numpy as np

ARCHIVE_META = {
    'train': 'train_set',
    'val': 'test_set',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Make imagenet dataset d2-style")
    parser.add_argument('--root', type=str, help="ImageNet root directory")
    parser.add_argument('--save', type=str, help="Result saving directory")

    args = parser.parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    print(os.path.join(args.root, ARCHIVE_META['train']))
    assert os.path.exists(os.path.join(args.root, ARCHIVE_META['train']))
    assert os.path.exists(os.path.join(args.root, ARCHIVE_META['val']))

    return args


def get_multi_label_array(index):
    labels = np.zeros(16, dtype=int)
    if index != 0:
        labels[index-1] = 1
    return labels.tolist()


def accumulate_imagenet_json(image_root, phase):
    # accumulate infos
    classes = [i for i in range(0, 16)]

    json_file = os.path.join(image_root, 'labels_' + phase + '.txt')
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(tqdm(imgs_anns.keys())):
        # FIXME: fake data has images_test\\ or images_train\\ appended in front of keys
        key = re.sub("images_\w{4,5}\\\\", "", v)
        filename = os.path.join(image_root, phase + '_set', key)
        # height, width = cv2.imread(filename).shape[:2]

        record = {
            "file_name": os.path.abspath(filename),  # Using abs path, ignore image root, less flexibility
            "image_id": idx,  # fake data only has a max of 1 transformed grid segment
            "label": get_multi_label_array(imgs_anns[v]["index"]),
        }
        dataset_dicts.append(record)

    return dataset_dicts


def main(args):
    # TODO: use GroupKFold https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
    # to split the train/test/val datasets
    # Accumulate train
    dataset_dicts_train = accumulate_imagenet_json(args.root, phase='train')
    # Accumulate val
    dataset_dicts_val = accumulate_imagenet_json(args.root, phase='test')
    # Save
    # TODO: add arg for train, val, test json file names
    with open(os.path.join(args.save, "train.json"), "w") as w_obj:
        json.dump(dataset_dicts_train, w_obj)
    with open(os.path.join(args.save, "val.json"), "w") as w_obj:
        json.dump(dataset_dicts_val, w_obj)


if __name__ == "__main__":
    args = parse_args()
    main(args)
