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
from sklearn.model_selection import GroupShuffleSplit

RANDOM_SEED = 42
TEST_SIZE = .2
VALIDATION_SIZE = .25
DATA_FOLDER = '../../data/full'
ARCHIVE_META = {
    'train': 'train_set',
    'val': 'test_set',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Make imagenet dataset d2-style")
    parser.add_argument('--path', type=str, help="Path of the directory containing the image data")
    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.path, ARCHIVE_META['train']))
    assert os.path.exists(os.path.join(args.path, ARCHIVE_META['val']))

    return args


def get_multi_label_array(index):
    labels = np.zeros(16, dtype=int)
    if index != 0:
        labels[index-1] = 1
    return labels.tolist()


def train_test_val_split(filenames):
    videonames = np.array([filename.split('_frame')[0] for filename in filenames])
    gs_split_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    gs_split_val = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = gs_split_test.split(filenames, groups=videonames)
    train_idx, val_idx = gs_split_val.split(filenames[train_idx], groups=videonames)
    return {'train': train_idx, 'test': test_idx, 'val': val_idx}


def unify_files():
    # In case there's already been a split, put the files back together in a single frames folder before splitting again
    if os.path.isdir(f'{DATA_FOLDER}/train'):
        os.mkdir(f'{DATA_FOLDER}/frames')
        for folder in ('train', 'test', 'val'):
            for filename in os.listdir(f'{DATA_FOLDER}/{folder}'):
                os.rename(f'{DATA_FOLDER}/{folder}/{filename}', f'{DATA_FOLDER}/frames/{filename}')


def split_files(dataset_dicts):
    # Split the files into train, test, and val again
    if not os.path.isdir(f'{DATA_FOLDER}/train'):
        for phase in ('train', 'test', 'val'):
            os.mkdir(f'{DATA_FOLDER}/{phase}')
    for phase in dataset_dicts:
        for record in dataset_dicts[phase]:
            filename = record['file_name']
            os.rename(f'{DATA_FOLDER}/frames/{filename}', f'{DATA_FOLDER}/{phase}/{filename}')
    os.rmdir(f'{DATA_FOLDER}/frames')


def accumulate_real_data_json(image_root):
    json_filename = os.path.join(image_root, 'labels.json')
    with open(json_filename, 'rb') as f:
        labels = json.load(f)['labels']
    filenames = np.array(os.listdir(f'{image_root}/frames'))
    idxs = train_test_val_split(filenames)
    dataset_dicts = {'train': [], 'test': [], 'val': []}
    for phase in idxs:
        for idx in tqdm(idxs[phase]):
            record = {
                'file_name': filenames[idx],
                'image_id' : idx,
                'label'    : labels[filenames[idx]]
            }
            dataset_dicts[phase].append(record)
    return dataset_dicts


def make_real_data_main():
    # to split the train/test/val datasets
    # Accumulate train
    unify_files()
    dataset_dicts = accumulate_real_data_json(DATA_FOLDER)
    split_files(dataset_dicts)
    # Accumulate val
    # Save
    # TODO: add arg for train, val, test json file names
    with open(os.path.join(DATA_FOLDER, "train.json"), "w") as w_obj:
        json.dump(dataset_dicts['train'], w_obj)
    with open(os.path.join(DATA_FOLDER, "test.json"), "w") as w_obj:
        json.dump(dataset_dicts['test'], w_obj)
    with open(os.path.join(DATA_FOLDER, "val.json"), "w") as w_obj:
        json.dump(dataset_dicts['val'], w_obj)


if __name__ == "__main__":
    make_real_data_main()
