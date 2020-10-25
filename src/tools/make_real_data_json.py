import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

RANDOM_SEED = 42
TEST_SIZE = .2
VALIDATION_SIZE = .25
DATA_FOLDER = '../../data/full'


def get_multi_label_array(index):
    labels = np.zeros(16, dtype=int)
    if index != 0:
        labels[index-1] = 1
    return labels.tolist()


def train_test_val_split(filenames):
    videonames = np.array([filename.split('_frame')[0] for filename in filenames])
    gs_split_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    gs_split_val = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gs_split_test.split(filenames, groups=videonames))
    train, test = filenames[train_idx], filenames[test_idx]
    train_idx, val_idx = next(gs_split_val.split(train, groups=videonames[train_idx]))
    train, val = train[train_idx], train[val_idx]
    assert not np.in1d(train, test).any() and not np.in1d(train, val).any()
    return {'train': train, 'test': test, 'val': val}


def unify_files():
    # In case there's already been a split, put the files back together in a single frames folder before splitting again
    if os.path.isdir(f'{DATA_FOLDER}/train'):
        os.mkdir(f'{DATA_FOLDER}/frames')
        for folder in ('train', 'test', 'val'):
            for filename in os.listdir(f'{DATA_FOLDER}/{folder}'):
                os.rename(f'{DATA_FOLDER}/{folder}/{filename}', f'{DATA_FOLDER}/frames/{filename}')


def split_files(dataset_dicts):
    # Split the files into train, test, and val again
    print('Now splitting the files into different folders...')
    if not os.path.isdir(f'{DATA_FOLDER}/train'):
        for phase in ('train', 'test', 'val'):
            os.mkdir(f'{DATA_FOLDER}/{phase}')
    for phase in dataset_dicts:
        for record in dataset_dicts[phase]:
            filename = record['file_name']
            os.rename(f'{DATA_FOLDER}/frames/{filename}', f'{DATA_FOLDER}/{phase}/{filename}')


def accumulate_real_data_json(image_root):
    print('Accumulating the JSON...')
    json_filename = os.path.join(image_root, 'labels.json')
    with open(json_filename, 'rb') as f:
        labels = json.load(f)['labels']
    filenames = np.array(os.listdir(f'{image_root}/frames'))
    phases = train_test_val_split(filenames)
    dataset_dicts = {'train': [], 'test': [], 'val': []}
    for phase in phases:
        for filename in tqdm(phases[phase]):
            if filename not in labels:
                continue
            record = {
                'file_name': os.path.abspath(os.path.join(DATA_FOLDER, phase, str(filename))),
                'image_id' : int(np.where(filenames == filename)[0]),
                'label'    : labels[filename]
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
    print('Saving the JSON files...')
    with open(os.path.join(DATA_FOLDER, "train.json"), "w") as w_obj:
        json.dump(dataset_dicts['train'], w_obj)
    with open(os.path.join(DATA_FOLDER, "test.json"), "w") as w_obj:
        json.dump(dataset_dicts['test'], w_obj)
    with open(os.path.join(DATA_FOLDER, "val.json"), "w") as w_obj:
        json.dump(dataset_dicts['val'], w_obj)


if __name__ == "__main__":
    make_real_data_main()
