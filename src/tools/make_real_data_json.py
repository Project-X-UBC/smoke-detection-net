import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import argparse

RANDOM_SEED = 42
TEST_SIZE = .2
VALIDATION_SIZE = .25
DATA_FOLDER = os.path.abspath('../../data/full/frames_100_final')


def parse_args():
    parser = argparse.ArgumentParser(description="Make real dataset")
    parser.add_argument('--path', type=str, help="Path of the directory containing the image data, directory must "
                                                 "have 'frames' directory storing .png files and corresponding "
                                                 "'labels.json' file", required=False)
    parser.add_argument('--gridsize', type=int, help="Size of the label grid, directory must have corresponding label file e.g. "
                                                     "for grid size 16 must have labels_16.json", required=False)
    parser.add_argument('--mankind', type=str, help="Use if you want AI for Mankind. --mankind val or --mankind test, depending on which set you want it for.")
    args = parser.parse_args()
    if args.path is not None:
        args.path = os.path.abspath(args.path)
    return args


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


def unify_files(image_root):
    # In case there's already been a split, put the files back together in a single frames folder before splitting again
    if os.path.isdir(f'{image_root}/train'):
        os.makedirs(f'{image_root}/frames', exist_ok=True)
        for folder in ('train', 'test', 'val'):
            for filename in os.listdir(f'{image_root}/{folder}'):
                os.rename(f'{image_root}/{folder}/{filename}', f'{image_root}/frames/{filename}')


def split_files(dataset_dicts, image_root):
    # Split the files into train, test, and val again
    print('Now splitting the files into different folders...')
    if not os.path.isdir(f'{image_root}/train'):
        for phase in ('train', 'test', 'val'):
            os.mkdir(f'{image_root}/{phase}')
    for phase in dataset_dicts:
        for record in dataset_dicts[phase]:
            filename = record['file_name']
            os.rename(f'{image_root}/frames/{filename}', f'{image_root}/{phase}/{filename}')


def accumulate_real_data_json(image_root, grid_size):
    print('Accumulating the JSON...')
    labels_file = f'labels_{grid_size}.json' if grid_size else 'labels.json'
    json_filename = os.path.join(image_root, labels_file)
    with open(json_filename, 'rb') as f:
        labels = json.load(f)['labels']
    filenames = np.array(os.listdir(os.path.join(image_root, 'frames')))
    phases = train_test_val_split(filenames)
    dataset_dicts = {'train': [], 'test': [], 'val': []}
    missing = 0
    for phase in phases:
        for filename in tqdm(phases[phase]):
            if filename not in labels:
                missing += 1
                continue
            record = {
                'file_name': os.path.abspath(os.path.join(image_root, 'frames', str(filename))),
                'image_id' : int(np.where(filenames == filename)[0]),
                'label'    : labels[filename]
            }
            dataset_dicts[phase].append(record)
    print("total missing: " + str(missing))
    return dataset_dicts


def make_mankind_set(args):
    # Call this once you already have a split but want to use AI for Mankind's set instead of your val or test set
    json_filename = os.path.join(args.path, f'labels.json')
    with open(json_filename, 'rb') as f:
        labels = json.load(f)['labels']
    filenames = os.listdir(os.path.join(args.path, 'frames'))
    dataset = []
    for filename in filenames:
        record = {
            'file_name': os.path.abspath(os.path.join(args.path, 'frames', filename)),
            'image_id' : len(dataset),
            'label'    : labels[filename]
        }
        dataset.append(record)
    return dataset


def make_real_data_main(args):
    # to split the train/test/val datasets
    # Accumulate train
    # unify_files()
    if args.path is None:
        args.path = DATA_FOLDER
    if args.mankind in ('val', 'test'):
        mankind_set = make_mankind_set(args)
        with open(os.path.join(args.path, f"{args.mankind}.json"), "w") as w_obj:
            json.dump(mankind_set, w_obj)
        return
    dataset_dicts = accumulate_real_data_json(args.path, args.gridsize)
    print('Saving the JSON files...')
    with open(os.path.join(args.path, "train.json"), "w") as w_obj:
        json.dump(dataset_dicts['train'], w_obj)
    with open(os.path.join(args.path, "test.json"), "w") as w_obj:
        json.dump(dataset_dicts['test'], w_obj)
    with open(os.path.join(args.path, "val.json"), "w") as w_obj:
        json.dump(dataset_dicts['val'], w_obj)


if __name__ == "__main__":
    args = parse_args()
    make_real_data_main(args)
