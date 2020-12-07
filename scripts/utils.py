import yaml
import os
import shutil
import json
import jsonlines
from tqdm import tqdm
from random import random
import pandas as pd
import matplotlib.pyplot as plt


def down_sample_dataset(dataset_path: str, fraction: float):
    """
    Downsamples the training data of a dataset to a specified fraction
    """
    dest = os.path.join(dataset_path, "size_" + str(fraction))
    if not os.path.isdir(dest):
        os.makedirs(dest)
    shutil.copy(os.path.join(dataset_path, "train.json"), dest)
    shutil.copy(os.path.join(dataset_path, "val.json"), dest)

    new_train = []
    # open up train.json from original dataset
    with open(os.path.join(dest, "train.json"), 'r') as f:
        train = json.load(f)

    # randomly sample frames for reduced dataset
    for sample in tqdm(train):
        if random() <= fraction:
            new_train.append(sample)

    # replace old train.json
    with open(os.path.join(dest, "train.json"), 'w') as f:
        json.dump(new_train, f)

    # print out the number of files in new dataset
    print("New directory made %s with fraction %.01f" % (os.path.abspath(dest), fraction))
    print("Total number of train frames %.0f, actual fraction %.04f" %
          (len(new_train), len(new_train) / len(train)))


def compute_baseline(num_grid_segments: int, json_file: str):
    """
    Computes the baseline accuracy if all 0s are predicted
    """
    with open(json_file, 'r') as json_f:
        js = json.load(json_f)

    labels = []
    for i in js:
        labels.append(i['label'])

    df = pd.DataFrame(data=labels)
    print(df.head())
    print(df.describe())
    baseline = 1 - df.sum().sum() / (len(df) * num_grid_segments)
    print("baseline if predict all 0s = " + str(baseline))


def compute_params(config_path: str):
    """
    Convenient helper script for determining num_epochs and num_validation_steps from config file
    """
    with open(config_path, 'r') as yml_file:
        cfg = yaml.full_load(yml_file)

    with open(os.path.join(cfg['DATA_DIR_PATH'], 'train.json'), 'r') as json_file:
        train_json = json.load(json_file)

    num_epochs = int(round(cfg['SOLVER']['MAX_ITER'] / (len(train_json) / cfg['SOLVER']['IMS_PER_BATCH'])))
    one_epoch = int(round(cfg['SOLVER']['MAX_ITER'] / num_epochs))
    num_validation_steps = int(cfg['SOLVER']['MAX_ITER'] / cfg['TEST']['EVAL_PERIOD'])
    return {'num_epochs': num_epochs,
            'one_epoch': one_epoch,
            "num_validation_steps": num_validation_steps}


def plot_loss(output_dir):
    # FIXME
    """
    Plot loss from the 'metrics.json' file
    """
    data = []
    with jsonlines.open(os.path.join(output_dir, 'metrics.json')) as reader:
        for obj in reader:
            if 'total_loss' in obj.keys() and 'val_loss' in obj.keys():
                data.append(obj)

    df = pd.json_normalize(data)
    plt.plot(df['iteration'], df['total_loss'], label='Training')
    plt.plot(df['iteration'], df['val_loss'], label='Validation')
    plt.xlabel('iteration #, 1 epoch = %i iterations' %
               compute_params(os.path.join(output_dir, 'config.yaml'))['one_epoch'])
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Model loss")
    plt.savefig(os.path.join(output_dir, 'model_loss.png'))
