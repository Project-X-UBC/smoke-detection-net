import os
import yaml
import json
from datetime import datetime
from src import train_net
import numpy as np
import pandas as pd


def calculate_pos_weights(data):
    # create df where rows are the samples and columns are the labels
    df = pd.DataFrame(pd.json_normalize(data)['label'].to_list())
    pos_counts = list(df.sum())
    pos_weights = np.ones(df.shape[1])
    neg_counts = [len(data) - pos_count for pos_count in pos_counts]
    for cdx, (pos_count, neg_count) in enumerate(zip(pos_counts, neg_counts)):
        pos_weights[cdx] = neg_count / (pos_count + 1e-5)

    return pos_weights


def compute_max_iter(num_imgs, batch_size, num_epochs):
    one_epoch = num_imgs / batch_size
    max_iter = one_epoch * num_epochs
    return int(max_iter)


def create_output_dir(output_dir):
    if not output_dir:
        creation_time = str(datetime.now().strftime('%m-%d_%H:%M/'))
        output_dir = os.path.join('./output', creation_time)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def update_config(params, config_path='src/config.yaml'):
    with open(config_path, 'r') as yml_file:
        cfg = yaml.full_load(yml_file)

    with open(os.path.join(params['data_dir'], 'train.json'), 'r') as json_file:
        train_json = json.load(json_file)

    pos_weight = calculate_pos_weights(train_json)

    cfg['SOLVER']['BASE_LR'] = params['base_lr']
    cfg['SOLVER']['IMS_PER_BATCH'] = params['batch_size']
    cfg['SOLVER']['MAX_ITER'] = compute_max_iter(len(train_json), params['batch_size'], params['num_epochs'])
    cfg['MODEL']['CLSNET']['INPUT_SIZE'] = params['input_size']
    cfg['MODEL']['MNET']['WIDTH_MULT'] = params['base_multiplier']
    cfg['OUTPUT_DIR'] = os.path.abspath(create_output_dir(params['output_dir']))
    cfg['DATA_DIR_PATH'] = os.path.abspath(params['data_dir'])
    cfg['MODEL']['POS_WEIGHT'] = [int(i) for i in pos_weight]  # yaml.dump spits out garbage if pos_weight are decimals

    # update config.yml file
    with open(config_path, 'w') as yml_file:
        yaml.dump(cfg, yml_file)


def set_params():
    """
    Sets the parameters of the pipeline
    """
    params = {
        # paths
        'data_dir': './data/mini',
        'output_dir': './output',  # default is ./output/$date_$time if left as empty string

        # hyperparameters
        'base_lr': 0.1,
        'batch_size': 64,
        'input_size': 224,  # resizes images to input_size x input_size e.g. 224x224
        'base_multiplier': 0.25,  # adjusts number of channels in each layer by this amount
        'num_epochs': 1  # total number of epochs, can be < 1
    }

    update_config(params)


if __name__ == '__main__':
    set_params()
    train_net.main()
