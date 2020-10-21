import os
import yaml
import json
from datetime import datetime
from src import train_net
import numpy as np
import pandas as pd
import torch
import random
from scripts.plot_loss import plot_loss


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    cfg['DATALOADER']['NUM_WORKERS'] = params['num_workers']
    cfg['EVAL_ONLY'] = params['eval_only_mode']

    if params['num_validation_steps'] != 0:
        cfg['TEST']['EVAL_PERIOD'] = int(cfg['SOLVER']['MAX_ITER'] / params['num_validation_steps'])

    if params['load_pretrained_weights'] or params['eval_only_mode']:
        cfg['MODEL']['WEIGHTS'] = os.path.abspath(params['model_weights'])
    else:
        cfg['MODEL']['WEIGHTS'] = ''

    # update config.yml file
    with open(config_path, 'w') as yml_file:
        yaml.dump(cfg, yml_file)


def set_params():
    """
    Sets the parameters of the pipeline
    """
    params = {
        # pipeline modes
        'eval_only_mode': False,  # evaluate model on test data, if true 'model_weights' param needs to be set
        'load_pretrained_weights': False,  # train model with pretrained model weights from file 'model_weights'

        # paths
        'data_dir': './data/synthetic',
        'output_dir': './output',  # default is ./output/$date_$time if left as empty string
        'model_weights': './output/model_final.pth',  # path to model weights file for training with pretrained weights

        # hyperparameters
        'base_lr': 0.01,
        'batch_size': 64,
        'input_size': 224,  # resizes images to input_size x input_size e.g. 224x224
        'base_multiplier': 0.25,  # adjusts number of channels in each layer by this amount
        'num_epochs': 1,  # total number of epochs, can be < 1
        'num_validation_steps': 1,  # number of evaluations on the validation set during training

        # misc
        'checkpoint_period': 5000,  # save a checkpoint after every this number of iterations
        'num_workers': 4,  # number of data loading threads
        'seed': 999  # seed so computations are deterministic
    }

    update_config(params)
    seed_all(params['seed'])

    return params


if __name__ == '__main__':
    p = set_params()
    train_net.main()
    if not p['eval_only_mode']:
        plot_loss(p['output_dir'])
