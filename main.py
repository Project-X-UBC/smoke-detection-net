import os
import yaml
import json
from datetime import datetime
from src import custom_train_loop
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
    torch.backends.cudnn.deterministic = True  # may result in a slowdown if set to True


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


def update_config(params, config_file='src/config.yaml'):
    with open(config_file, 'r') as yml_file:
        cfg = yaml.full_load(yml_file)

    with open(os.path.join(params['data_dir'], 'train.json'), 'r') as json_file:
        train_json = json.load(json_file)

    pos_weight = calculate_pos_weights(train_json)

    cfg['SOLVER']['BASE_LR'] = params['base_lr']
    cfg['SOLVER']['IMS_PER_BATCH'] = params['batch_size']
    cfg['SOLVER']['MAX_ITER'] = compute_max_iter(len(train_json), params['batch_size'], params['num_epochs'])
    cfg['SOLVER']['CHECKPOINT_PERIOD'] = params['checkpoint_period']
    cfg['SOLVER']['STEPS'] = params['decrease_lr_iter']
    cfg['SOLVER']['GAMMA'] = params['gamma']
    cfg['MODEL']['CLSNET']['INPUT_SIZE'] = params['input_size']
    cfg['MODEL']['CLSNET']['NUM_CLASSES'] = params['num_classes']
    cfg['MODEL']['MNET']['WIDTH_MULT'] = params['base_multiplier']
    cfg['MODEL']['BACKBONE']['NAME'] = params['backbone']
    cfg['MODEL']['BACKBONE']['FREEZE_AT'] = params['freeze_at']
    cfg['OUTPUT_DIR'] = os.path.abspath(create_output_dir(params['output_dir']))
    cfg['DATA_DIR_PATH'] = os.path.abspath(params['data_dir'])
    cfg['MODEL']['POS_WEIGHT'] = [round(i) for i in pos_weight]  # yaml.dump spits out garbage if pos_weight are decimals
    cfg['DATALOADER']['NUM_WORKERS'] = params['num_workers']
    cfg['EVAL_ONLY'] = params['eval_only_mode']
    cfg['SEED'] = params['seed']
    cfg['EARLY_STOPPING']['ENABLE'] = params['early_stopping']
    cfg['EARLY_STOPPING']['MONITOR'] = params['early_stopping_monitor']
    cfg['EARLY_STOPPING']['PATIENCE'] = params['patience']
    cfg['EARLY_STOPPING']['MODE'] = params['early_stopping_mode']

    if params['num_validation_steps'] != 0:
        cfg['TEST']['EVAL_PERIOD'] = int(cfg['SOLVER']['MAX_ITER'] / params['num_validation_steps'])

    if params['load_pretrained_weights'] or params['eval_only_mode']:
        if 'detectron2://' in params['model_weights']:
            cfg['MODEL']['WEIGHTS'] = params['model_weights']
        else:
            cfg['MODEL']['WEIGHTS'] = os.path.abspath(params['model_weights'])
    else:
        cfg['MODEL']['WEIGHTS'] = ''

    # update config.yml file
    with open(config_file, 'w') as yml_file:
        yaml.dump(cfg, yml_file)


def set_params():
    """
    Sets the parameters of the pipeline
    """
    params = {
        # configuration file
        'config': 'src/config_resnext.yaml',  # 'src/config_resnet.yaml' to load pretrained resnet model

        # architecture
        'backbone': 'build_resnet_cls_backbone',  # select model backbone
                                              # custom backbones are in src/imgcls/modeling/backbone

        # compute settings
        'num_gpus': 1,  # number of gpus, can check with `nvidia-smi`

        # pipeline modes
        'eval_only_mode': False,  # evaluate model on test data, if true 'model_weights' param needs to be set
        'resume': False,  # resume training from last checkpoint in 'output_dir', useful when training was interrupted
        'load_pretrained_weights': True,  # train model with pretrained model weights from file 'model_weights'
        'early_stopping': True,  # option to early stop model training if a certain condition is met
        'early_stopping_monitor': 'f1_score',  # metric to monitor for early stopping
                                               # current options accuracy, recall, precision, f1_score, roc_auc, val_loss
        'early_stopping_mode': 'max',  # the objective of the 'early_stopping_monitor' metric, e.g. 'min' for loss

        # paths
        'data_dir': './data/full/frames_100/',
        'output_dir': './output/resnext-101-proper-lr-test',  # default is ./output/$date_$time if left as empty string
        'model_weights': './pretrained_models/X-101-32x8d.pkl',  # path to model weights file for training with pretrained weights
                                                      # resnet-50 pretrained weights 'detectron2://ImageNetPretrained/MSRA/R-50.pkl'

        # hyperparameters
        'base_lr': 0.0001,
        'batch_size': 16,
        'input_size': 224,  # resizes images to input_size x input_size e.g. 224x224
        'base_multiplier': 0.5,  # adjusts number of channels in each layer by this amount for mobilenetv1
        'num_classes': 16,  # specifies the number of classes + number of nodes in final model layer
        'freeze_at': 2,  # freeze layers of network
        'decrease_lr_iter': (2000, 20000),  # the iteration number to decrease learning rate by gamma
        'gamma': 0.01,  # factor to decrease lr by

        # misc
        'patience': 10,  # number of val steps where no improvement is made before triggering early stopping
        'num_epochs': 50,  # total number of epochs, can be < 1
        'num_validation_steps': 200,  # number of evaluations on the validation set during training
        'checkpoint_period': 10000,  # save a checkpoint after every this number of iterations
        'num_workers': 4,  # number of data loading threads
        'seed': 999  # seed so computations are deterministic
    }

    update_config(params, config_file=params['config'])
    seed_all(params['seed'])

    return params


if __name__ == '__main__':
    p = set_params()
    custom_train_loop.main(num_gpus=p['num_gpus'],
                           config_file=p['config'],
                           resume=p['resume'])
    if not p['eval_only_mode']:
        plot_loss(p['output_dir'])
