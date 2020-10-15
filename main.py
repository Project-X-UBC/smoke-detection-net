import os
import yaml
import json
from datetime import datetime
from src import train_net


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

    with open(params['train_json_file'], 'r') as json_file:
        train_json = json.load(json_file)
        num_imgs = len(train_json)

    print(cfg)
    cfg['SOLVER']['BASE_LR'] = params['base_lr']
    cfg['SOLVER']['IMS_PER_BATCH'] = params['batch_size']
    cfg['SOLVER']['MAX_ITER'] = compute_max_iter(num_imgs, params['batch_size'], params['num_epochs'])
    cfg['MODEL']['CLSNET']['INPUT_SIZE'] = params['input_size']
    cfg['MODEL']['MNET']['WIDTH_MULT'] = params['base_multiplier']
    cfg['OUTPUT_DIR'] = os.path.abspath(create_output_dir(params['output_dir']))

    # update config.yml file
    with open(config_path, 'w') as yml_file:
        yaml.dump(cfg, yml_file)


def set_params():
    """
    Sets the parameters of the pipeline
    """
    params = {
        # paths
        'data_dir': './data',
        'output_dir': './output',  # default is ./output/$date_$time if left as empty string
        'train_json_file': './data/train.json',
        'val_json_file': './data/val.json',

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
