""" Convenient helper script for determining num_epochs and num_validation_steps from config file"""
import yaml
import json
import os
import argparse


def compute_params(config_path):
    with open(config_path, 'r') as yml_file:
        cfg = yaml.full_load(yml_file)

    with open(os.path.join(cfg['DATA_DIR_PATH'], 'train.json'), 'r') as json_file:
        train_json = json.load(json_file)

    num_epochs = int(round(cfg['SOLVER']['MAX_ITER']/(len(train_json)/cfg['SOLVER']['IMS_PER_BATCH'])))
    one_epoch = int(round(cfg['SOLVER']['MAX_ITER']/num_epochs))
    num_validation_steps = int(cfg['SOLVER']['MAX_ITER']/cfg['TEST']['EVAL_PERIOD'])
    return {'num_epochs': num_epochs,
            'one_epoch': one_epoch,
            "num_validation_steps": num_validation_steps}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False)

    args = parser.parse_args()
    if args.config is not None:
        config = args.config
    else:
        config = '../src/config.yaml'  # can specify here the path of the config or via argument passing

    params = compute_params(config)
    print("num_epochs: %i" % params['num_epochs'])
    print("number of iterations per epoch: %i" % params['one_epoch'])
    print("num_validation_steps: %i" % params['num_validation_steps'])
