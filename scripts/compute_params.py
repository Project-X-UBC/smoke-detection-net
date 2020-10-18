""" Convenient helper script for determining num_epochs and num_validation_steps from config.yaml"""
import yaml
import json
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False)

    args = parser.parse_args()
    if args.config is not None:
        config = args.config
    else:
        config = '../src/config.yaml'  # can specify here the path of the config or via argument passing

    with open(config, 'r') as yml_file:
        cfg = yaml.full_load(yml_file)

    with open(os.path.join(cfg['DATA_DIR_PATH'], 'train.json'), 'r') as json_file:
        train_json = json.load(json_file)

    print("num_epochs: %i" % int(round(cfg['SOLVER']['MAX_ITER']/(len(train_json)/cfg['SOLVER']['IMS_PER_BATCH']))))
    print("num_validation_steps: %i" % int(cfg['SOLVER']['MAX_ITER']/cfg['TEST']['EVAL_PERIOD']))

