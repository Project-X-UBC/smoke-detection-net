#!/bin/bash

# modify the name of the data directory which should be inside /data
DATA_DIR=fake_data

cd src/tools
python make_imagenet_json.py --root ../../data/$DATA_DIR --save ../../data