#!/bin/bash

# modify the name of the data directory which should be inside /data
DATA_DIR=$1

if [ -z "$DATA_DIR" ]
then
      echo "\$DATA_DIR is empty, pass in 1 argument when calling script e.g. './generate_metadata_jsons.sh data/mini'"
      exit 1
fi

cd ../src/tools
python make_imagenet_json.py --path ../../"$DATA_DIR"
