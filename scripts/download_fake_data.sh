#!/bin/bash

sudo apt install unzip

data_path=../data/synthetic
# download the fake data
gdown https://drive.google.com/uc?id=1V2xcLqkyJKqfAzUO7C5bUceyQtq0BjNe -O $data_path/fake_data.zip
unzip $data_path/fake_data.zip -d $data_path
mv $data_path/fake_data/* $data_path
rm -rf $data_path/fake_data
rm -rf $data_path/__MACOSX/

# make the json files for metadata
exec ./generate_metadata_jsons.sh data/synthetic
