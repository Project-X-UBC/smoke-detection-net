#!/bin/bash

sudo apt install unzip

# download the fake data
gdown https://drive.google.com/uc?id=1V2xcLqkyJKqfAzUO7C5bUceyQtq0BjNe -O data/synthetic/fake_data.zip
unzip data/synthetic/fake_data.zip -d data/synthetic
mv fake_data/* data/synthetic
rm -rf fake_data
rm -rf data/synthetic/__MACOSX/

# make the json files for metadata
exec ./generate_metadata_jsons.sh data/synthetic
