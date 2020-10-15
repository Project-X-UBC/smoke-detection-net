#!/bin/bash

# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

# setup conda env
source ~/miniconda/bin/activate
conda init
conda create -y --name projectx python=3.6
echo "conda activate projectx" >> ~/.bashrc
conda activate projectx

# install detectron2 + dependencies
pip install pyyaml==5.1 pycocotools==2.0.1
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -y -c conda-forge opencv
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html
# make sure installation contains no errors
python -m detectron2.utils.collect_env

# install rest of dependencies
conda env update --name projectx --file env.yml

# download the fake data
gdown https://drive.google.com/uc?id=1V2xcLqkyJKqfAzUO7C5bUceyQtq0BjNe -O data/fake_data.zip
sudo apt install unzip
unzip data/fake_data.zip

# make the json files for metadata
cd src/tools
python make_imagenet_json.py --root ../../data/fake_data --save ../../data
exit 0
