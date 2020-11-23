import os
import shutil  
import sys
import json
from tqdm import tqdm
from random import random
from os import listdir

sys.path.append('../src/tools')
from make_real_data_json import make_real_data_main, parse_args

DATASET_PATH = "../data/full/frames_100"
FRACTION = 0.1

dest = os.path.join(DATASET_PATH, "size_" + str(FRACTION))
if not os.path.isdir(dest):
    os.makedirs(dest)
shutil.copy(os.path.join(DATASET_PATH, "train.json"), dest)
shutil.copy(os.path.join(DATASET_PATH, "val.json"), dest)

new_train = []
# open up train.json from original dataset
with open(os.path.join(dest, "train.json"), 'r') as f:
    train = json.load(f)

# randomly sample frames for reduced dataset
for sample in tqdm(train):
    if random() <= FRACTION:
        new_train.append(sample)

# replace old train.json
with open(os.path.join(dest, "train.json"), 'w') as f:
    json.dump(new_train, f)

# print out the number of files in new dataset
print("New directory made %s with fraction %.01f" % (os.path.abspath(dest), FRACTION))
print("Total number of train frames %.0f, actual fraction %.04f" %
            (len(new_train), len(new_train)/len(train)))
