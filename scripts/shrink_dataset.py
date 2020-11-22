import os
import shutil  
import sys
from tqdm import tqdm
from random import random
from os import listdir

sys.path.append('../src/tools')
from make_real_data_json import make_real_data_main, parse_args

DATASET_PATH = "../data/full/frames_100"
FRACTION = 0.2

dest = os.path.join(DATASET_PATH, "size_" + str(FRACTION))
shutil.copytree(DATASET_PATH, dest)
shutil.copy(os.path.join(DATASET_PATH, "labels.json"), dest)

frames = os.path.join(dest, "frames")
for f in tqdm(listdir(frames)):
    if random() > FRACTION:
        os.remove(os.path.join(frames, f))

# print out the number of files in new dataset
print("New directory made %s with fraction %.01f" % (os.path.abspath(dest), FRACTION))
print("Total number of files %.0f, actual fraction %.04f" %
            (len(listdir(frames)), len(listdir(frames))/len(listdir(os.path.join(DATASET_PATH, "frames")))))

# make the train/val splits
args = parse_args()
args.path = os.path.abspath(dest)
make_real_data_main(args)
