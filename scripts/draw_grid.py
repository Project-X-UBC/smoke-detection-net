# Picks a random frame and draws its corresponding boxes and grid
# You can also give it a specific file name


import random
import json
import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from os import listdir
import pandas as pd

GRID_SIZE = 4
DATA_FOLDER = '../data/full'


def get_sample_filename():
    filenames = listdir(DATA_FOLDER)
    filenames.remove('.gitignore')
    filename = random.choice(filenames)
    return filename


def get_label_and_boxes(filename):
    with open('../data/labels.json', 'rb') as f:
        labels_json = json.load(f)
    label = np.array(labels_json['labels'][filename]).reshape((GRID_SIZE,GRID_SIZE))
    boxes = labels_json['cvat_boxes'][filename] if filename in labels_json['cvat_boxes'] else []
    return label, boxes


def draw_grid(img, grid_vector, color=(255, 0, 0)):
    height, width, _ = img.shape
    grid_w, grid_h = width//GRID_SIZE, height//GRID_SIZE
    img = img.copy()
    # first draw grid lines
    for i in range(GRID_SIZE):
        x = i*grid_w
        y = i*grid_h
        for j in range(height):
            img[j, x, :] = color
        for j in range(width):
            img[y, j, :] = color
    # now highlight smoke areas
    highlight = np.vectorize(lambda x, s: min(255, x+s//10))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid_vector[j][i] == 1:
                for k in range(i*grid_w, (i+1)*grid_w):
                    for l in range(j*grid_h, (j+1)*grid_h):
                        img[l, k, :] = highlight(img[l, k, :], color)
    return img


def draw_grid_and_box(filename, label, boxes=[]):
    img = cv2.imread(DATA_FOLDER + '/' + filename)
    for box in boxes:
        tl = (int(box[0][0]), int(box[0][1]))
        br = (int(box[1][0]), int(box[1][1]))
        img = cv2.rectangle(img, tl, br, (0, 0, 255), 2)
    img = draw_grid(img, label, color=(255, 0, 0))
    return img


def get_pred_and_true(filename):
    df = pd.read_csv('../data/results.csv')
    idx = df[df['file_path'] == filename].index[0]
    pred = np.array([int(float(val)) for val in df.iloc[idx]['pred'].strip('[]').split(' ')]).reshape((GRID_SIZE, GRID_SIZE))
    true = np.array([int(float(val)) for val in df.iloc[idx]['label'].strip('[]').split(' ')]).reshape((GRID_SIZE, GRID_SIZE))
    return pred, true


def draw_pred_vs_true(filename, pred, true):
    img = cv2.imread(DATA_FOLDER + '/' + filename)
    exclusive = np.vectorize(lambda x, y: 1 if x == 1 and y == 0 else 0)
    pred_only, true_only = exclusive(pred, true), exclusive(true, pred)
    both = pred & true
    img = draw_grid(img, pred_only, (0, 0, 255))
    img = draw_grid(img, true_only, (255, 0, 0))
    img = draw_grid(img, both, (0, 255, 0))
    return img


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Use: python draw_grid.py evaluate/gridbox [filename]')
        print('evaluate to compare prediction to true label, gridbox to plot both grid and CVAT box')
        sys.exit(0)
    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        filename = get_sample_filename()
    if sys.argv[1] == 'evaluate':
        pred, true, = get_pred_and_true(filename)
        img = draw_pred_vs_true(filename, pred, true)
    elif sys.argv[1] == 'gridbox':
        label, boxes = get_label_and_boxes(filename)
        img = draw_grid_and_box(filename, label, boxes)
    else:
        print(f'Unknown argument {sys.argv[1]}')
    plt.figure(num=filename)
    plt.imshow(img)
    plt.show()
