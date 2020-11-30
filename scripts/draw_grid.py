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
import argparse

GRID_SIZE = 4
DATA_FOLDER = '../data/full/frames_100'


def parse_args():
    parser = argparse.ArgumentParser(description="Make labels")
    parser.add_argument('type', help='evaluate or gridbox')
    parser.add_argument('filename', nargs='?', help='name of the image file')
    parser.add_argument('--gridsize', type=int, help="Size of the label grid", required=False)
    args = parser.parse_args()
    return args


def get_sample_filename():
    filenames = listdir(DATA_FOLDER)
    while True:
        filename = random.choice(filenames)
        if 'jpg' in filename:
            return filename


def get_label_and_boxes(filename, gridsize=None):
    if gridsize is None:
        gridsize = GRID_SIZE
    with open(f'../data/full/labels_{gridsize}.json', 'rb') as f:
        labels_json = json.load(f)
    label = np.array(labels_json['labels'][filename]).reshape((GRID_SIZE,GRID_SIZE))
    boxes = labels_json['cvat_boxes'][filename] if filename in labels_json['cvat_boxes'] else []
    return label, boxes


def draw_grid(img, grid_vector, color=(255, 0, 0), gridsize=None):
    if gridsize is None:
        gridsize = GRID_SIZE
    height, width, _ = img.shape
    grid_w, grid_h = width//gridsize, height//gridsize
    img = img.copy()
    # first draw grid lines
    for i in range(gridsize):
        x = i*grid_w
        y = i*grid_h
        for j in range(height):
            img[j, x, :] = color
        for j in range(width):
            img[y, j, :] = color
    # now highlight smoke areas
    highlight = np.vectorize(lambda x, s: min(255, x+s//5))
    for i in range(gridsize):
        for j in range(gridsize):
            if grid_vector[j][i] == 1:
                for k in range(i*grid_w, (i+1)*grid_w):
                    for l in range(j*grid_h, (j+1)*grid_h):
                        img[l, k, :] = highlight(img[l, k, :], color)
    return img


def draw_grid_and_box(filename, label, boxes=[], gridsize=None):
    img = cv2.imread(DATA_FOLDER + '/' + filename)
    for box in boxes:
        tl = (int(box[0][0]), int(box[0][1]))
        br = (int(box[1][0]), int(box[1][1]))
        img = cv2.rectangle(img, tl, br, (0, 0, 255), 2)
    img = draw_grid(img, label, color=(255, 0, 0), gridsize=gridsize)
    return img


def get_pred_and_true(filename, gridsize):
    if gridsize is None:
        gridsize = GRID_SIZE
    df = pd.read_csv('../data/results.csv')
    idx = df[df['file_path'] == filename].index[0]
    pred = np.array([int(float(val)) for val in df.iloc[idx]['pred'].strip('[]').split(' ')]).reshape((gridsize, gridsize))
    true = np.array([int(float(val)) for val in df.iloc[idx]['label'].strip('[]').split(' ')]).reshape((gridsize, gridsize))
    return pred, true


def draw_pred_vs_true(img, pred, true, gridsize):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    exclusive = np.vectorize(lambda x, y: 1 if x == 1 and y == 0 else 0)
    pred_only, true_only = exclusive(pred, true), exclusive(true, pred)
    both = pred & true
    img = draw_grid(img, pred_only, (0, 0, 255), gridsize)
    img = draw_grid(img, true_only, (255, 0, 0), gridsize)
    img = draw_grid(img, both, (0, 255, 0), gridsize)
    return img


if __name__ == '__main__':
    args = parse_args()
    if len(sys.argv) == 1:
        print('Use: python draw_grid.py evaluate/gridbox [filename]')
        print('evaluate to compare prediction to true label, gridbox to plot both grid and CVAT box')
        sys.exit(0)
    if args.filename is not None:
        filename = args.filename
    else:
        filename = get_sample_filename()
    if args.type == 'evaluate':
        pred, true, = get_pred_and_true(filename, args.gridsize)
        img = draw_pred_vs_true(cv2.imread(DATA_FOLDER + '/' + filename), pred, true, args.gridsize)
    elif args.type == 'gridbox':
        label, boxes = get_label_and_boxes(filename)
        img = draw_grid_and_box(filename, label, boxes, args.gridsize)
    else:
        print(f'Unknown argument {args.type}')
    plt.figure(num=filename)
    plt.imshow(img)
    plt.show()
