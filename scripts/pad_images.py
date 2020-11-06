import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor


STD_SHAPE = (1920, 1080)
FINAL_SIZE = 224
IN_FOLDER = '../data/full'
OUT_FOLDER = '../data/padded'


def pad_image(img):
    # takes a 1920x1080 px image and pads it with zeros to make it 1920x1920
    bigger_side = max(STD_SHAPE)
    padded = np.zeros((bigger_side, bigger_side, 3))
    diff_x = (bigger_side - img.shape[0]) // 2
    diff_y = (bigger_side - img.shape[1]) // 2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            padded[diff_x + i][diff_y + j][:] = img[i][j][:]
    return padded


def make_padded_resized_image(filename):
    # takes image filename and 
    print(filename)
    img = cv2.imread(IN_FOLDER + '/' + filename)
    print(type(img))
    if img.shape != STD_SHAPE:
        img = cv2.resize(img, STD_SHAPE)
    padded = pad_image(img)
    padded = cv2.resize(padded, (FINAL_SIZE, FINAL_SIZE))
    cv2.imwrite(OUT_FOLDER + '/' + filename, padded)
    print('Done')


if __name__ == '__main__':
    filenames = ["20130711_Bison Fire 2013-07-05 x25 Time Lapse.mp4_frame_0.jpg", "20150131_Sunrise from North Tahoe, Jan. 28th 2015.mp4_frame_0.jpg", "20150815_First ignition of Cold Springs Fire as seen from Fairview Peak at 3 PM on August 14th, 2015.mp4_frame_0.jpg", "20160615_Hawken Fire 3rd hour from Peavine camera.mp4_frame_0.jpg", "20160731_12 hour time lapse of Tule Fire 3rd night from Virginia Peak fire camera.mp4_frame_0.jpg"]
    for filename in filenames:
        make_padded_resized_image(filename)