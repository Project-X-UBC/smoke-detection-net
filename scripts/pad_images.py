import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor


STD_SHAPE = (1920, 1080)
FINAL_SIZE = 1920
IN_FOLDER = '../data/full'
OUT_FOLDER = '../data/padded'


def pad_image(img):
    # takes a 1920x1080 px image and pads it with zeros to make it 1920x1920
    padded = np.zeros((FINAL_SIZE, FINAL_SIZE, 3))
    diff_x = (FINAL_SIZE - img.shape[0]) // 2
    diff_y = (FINAL_SIZE - img.shape[1]) // 2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            padded[diff_x + i][diff_y + j][:] = img[i][j][:]
    return padded


def make_padded_image(filename):
    img = cv2.imread(IN_FOLDER + '/' + filename)
    if img.shape != STD_SHAPE:
        img = cv2.resize(img, STD_SHAPE)
    padded = pad_image(img)
    cv2.imwrite(OUT_FOLDER + '/' + filename, padded)
    img.release()


if __name__ == '__main__':
    filenames = ["20130711_Bison Fire 2013-07-05 x25 Time Lapse.mp4_frame_336.jpg", "20150131_Sunrise from North Tahoe, Jan. 28th 2015.mp4_frame_1393.jpg", "20150815_First ignition of Cold Springs Fire as seen from Fairview Peak at 3 PM on August 14th, 2015.mp4_frame_875.jpg", "20160615_Hawken Fire 3rd hour from Peavine camera.mp4_frame_427.jpg", "20160731_12 hour time lapse of Tule Fire 3rd night from Virginia Peak fire camera.mp4_frame_0.jpg"]
    with ProcessPoolExecutor() as executor:
        executor.map(make_padded_image, filenames)