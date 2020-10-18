import cv2
from os import listdir
import random
import requests
import json
import time
import re
import sys
import numpy as np


VIDEO_FOLDER = '../../datasets/alert_wildfire'
FRAMES_FOLDER = '../data/full'
FRAMES_PER_VIDEO = 200


def is_image_grayscale(img):
    # True if image is black and white, False if not
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    width, height, _ = img.shape
    img = img[width//3 : width//3*2, height//3 : height//3*2, :]
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    similar = np.vectorize(lambda x, y: abs(int(x)-int(y)) < 10)
    if (similar(b,g)).all() and (similar(b,r)).all(): return True
    return False


def split_video(videoname):
    # Saves FRAMES_PER_VIDEO frames of a video as jpg's. If it gets a black and white frame, it will keep looking until it finds a color frame
    vidcap = cv2.VideoCapture(VIDEO_FOLDER + '/' + videoname)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_gap = num_frames // FRAMES_PER_VIDEO if num_frames > FRAMES_PER_VIDEO else 1
    success = True
    frame, saved_frames_count = 0, 0
    print(videoname)
    print(f'Frame Gap: {frame_gap}')
    while saved_frames_count < FRAMES_PER_VIDEO:
        success, img = vidcap.read()
        if frame % frame_gap == 0:
            while success and is_image_grayscale(img):
                success, img = vidcap.read()
                frame += 1
            if not success:
                break
            jpg_name = videoname + '_frame_' + str(frame) + '.jpg'
            cv2.imwrite(FRAMES_FOLDER + '/' + jpg_name, img)
            saved_frames_count += 1
        frame += 1
    vidcap.release()


def get_done_videos():
    # Get list of videos that have already been split into frames
    with open('../data/done_videos.json', 'rb') as f:
        done_videos = set(json.load(f))
    return done_videos


def save_done_videos(done_videos):
    with open('../data/done_videos.json', 'w') as f:
        json.dump(list(done_videos), f)


if __name__ == '__main__':
    with open('../data/labels.json', 'rb') as f:
        videonames = json.load(f)['videonames']
    done_videos = get_done_videos()
    try:
        for videoname in videonames:
            if videoname not in done_videos:
                split_video(videoname)
                done_videos.add(videoname)
    except KeyboardInterrupt:
        print('Process interrupted by user. Saving list of done videos now.')
    except Exception as e:
        print(e)
    finally:
        save_done_videos(done_videos)