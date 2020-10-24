import cv2
from os import listdir
import json
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# This script splits the videos in the raw_data folder into frames. Since it is a multi-hour long process, it saves the names of the videos it has already split into a json file called done_videos.json
# If it is called again and done_videos.json already exists, it will skip the videos that have already been split


#VIDEO_FOLDER = '../../datasets/alert_wildfire/raw_data'
#FRAMES_FOLDER = '../../datasets/frames_test'
#DONE_VIDEOS_PATH = '../../datasets/done_videos.json'
VIDEO_FOLDER = '../data/full/raw_data'
FRAMES_FOLDER = '../data/full/frames'
DONE_VIDEOS_PATH = '../data/done_videos.json'
FRAMES_PER_VIDEO = 100


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
    while saved_frames_count < FRAMES_PER_VIDEO:
        success, img = vidcap.read()
        if frame % frame_gap == 0:
            while success and is_image_grayscale(img):
                # I make 10 frame jumps here because checking for b&w is slow
                for _ in range(10):
                    success, img = vidcap.read()
                    frame += 1
            if not success:
                break
            jpg_name = videoname + '_frame_' + str(frame) + '.jpg'
            cv2.imwrite(FRAMES_FOLDER + '/' + jpg_name, img)
            saved_frames_count += 1
        frame += 1
    vidcap.release()
    done_videos = get_done_videos()
    done_videos.add(videoname)
    save_done_videos(done_videos)


def get_done_videos():
    # Get list of videos that have already been split into frames
    try:
        with open(DONE_VIDEOS_PATH, 'rb') as f:
            done_videos = set(json.load(f))
    except IOError:
        done_videos = set()
    return done_videos


def save_done_videos(done_videos):
    with open(DONE_VIDEOS_PATH, 'w') as f:
        json.dump(list(done_videos), f)
    print(done_videos)


if __name__ == '__main__':
    videonames = listdir(VIDEO_FOLDER)
    with ProcessPoolExecutor() as executor:
        executor.map(split_video, videonames)