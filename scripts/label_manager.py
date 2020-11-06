import cv2
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import random
import requests
import json
import time
import re
import sys

GRID_SIZE = 4
OVERLAP_THRESHOLD = 1 # Percentage of a grid square needed to be filled to be a 1
API_URL = 'http://216.232.184.219/api/v1'
LABEL_FILENAME = '../../datasets/labels.json'
VIDEO_FOLDER = '../../datasets/alert_wildfire/raw_data'
FRAMES_FOLDER = 'alert_wildfire_frames'
API_KEY = None


def overlap_area_percentage(tl1, br1, tl2, br2):
    #check if overlap
    if br1[1] < tl2[1] or br2[1] < tl1[1] or br1[0] < tl2[0] or br2[0] < tl1[0]:
        return 0.
    overlap_area = (min(br1[0], br2[0]) -  max(tl1[0], tl2[0])) * (min(br1[1], br2[1]) - max(tl1[1], tl2[1]))
    grid_area = (br2[0] - tl2[0]) * (br2[1] - tl2[1])
    return overlap_area / grid_area
    

def box_to_grid_vector(tl, br, width, height, grid_size=GRID_SIZE):
    grid_vector, overlap_areas = [], []
    grid_w, grid_h = width/grid_size, height/grid_size
    for i in range(grid_size):
        grid_vector.append([])
        overlap_areas.append([])
        for j in range(grid_size):
            grid_tl = (grid_w*j, grid_h*i)
            grid_br = (grid_w*(j+1), grid_h*(i+1))
            overlap_areas[i].append(overlap_area_percentage(tl, br, grid_tl, grid_br))
            if overlap_areas[i][j] > OVERLAP_THRESHOLD:
                grid_vector[i].append(1)
            else:
                grid_vector[i].append(0)
    # if box is not big enough for any square, fill the one it overlaps most
    if np.array_equal(grid_vector, np.zeros((grid_size, grid_size))):
        grid_vector[np.argmax(overlap_areas) // grid_size][np.argmax(overlap_areas) % grid_size] = 1
    return np.array(grid_vector)


def cvat_to_grid_vector(cvat_xml, video_filename):
    boxes = {}
    vidcap = cv2.VideoCapture(video_filename)
    width, height = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    labels = np.array([[[0 for j in range(GRID_SIZE)] for i in range(GRID_SIZE)] for _ in range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))])
    vidcap.release()
    root = cvat_xml
    cvat_size = root.find('meta').find('task').find('original_size')
    cvat_width, cvat_height = int(cvat_size.find('width').text), int(cvat_size.find('height').text)
    tracks = root.findall('track')
    for track in tracks:
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            if frame not in boxes:
                boxes[frame] = []
            xtl, ytl = int(float(box.get('xtl'))/cvat_width*width), int(float(box.get('ytl'))/cvat_height*height)
            xbr, ybr = int(float(box.get('xbr'))/cvat_width*width), int(float(box.get('ybr'))/cvat_height*height)
            tl, br = (xtl, ytl), (xbr, ybr)
            boxes[frame].append([tl, br])
            labels[frame] = labels[frame, :, :] | box_to_grid_vector(tl, br, width, height)
    return labels, boxes


def get_headers():
    global API_KEY
    if API_KEY is None:
        username = 'Raphael'
        email = 'raphael.menoni@gmail.com'
        password = 'projectx2020'
        auth = requests.post(f'{API_URL}/auth/login', data={
            'username': username,
            'email': email,
            'password': password
        })
        if auth.status_code != 200:
            print(f'Error when requesting auth key: {auth.status_code}\n{auth.content}')
            return None
        API_KEY = json.loads(auth.content)['key']
    return {'Authorization': f'Token {API_KEY}'}


def get_annotations(task_id):
    headers = get_headers()
    params = {'format': 'CVAT for video 1.1',
              'filename': './testtime',
              'action': 'download'}
    status_code = 202
    while status_code == 202 or status_code == 500:
        response = requests.get(f'{API_URL}/tasks/{task_id}/annotations',params=params, headers=headers)
        status_code = response.status_code
        time.sleep(1)
    if status_code != 200:
        raise Exception(f'Could not get annotations. Code {status_code}\nContent: {response.content}')
    annotations = ET.fromstring(response.content.decode('unicode-escape').split('annotations.xml')[1].split('PK')[0])
    return annotations


def get_task_page(page=1, page_size=10, completed=False, name=None, owner=None):
    headers = get_headers()
    params = {'page': page, 'page_size': page_size}
    if completed:
        params['status'] = 'completed'
    if name is not None:
        params['name'] = name
    if owner is not None:
        params['owner'] = owner
    response = requests.get(f'{API_URL}/tasks', headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f'{response.content}')
    task_page = json.loads(response.content)
    return task_page


def get_all_tasks(completed=False, name=None, owner=None):
    task_page = get_task_page(completed=completed, name=name, owner=owner)
    page_size = task_page['count']
    all_tasks = get_task_page(page_size=page_size, completed=completed, name=name, owner=owner)
    return all_tasks['results']


def videoname_from_task_id(task_id):
    headers = get_headers()
    response = requests.get(f'{API_URL}/tasks/{task_id}/data/meta', headers=headers)
    videoname = json.loads(response.content)['frames'][0]['name']
    videoname = '/'.join(videoname.split('/')[1:])
    return videoname


def file_exists(filename):
    try:
        f = open(filename)
    # Do something with the file
    except IOError:
        return False
    finally:
        f.close()
        return True


def get_video_metadata(videoname):
    vidcap = cv2.VideoCapture(f'{VIDEO_FOLDER}/{videoname}')
    width, height = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.release()
    return {
        'width': width,
        'height': height,
        'num_frames': num_frames
    }


def mankind_xml_to_size_and_boxes(xml_filename):
    print(xml_filename)
    root = ET.parse(xml_filename).getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        box = obj.find('bndbox')
        tl = [int(box.find('xmin').text), int(box.find('ymin').text)]
        br = [int(box.find('xmax').text), int(box.find('ymax').text)]
        boxes.append([tl, br])
    return (width, height), boxes


def main_cvat_to_grid_server():
    tasks = get_all_tasks(completed=True)
    print(len(tasks))
    try:
        label_file = open(LABEL_FILENAME, 'rb')
    except FileNotFoundError:
        label_json = {'videonames': [], 'labels': {}, 'cvat_boxes': {}}
    else:
        label_json = json.load(label_file)
        label_file.close()
    videonames, labels, cvat_boxes = set(label_json['videonames']), label_json['labels'], label_json['cvat_boxes']
    try:
        for task in tasks:
            videoname = videoname_from_task_id(task['id'])
            if videoname not in videonames:
                print(videoname, task['id'])
                annotations = get_annotations(task['id'])
                ET.ElementTree(annotations).write('output.xml')
                grid_vector, boxes = cvat_to_grid_vector(annotations, f'{VIDEO_FOLDER}/{videoname}')
                for frame in range(grid_vector.shape[0]):
                    jpg_name = videoname + '_frame_' + str(frame) + '.jpg'
                    labels[jpg_name] = grid_vector[frame].flatten().tolist()
                    if frame in boxes:
                        cvat_boxes[jpg_name] = boxes[frame]
                videonames.add(videoname)
    except KeyboardInterrupt:
        print('Interrupted by user. Saving file now.')
    except FileNotFoundError as e:
        print(f'Could not find file:\n{e}')
    except Exception as e:
        print(e)
    finally:
        label_json = {'videonames': list(videonames), 'labels': labels, 'cvat_boxes': cvat_boxes}
        with open(LABEL_FILENAME, 'w') as label_file:
            json.dump(label_json, label_file)
            sys.exit(0)


def main_cvat_to_grid_local(grid_size=GRID_SIZE):
    with open(LABEL_FILENAME, 'rb') as label_file:
        label_json = json.load(label_file)
    boxes = label_json['cvat_boxes']
    videonames = label_json['videonames']
    new_labels = {}
    for videoname in videonames:
        metadata = get_video_metadata(videoname)
        width, height, num_frames = metadata['width'], metadata['height'], metadata['num_frames']
        for frame in range(num_frames):
            grid_vector = np.zeros((grid_size, grid_size), int)
            framename = f'{videoname}_frame_{frame}.jpg'
            if framename in boxes:
                for box in boxes[framename]:
                    tl, br = box[0], box[1]
                    grid_vector = grid_vector | box_to_grid_vector(tl, br, width, height, grid_size)
            new_labels[framename] = grid_vector.flatten().tolist()
    new_label_json = {'labels': new_labels, 'videonames': videonames, 'cvat_boxes': boxes}
    with open(f'../data/full/labels_{grid_size}.json', 'w') as f:
        json.dump(new_label_json, f)


def main_mankind_to_grid(grid_size=GRID_SIZE):
    xml_folder = '../../datasets/annotated_bounding_box_hpwren/annotated_bounding_box_hpwren/xmls/'
    res = {'labels': {}}
    for filename in listdir(xml_folder):
        if '._' in filename:
            continue
        print(filename)
        size, boxes = mankind_xml_to_size_and_boxes(xml_folder + filename)
        grid_vector = np.zeros((grid_size, grid_size), int)
        for box in boxes:
            grid_vector = grid_vector | box_to_grid_vector(box[0], box[1], size[0], size[1], grid_size)
        img_filename = filename.split('.xml')[0] + '.jpeg'
        res['labels'][img_filename] = grid_vector.flatten().tolist()
    with open(f'../data/full/labels_mankind_{grid_size}.json', 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Options:\nserver if you want to fetch annotations from CVAT server\nlocal if you have a local labels.json already')
    elif sys.argv[1] == 'server':
        main_cvat_to_grid_server()
    elif sys.argv[1] == 'local':
        main_cvat_to_grid_local(int(sys.argv[2]))
    elif sys.argv[1] == 'mankind':
        main_mankind_to_grid(int(sys.argv[2]))
