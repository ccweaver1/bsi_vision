
from utils.file_manager import FileManager
from vistools.config import ConfigManager
import os
import re
import json
import sys
import csv
import numpy as np
import random


'''
need to write csv_data to a local file
timestamp it
then transfer it to the cloud in an appropriate dir
'''

def extract_training_data(s3_bucket_name, top_dir):
    fm = FileManager(s3_bucket_name)

    annotation_dir = os.path.join(top_dir, 'annotations')
    frames_dir = os.path.join(top_dir, 'frames')
    
    annotations_names = fm.get_folder_list(annotation_dir, extension_filter='json')
    frame_names = [x for x in fm.get_folder_list(frames_dir) if len(x.split('.')) == 2]
    frame_exts = set([x.split('.')[1] for x in frame_names])
    frame_names = set(frame_names)
    
    data = []
    for filename in annotations_names:
        # if not json file, skip
        if not filename.split('.')[1] == 'json':
            continue
        # strange behavior with os.path.join
        # depending on whether or not looking in s3 bucket
        if s3_bucket_name is not None:
            annotation_filename = annotation_dir + filename
        else:
            annotation_filename = os.path.join(annotation_dir, filename)
        M = get_matrix_from_annotations(s3_bucket_name, annotation_filename)
        if M is not None:
            filename_noext = filename.split('.')[0]
            for ext in frame_exts:
                frame_name = filename_noext + '.' + ext
                if frame_name in frame_names:
                    # same strange behavior 
                    if s3_bucket_name is not None:
                        frame_path_from_bucket = top_dir + '/frames' + frame_name
                    else:
                        frame_path_from_bucket = os.path.join(top_dir, 'frames', frame_name)
                    data.append((frame_path_from_bucket, M))
    
    return data

def get_matrix_from_annotations(s3_bucket_name, annotation_path):
    if not annotation_path.endswith('.json'):
        raise ValueError('Cannot read from non json annotation file')
    fm = FileManager(s3_bucket_name)
    im_data = fm.read_image_dict('', annotation_path)
    M = im_data.get('warp', {}).get('M', {})
    if not M:
        return None

    M = np.array(M, dtype=np.float32).reshape(9)
    return M

def split_train_test_data(data, train_test_ratio, div_by=1):
    train, test = [], []
    data_len = len(data)
    train_len = int(data_len * train_test_ratio)
    train_len -= (train_len % div_by)
    test_len = int(data_len * (1 - train_test_ratio))
    test_len -= (test_len % div_by)
    remainder = data_len - train_len - test_len
    
    random.shuffle(data)
    while len(train) < train_len:
        train.append(data.pop())
    while len(test) < test_len:
        test.append(data.pop())
    if remainder > div_by:
        for _ in range(div_by):
            test.append(data.pop())
    
    # for d in data:
    #     if random.random() > train_test_ratio:
    #         test.append(d)
    #     else:
    #         train.append(d)
    return train, test

def write_csv_data_to_file(filename, data):
    with open(filename, 'wb') as f:
        wr = csv.writer(f)
        for filename, M in data:
            row = [x for x in M]
            row = [filename] + row
            wr.writerow(row)

if __name__ == '__main__':


    s3_top_dir = ''
    data = extract_training_data('bsivisiondata', 'PHI-PIT_6m-8m')
    print('Added PHI-PIT')
    data += extract_training_data('bsivisiondata', 'NYR-BOS_22m12s-22m30s')
    print('Added NYR-BOS')
    data += extract_training_data('bsivisiondata', 'DET-NSH_0h10m12s-0h10m17s')
    print('Added DET-NSH 1')
    data += extract_training_data('bsivisiondata', 'DET-NSH_0h7m45s-0h8m5s')
    print('Added DET-NSH 2')

    
    
    # print data
    data_len = len(data)
    train, test = split_train_test_data(data, 0.7, div_by=8)
    write_csv_data_to_file('csvs/train_data.csv', train)
    write_csv_data_to_file('csvs/test_data.csv', test)
    print('{} train samples, {} test samples, {} unused'.format(len(train), len(test), data_len - len(train) - len(test)))