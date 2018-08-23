import xmltodict as xd
import json
import os
import cv2
import urllib
import boto3
import numpy as np
import csv
from utils.file_manager import FileManager
from utils.processor import *
from random import shuffle


class DataExtractor:
    def __init__(self, parent_dir, s3_bucket=None, csv_file=None):

        self.parent_dir = parent_dir
        self.csv_file = csv_file

        if s3_bucket is None :
            if not csv_file:
                self.data_dir = os.path.join(parent_dir, 'annotations')
                self.images_dir = os.path.join(parent_dir, 'frames')
            self.fm = FileManager()
        else :
            if not csv_file:
                self.data_dir = parent_dir + '/annotations/'
                self.images_dir = parent_dir + '/frames/'
            self.fm = FileManager(s3_bucket)

    def get_all_data(self):
        '''
        returns: list of dictionaries: [{'data': <data_dict>, 'image': <cv2 image>}, ...]
        '''
        all_data = []

        for im_data in self.extractor.get_data_generator() :
            all_data.append(im_data)
        return all_data       

    def get_data_generator(self, randomize):
        '''
        Simple data generator.
        '''
        if self.csv_file:
            zip_files_labels = self.get_imglist()
            if randomize:
                shuffle(zip_files_labels)
            for f_file, label in zip_files_labels:
                frame = self.fm.read_image_file('',f_file)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                data_dict = {'warp': {'M': label}}
                
                yield {'data': data_dict, 'image': frame, 'filename': f_file}
        else:
            file_list = self.fm.get_folder_list(self.data_dir, extension_filter='json')
            if randomize:
                shuffle(file_list)
            for filename in file_list :
                x_imageDict = self.fm.read_image_dict(self.data_dir, filename)
                im_file = x_imageDict['annotation']['filename']
                
                frame = self.fm.read_image_file(self.images_dir,im_file)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

                yield {'data': x_imageDict, 'image': frame, 'filename' : filename}


    def get_annotation_generator(self, randomize):
        '''
        Simple data generator.
        '''
        if self.csv_file:
            zip_files_labels = self.get_imglist()
            if randomize:
                shuffle(zip_files_labels)
            for f_file, label in zip_files_labels:
                frame = self.fm.read_image_file('',f_file)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                data_dict = {'warp': {'M': label}}
                
                yield {'data': data_dict, 'image': frame, 'filename': f_file}
        else:
            file_list = self.fm.get_folder_list(self.data_dir, extension_filter='json')
            if randomize:
                shuffle(file_list)
            for filename in file_list :
                x_imageDict = self.fm.read_image_dict(self.data_dir, filename)
                im_file = x_imageDict['annotation']['filename']

                yield {'data': x_imageDict, 'filename' : filename}




    def get_processed_data_generator(self, randomize, batch_size, target_shape, return_unprocessed=False):
        """
        Return a data generator with preprocessing/augmentation
        (should be the 'training' section of cfg)
        Yields lists X and y of batch_size       
        """
        #need to extract outputs from data
        #convert (annotation_filepath, frame_filepath) -> (M, fram_filepath)
        #note: we're loading our y_truth vals into memory, but loading and preprocessing frames and gen time        
        data = self.get_data_generator(randomize=randomize)
        while 1:
            X, y, filepaths, originals = [], [], [], []
            # Generate batch_size samples.
            for _ in range(batch_size):
                # Get a random sample.
                train_sample = next(data) # {'data': dict, 'image': numpy.ndarray, 'filename': name}
                frame = process_image(train_sample['image'], target_shape)
                #preprocess the frame

                inpt = frame #for now (should preprocess further)
                
                output = train_sample['data']['warp']['M']

                X.append(inpt)
                y.append(output)
                if return_unprocessed:
                    originals.append(train_sample['image'])
                filepaths.append(train_sample['filename'])

            if return_unprocessed:
                yield np.array(X), np.array(y), filepaths, np.array(originals)
            else:
                yield np.array(X), np.array(y), filepaths    


    def get_imglist(self, csv_file, div_by=1):
        #returns a list of [[frame_file, label], [frame_file...]]
        # if div_by is specified, length of list will be divisible by that number

        with open(csv_file, 'rb') as f:
            reader = csv.reader(f)
            rows = [x for x in reader]
            frame_files = [os.path.join(self.parent_dir, x[0]) for x in rows]
            labels = np.array([np.array(x[1:], dtype=np.float32) for x in rows])
    
        zip_files_labels = zip(frame_files, labels)
        remainder = len(zip_files_labels) % div_by
        if remainder > 0:
            return zip_files_labels[:-remainder]
        return zip_files_labels

    def get_data_length(self, csv_file=None):
        if not csv_file:
            csv_file = self.csv_file
        if csv_file:
            with open(csv_file, 'rb') as f:
                for i, l in enumerate(f):
                    pass
                return i+1
        else:
            file_list = self.fm.get_folder_list(self.data_dir, extension_filter='json')
            return len(file_list)

    def save_imdata(self, filename, data):

        filename = filename.replace('.xml', '.json')
        self.fm.save_image_dict(self.data_dir,filename,data)
