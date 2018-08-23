import numpy as np
import json
# import xmltodict
import os
import re
import random
from processor import process_image


class Data:
    def __init__(self, data, config, train_test):
        '''
        data: input data to be managed by class.  If train_test = 'train', can be path to dir
            with frames/, annotations/ or can be txt file with list of paths to such dirs.  If
            train_test = 'test' can be path to image or dir of images
        train_test: either 'train' or 'test'.  If 'train', a subset of samples are set aside for
            evaluation (self.test_paths).  The ratio is determined in config
        config: a dictionary of config variables
        '''
        self.config = config
        if train_test == 'train':
            # If we're training, we expect 'data' to be a 'training dir' or 
            # a txt file containing a list of paths to 'training dirs'
            # Each of those paths should have frames/ and annotations/
            if os.path.isdir(data):
                paths = self.get_filepaths_from_dir(data)
            elif os.path.isfile(data):
                paths = self.get_paths_from_txtfile(data)
            else:
                raise ValueError('Cannot process directory, or file path: data')         
            self.train_paths, self.test_paths = self.split_train_test(paths, self.config['train_test_ratio'])
            self.train_len = len(self.train_paths)
            self.test_len = len(self.test_paths)
        elif train_test == 'test':
            # If we're testing, 'data' could be an image or a dir of images
            data_path = os.path.abspath(data)
            if os.path.isdir(data_path):
                if os.path.exists(os.path.join(data_path, 'frames')):
                    data_path = os.path.join(data_path, 'frames')
                self.test_paths = map(lambda x: os.path.join(data_path, x), os.listdir(data_path))
            elif os.path.isfile(data_path):
                self.test_paths = [data_path]
            else:
                raise ValueError('Please pass in an image, or a path to a directory of images.')
            self.test_len = len(self.test_paths)
    
    @staticmethod
    def get_paths_from_txtfile(trainfile):
        with open(trainfile) as paths:
            for path in paths:
                path = path.rstrip()
                self.get_data_from_path(path)

    @staticmethod
    def get_filepaths_from_dir(path):
        files = []
        try:
            # construct the annotations and frames dirs we expect inside top_dir
            annotations_dir = os.path.join(path, 'annotations') 
            frames_dir = os.path.join(path, 'frames')
            # get frame names and sort so we can match them up alongisde sorted annotations
            frame_names = sorted(os.listdir(frames_dir))                
            for filename in sorted(os.listdir(annotations_dir)):
                # if not a xml or json file, skip
                if filename.split('.')[1] not in ['xml', 'json']:
                    continue
                #get the full path for the annotation file
                annotations_path = os.path.join(annotations_dir, filename)
                filename_noext = filename.split('.')[0]
                for i in range(len(frame_names)):
                    # matchup the filenames of annotation and frame file
                    if re.match('{}.*'.format(filename_noext), frame_names[i]):
                        frame_path = os.path.join(frames_dir, frame_names[i])
                        # append both paths to our return list
                        files.append((annotations_path, frame_path))
                        # remove the frame name from our list since we already found it
                        frame_names.pop(i)
                        # get out of loop
                        break
        except IOError:
            print("{path} not setup properly")
            raise
        return files

    @staticmethod
    def split_train_test(files, ratio):
        test, train = [], []
        for f in files:
            if np.random.randn() > ratio:
                test.append(f)
            else:
                train.append(f)
        return train, test

    @staticmethod
    def get_data_from_paths(paths):
        '''
        returns tuple of np.array M matrices with shape (9,) and corresponding frame paths
        '''
        ret = []
        unannotated_files = 0
        for annotation_path, frame_path in paths:
            with open(annotation_path) as f:
                if annotation_path.endswith('.json'):
                    im_data = json.load(f)
                elif annotation_path.endswith('.xml'):
                    im_data = xd(f)
                else:
                    continue

                M = im_data.get('warp', {}).get('M', {})
                if not M:
                    unannotated_files += 1
                    continue
                M = np.array(M, dtype=np.float32).reshape(9)
                ret.append((M, frame_path))
        print('{} annotated files, {} un-annotated.'.format(len(ret), unannotated_files))
        return ret
            
    def train_test_generator(self, train_test):
        """Return a train or a val generator that we can use to train on.
        Depending on train_test will read from sel.train_paths or self.test_paths
        
        train_test: one of either 'train' or 'test'
        """
        data_paths = self.train_paths if train_test == 'train' else self.test_paths
        #need to extract outputs from data
        #convert (annotation_filepath, frame_filepath) -> (M, fram_filepath)
        #note: we're loading our y_truth vals into memory, but loading and preprocessing frames and gen time        
        data = self.get_data_from_paths(data_paths)
        print("Creating %s generator with %d samples." % (train_test, len(data)))
        while 1:
            X, y = [], []
            # Generate batch_size samples.
            for _ in range(self.config['hyperparameters']['batch_size']):
                # Get a random sample.
                train_sample = random.choice(data) # (M, frame)
                frame = process_image(train_sample[1], self.config['model']['target_shape'])
                #preprocess the frame

                inpt = frame #for now (should preprocess further)
                
                output = train_sample[0]

                X.append(inpt)
                y.append(output)

            yield np.array(X), np.array(y)


    def test_generator(self):
        '''
        Generates only input data, no validation
        '''

        for path in self.test_paths:
            try:
                single_frame_batch = np.array([process_image(path, self.config['model']['target_shape'])])
            except IOError as e:
                continue
            yield single_frame_batch


    def get_data_for_sagemaker(self, train_test, model_input):
        """Return a train or a val data set that we can use to train with.
        Depending on train_test will read from self.train_paths or self.test_paths

        This function is used by sagemaker
        
        train_test: one of either 'train' or 'test'

        return: np.array(data), np.array(labels)
        """
        # import tensorflow as tf

        data_paths = self.train_paths if train_test == 'train' else self.test_paths
        #need to extract outputs from data
        #convert (annotation_filepath, frame_filepath) -> (M, frame_filepath)
        data = self.get_data_from_paths(data_paths)
        outputs = []
        inputs = []

        for _ in range(self.config['hyperparameters']['batch_size']):
            # Get a random sample.
            train_sample = random.choice(data) # (M, frame)
            frame = process_image(train_sample[1], self.config['model']['target_shape'])
            #preprocess the frame
            
            label = train_sample[0]
            
            inputs.append(frame)
            outputs.append(label)

        return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)
        # return {model_input: tf.convert_to_tensor(inputs)}, tf.convert_to_tensor(outputs)