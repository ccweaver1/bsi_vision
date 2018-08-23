from optparse import OptionParser
from vistools.research_models import ResearchModels
from vistools.data import Data
from vistools.config import ConfigManager
from utils.display import Display
from vistools.test_helper import get_model_and_load_weights, compile_model
from vistools.finding_file_helper import *
import numpy as np
import time
import datetime
import os
import re

'''
For local testing.  Uses configuration specified by -c, otherwise loads most recent.
Reads weight file from config file as well as other model params.
--path: a dir of images, or an image.
--config: a path to a training output .json file.  Specifies config, as well as results of training
'''

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to testing data.  Either a dir or an image")
parser.add_option("-c", "--config", dest="config_file", help="Config file.  Defaults to most recent")

(options, args) = parser.parse_args()

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%m-%d_%H%M')

if not options.test_path:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')


# Set up the configuration variables
# def get_most_recent_config():
#     return sorted(os.listdir('train_history/'), key=lambda x: os.path.getmtime(os.path.join('train_history', x)), reverse=True)[0]

if options.config_file:
    config_file = options.config_file
else:
    # config_file = get_most_recent_file_from_dir('train_history/')
    config_file = choose_file_with_stdin('train_history/')
print("Config file: {}".format(config_file))
cm = ConfigManager(config_file, folder='train_history')
cfg = cm.get_json()

model = get_model_and_load_weights(cfg)
compile_model(model, cfg['training'])

data = Data(options.test_path, cfg['training'], train_test='test')
print("Testing on {} samples".format(data.test_len))
data_gen = data.test_generator()

displayer = Display()
for frame_batch in data_gen:
    print('Frame batch')
    pred = model.predict_on_batch(frame_batch)
    # print pred
    H = np.reshape(pred, (3,3))
    print('Prediction:\n{}'.format(H))
    frame = np.array(frame_batch[0]*255).astype('uint8')

    warped = displayer.get_warp_overlay(frame, H, dsize=(300,300))
    displayer.display_vstacked(frame, warped, dsize=(600,600))
    # displayer.display_warp_overlay(frame, H, dsize=(600,600))
    print 'done'