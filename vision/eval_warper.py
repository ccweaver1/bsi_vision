from vistools.research_models import ResearchModels
# from vistools.data import Data
from utils.data_extractor import DataExtractor
from vistools.config import ConfigManager
from utils.display import Display
from vistools.test_helper import get_model_and_load_weights, compile_model, ckpt_up_to_date
from vistools.finding_file_helper import *
from keras.losses import mean_squared_error
from keras import backend as K
import numpy as np
import time
import datetime
import os
import re
import sys




def evaluate(csv_file, eval_path="", s3_bucket="bsivisiondata", train_config=None, write_output_images=False):
    '''
    eval_path: top directory, look for frames/ and annotations/ underneath here.  If csv specified, then paths
        will be relative to this eval_path
    csv_file: a csv of paths_to_images,labels.  Paths are relative to eval_path
    s3_bucket: name of an s3 bucket where image and label data is read from. If csv is specified, labels will be
        taken from the csv
    train_config: a .json config file from train_history that gives information on the training job that we're
        evaluating
    write_output_images: bool, whether or not to write output images to data/eval_images/training_job_name_steps
    loop_eval_period: int, the period in seconds to loop evaluation runs.  If not set, only runs once
    '''

    if train_config:
        config_file = train_config
    else:
        # config_file = get_most_recent_file_from_dir('train_history/')
        config_file = choose_file_with_stdin('train_history/')
    print("Config file: {}".format(config_file))
    cm = ConfigManager(config_file, folder='train_history')
    cfg = cm.get_json()

    data = DataExtractor(eval_path, s3_bucket, csv_file=csv_file)
        
    # batch_size = cfg['training']['hyperparameters']['batch_size']
    batch_size = 1
    target_shape = cfg['training']['model']['target_shape']
    data_gen = data.get_processed_data_generator(True, batch_size, target_shape, True)

    print("Testing on {} samples".format(data.get_data_length()))

    displayer = Display()

    model, path_and_prefix = get_model_and_load_weights(cfg)
    compile_model(model, cfg['training'])
    ckpt_prefix = os.path.basename(path_and_prefix)
    
    total_loss = 0
    num_testing_samples = 0
    for X, y, filepaths, originals in data_gen:

        pred = model.predict_on_batch(X)

        mse_out = mean_squared_error(y, pred)
        loss = mse_out.eval(session=K.get_session())
        total_loss += loss[0]
        num_testing_samples += 1
        print("Loss: {}".format(loss[0]))

        H = np.reshape(pred, (3,3))
        # print('Prediction:\n{}'.format(H))
        
        if write_output_images: 
            frame = originals[0]
            filepath = filepaths[0]
            image_text = "Pred: {}\nTruth: {}\nLoss: {}".format(np.array_str(pred), np.array_str(y[0]), str(loss[0]))
            warped = displayer.get_warp_overlay(frame, H, dsize=(1000,425))
            vstacked = displayer.get_vstacked(frame, warped, dsize=(1000,850))
            displayer.put_rectangle(vstacked, (10, 10), (800, 80))
            displayer.put_text(vstacked, "Pred: {}".format(np.array_str(pred, precision=2)), text_loc=(10,30))
            displayer.put_text(vstacked, "Truth: {}".format(np.array_str(y[0], precision=2)), text_loc=(10,50))
            displayer.put_text(vstacked, "loss: {}".format(loss[0]), text_loc=(10,70))
            step_number = path_and_prefix.split('-')[-1]
            job_name_and_step_num = cfg['sagemaker_job_info']['job_name'] + '-step' + str(step_number)
            job_name_dir = os.path.join('data/eval_images', job_name_and_step_num)
            if not os.path.exists(job_name_dir):
                os.makedirs(job_name_dir)
            filename = "{:.2E}-{}".format(loss[0], os.path.basename(filepath))
            displayer.save_image(os.path.join(job_name_dir, filename), vstacked)
        # displayer.display_vstacked(frame, warped, dsize=(1000,850))
        # displayer.display_warp_overlay(frame, H, dsize=(600,600))
    average_loss = total_loss / num_testing_samples
    print("Average loss: {}".format(average_loss))
    return average_loss, path_and_prefix


def evaluate_on_timer(csv_file, top_dir="", s3_bucket="bsivisiondata", train_config=None, write_output_images=False, timer_period=300, num_iterations=3): 
    if train_config:
        config_file = train_config
    else:
        # config_file = get_most_recent_file_from_dir('train_history/')
        config_file = choose_file_with_stdin('train_history/')
    print("Config file: {}".format(config_file))
    cm = ConfigManager(config_file, folder='train_history')
    cfg = cm.get_json()

    average_loss, path_and_prefix = evaluate(csv_file, top_dir, s3_bucket, train_config, write_output_images)
    ckpt_prefix = os.path.basename(path_and_prefix)
    for i in range(num_iterations-1):
        print("Sleeping")
        time.sleep(timer_period)
        ckpt_dir = cfg['sagemaker_job_info']['ckpt_dir']
        ckpt_s3_bucket = cfg['sagemaker']['output_bucket_name']
        if ckpt_up_to_date(ckpt_prefix, ckpt_s3_bucket, ckpt_dir): 
            print('No updated ckpt available for evaluation')
            continue
        average_loss, ckpt_prefix = evaluate(csv_file, top_dir, s3_bucket, train_config, write_output_images)


if __name__ == "__main__":
    '''
    For local evaluation.  Uses configuration specified by, otherwise will ask you to choose one.
    Reads weight file from config file as well as other model params.
    '''
    csv_file = "csvs/test_data.csv"
    top_dir = ""
    s3_bucket = "bsivisiondata"
    train_config = "train_history/bsi-training-vgg16-pool-2018-06-12-21-36-34.json"
    write_output_images = True

    evaluate(csv_file, top_dir, s3_bucket, train_config, write_output_images)
    # evaluate_on_timer(csv_file, top_dir, s3_bucket, train_config, write_output_images, timer_period=10, num_iterations=3)