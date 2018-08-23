from model_helper import get_model_and_load_weights, get_callbacks, compile_model
# from data import Data
import tensorflow as tf
import os
import csv
import numpy as np
from processor import process_image
from random import shuffle



input_layer_name = 'input_1'

def keras_model_fn(hyperparameters):
    '''
    hyperparameters: The hyperparameters passed to SageMaker TrainingJob that runs your TensorFlow training
        script. You can use this to pass hyperparameters to your training script.
    '''
    # Logic to do the following:
    # 1. Instantiate the Keras model
    # 2. Compile the keras model

    config = hyperparameters
    model = get_model_and_load_weights(config)
    compile_model(model, config)

    return model


def train_input_fn(training_dir, hyperparameters):
    # Logic to the following:
    # 1. Reads the **training** dataset files located in training_dir
    # 2. Preprocess the dataset
    # 3. Return 1)  a dict of feature names to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    config = hyperparameters
    train_file = os.path.join(training_dir, 'train_data.csv')
    print train_file   
    batch_size = batch_size = config['hyperparameters']['batch_size']
    features, labels = _input_fn(training_dir, train_file, config, batch_size)
    # print features
    # print labels
    return tf.estimator.inputs.numpy_input_fn(
                x={input_layer_name: features},
                y=labels,
                num_epochs=None,
                shuffle=True)()
 
def eval_input_fn(training_dir, hyperparameters):
    # Logic to the following:
    # 1. Reads the **evaluation** dataset files located in training_dir
    # 2. Preprocess the dataset
    # 3. Return 1)  a dict of feature names to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels

    config = hyperparameters
    test_file = os.path.join(training_dir, 'test_data.csv')
    # features, labels = data.get_data_for_sagemaker('test', input_layer_name)
    features, labels = _input_fn(training_dir, test_file, config, None)

    return tf.estimator.inputs.numpy_input_fn(
                x={input_layer_name: features},
                y=labels,
                num_epochs=None,
                shuffle=True)()

def _input_fn(training_dir, data_file, config, batch_size):
    with open(data_file, 'rb') as f:
        reader = csv.reader(f)
        rows = [x for x in reader]
        frame_files = [os.path.join(training_dir, x[0]) for x in rows]
        labels = [np.array(x[1:], dtype=np.float32) for x in rows]
        zip_files_labels = zip(frame_files, labels)
        shuffle(zip_files_labels)
        frame_files = [x[0] for x in zip_files_labels[:batch_size]]
        labels = np.array([x[1] for x in zip_files_labels[:batch_size]])

    # print labels
    features = np.array([process_image(f, config['model']['target_shape']) for f in frame_files])
    return features, labels




if __name__ == "__main__":
    import tensorflow as tf
    import json
    from config import ConfigManager
    config_manager = ConfigManager('resnet50_pool_config_sage.json', folder='../configs/')
    config = config_manager.get_json()

    m = keras_model_fn(config)
    print(m.summary())
    features, labels = train_input_fn('', 'train_data.csv', config)
    print features
    
    import os
    print('Fitting model, pid = {}'.format(os.getpid()))
    # m.fit(
    #     generator=gen,
    #     steps_per_epoch=2,
    #     epochs=config['hyperparameters']['nb_epochs'],
    #     verbose=1,
    #     validation_data=test_gen,
    #     validation_steps=d.test_len)    

