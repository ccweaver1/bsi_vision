import os
import csv
import numpy as np
from processor import process_image
from random import shuffle
import mxnet as mx
from inceptionv3 import get_symbol
# import logging


input_layer_name = 'input_1'

def get_train_context(num_cpus, num_gpus):
    if num_gpus > 0:
        return mx.gpu()
    return mx.cpu()

def get_graph(output_name):
    return get_symbol(num_outputs=9, output_name=output_name)

def train(hyperparameters, channel_input_dirs, num_cpus, num_gpus, **kwargs):
    '''
    hyperparameters: The hyperparameters passed to SageMaker TrainingJob that runs your TensorFlow training
        script. You can use this to pass hyperparameters to your training script.
    '''
    config = hyperparameters
    batch_size = config['hyperparameters']['batch_size']
    target_shape = (3,) + tuple(config['model']['target_shape'])
    batch_shape = (batch_size,) + target_shape
    output_batch_shape = (batch_size, int(config['model']['nb_outputs']))
    
    # logging.getLogger().setLevel(logging.INFO)

    input_data_name = 'data'
    output_data_name = 'reg_out'

    if 'path_root' in kwargs:
        path_root = kwargs['path_root']
    else:
        path_root = None

    train_imglist = _input_fn(os.path.join(channel_input_dirs["root"], 'train_data.csv'), config, batch_size)
    train_img_iter = mx.image.ImageIter(batch_size, target_shape, imglist=train_imglist, path_root=path_root, shuffle=True, data_name=input_data_name, label_width=9, label_name=output_data_name)


    eval_imglist = _input_fn(os.path.join(channel_input_dirs['root'], 'test_data.csv'), config, batch_size)
    eval_img_iter = mx.image.ImageIter(batch_size, target_shape, imglist=eval_imglist, path_root=path_root, shuffle=True, data_name=input_data_name, label_width=9, label_name=output_data_name)


    tmat = train_img_iter.next_sample()[0]
    emat = eval_img_iter.next_sample()[0]

    learning_rate = float(config['hyperparameters']['learning_rate'])
    opt_name = config['model']['optimizer']
    if opt_name == 'adam':
        opt = mx.optimizer.Adam(learning_rate=learning_rate)
    num_epoch = config['hyperparameters']['nb_epochs']
    mod = mx.mod.Module(symbol=get_graph(output_name="reg_out"),
                            context=get_train_context(num_cpus, num_gpus),
                            data_names=[input_data_name],
                            label_names=[output_data_name])


    sym, arg_params, aux_params = mx.model.load_checkpoint('mx_model_zoo/inception_v3/Inception-7', 1)

    mod.bind(for_training=True, data_shapes=train_img_iter.provide_data, 
         label_shapes=train_img_iter.provide_label)
    mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True, force_init=True)

    print mod.predict(train_img_iter)
    train_img_iter.reset()
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])
    
    # data1 = train_img_iter.next_sample()
    # print data1[1]
    # import sys
    # sys.stdout.flush()
    # # print data1.shape
    # mod.forward(Batch(data1[0]))
    # print mod.get_outputs()[0].asnumpy()
    # batch = train_img_iter.next_sample()
    # mod.forward(train_img_iter)
    # print mod.get_outputs()[0].asnumpy()
    # print batch.label[0]
    # mod.backward()
    # mod.update()

    train_img_iter.reset()
    mod.fit(train_img_iter,
            eval_data=eval_img_iter,
            eval_metric='mse',
            num_epoch=num_epoch,
            initializer=mx.init.Xavier(),
            force_init=False,
            optimizer='adam',
            optimizer_params={"learning_rate": learning_rate}
            )
    return mod
    
def _input_fn(csv_filepath, config, batch_size):
    # data_filepath = os.path.join(training_dir, data_file)
    with open(csv_filepath, 'rb') as f:
        reader = csv.reader(f)
        rows = [x for x in reader]
        frame_files = [x[0] for x in rows]
        labels = [np.array(x[1:], dtype=np.float32) for x in rows]
        
    return zip(labels, frame_files)

if __name__ == "__main__":
    from config import ConfigManager
    config_manager = ConfigManager('inception_pool_config_gpu_sage.json', folder='configs/')
    config = config_manager.get_json()

    mx.viz.plot_network(get_graph('reg_out'))

    m = train(config['training'], {"root": "csvs/"}, 1, 0, path_root="/Users/campbellweaver/Documents/VisionData/")