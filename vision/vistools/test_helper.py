from utils.file_manager import FileManager
from tensorflow.python import pywrap_tensorflow
from research_models import ResearchModels
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
import re
try:
    from keras import losses
    from keras.optimizers import Adam
except ImportError:
    from tensorflow.python.keras import losses
    from tensorflow.python.keras.optimizers import Adam


#need to create a dir to put the downloaded stuff in before you try to download
#need to get them downloaded and load weights
#then run the thing and get predictions locally
#then figure out deploying test endpoint

def get_model_and_load_weights(cfg):
    models = ResearchModels(cfg['training'])
    model = models.get_model()
    local_ckpt_dir = os.path.join('tf_ckpts', cfg['sagemaker_job_info']['job_name'], 'checkpoints')
    if not os.path.isdir(local_ckpt_dir):
        # create a local dir to mirror s3
        os.makedirs(local_ckpt_dir)
        # os.mkdir(os.path.join('tf_ckpts', cfg['sagemaker_job_info']['job_name']), 'checkpoints')
    # 
    s3_bucket = cfg['sagemaker']['output_bucket_name']
    ckpt_dir = cfg['sagemaker_job_info']['ckpt_dir']
    if len(os.listdir(local_ckpt_dir)) < 2: #not enough files in dir
        # download from s3 into dir
        path_and_prefix = download_ckpt_to_dir(s3_bucket, ckpt_dir, local_ckpt_dir)
    else: #existing files in local dir
        ckpt_names = os.listdir(local_ckpt_dir)
        most_recent_ckpt_prefix = most_recent_ckpt_from_list(ckpt_names)
        if not ckpt_up_to_date(most_recent_ckpt_prefix, s3_bucket, ckpt_dir): #not up to date
            path_and_prefix = download_ckpt_to_dir(s3_bucket, ckpt_dir, local_ckpt_dir)
        else: #up to date local files
            path_and_prefix = os.path.join(local_ckpt_dir, most_recent_ckpt_prefix)

    keras_load_ckpt(path_and_prefix, model)
    return model, path_and_prefix


def ckpt_up_to_date(local_ckpt_prefix, bucket, folder):
    fm = FileManager(bucket)
    ckpt_names = fm.get_folder_list(folder)
    most_recent_prefix = most_recent_ckpt_from_list(ckpt_names)
    if not local_ckpt_prefix == most_recent_prefix:
        return False
    return True

def most_recent_ckpt_from_list(ckpts):
    exp = re.compile('model\.ckpt-(\d+).*')
    max_step_num = 0
    ret = None
    for ckpt in ckpts:
        search = re.search(exp, ckpt)
        if search:
            train_step_num = int(search.group(1))
            if train_step_num > max_step_num:
                max_step_num = train_step_num
                ret = ckpt
    prefix = '.'.join(ret.split('.')[:2])
    return prefix

def download_ckpt_to_dir(bucket, folder, dest_dir):
    '''
    download most recent ckpt files (index, meta, data) from s3 bucket and directory.
    dest: local folder to put files into
    returns: path/to/ckpt_prefix
    '''

    fm = FileManager(bucket)
    # need .data and .index files (don't necessarily need meta, but will download)
    ckpt_names = fm.get_folder_list(folder)
    most_recent_ckpt_prefix = most_recent_ckpt_from_list(ckpt_names)  
    print('Downloading ckpts from s3: {}'.format(most_recent_ckpt_prefix))
    ckpt_file_names = [x for x in ckpt_names if most_recent_ckpt_prefix in x]

    path_and_prefix = dest_dir + '/' + most_recent_ckpt_prefix
    for key in ckpt_file_names:
        dest_filepath = dest_dir + '/' + key
        fm.download_file(folder, key, dest_filepath)
    return path_and_prefix

    # fm.download_file


def keras_load_ckpt(checkpoint_prefix, model):
    '''
    checkpoint_file: path to a tensorflow ckpt.  Should be a filename prefix.
        Requires .data, .index, .meta files
    model: a keras model into which the weights are loaded
    '''
    # model.summary()
    # print_tensors_in_checkpoint_file(checkpoint_file, 'conv1/kernel', False, all_tensor_names=False)
    print("Loading weights from {}".format(checkpoint_prefix))
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_prefix)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # inspect_ckpt(checkpoint_prefix)
    for layer in model.layers:
        if layer.count_params() == 0:
            continue
        # print layer.name, layer.count_params()
        # print reader.get_tensor(layer.name)

        weights_name = layer.name + '/kernel'
        bias_name = layer.name + '/bias'
        keras_params = []
        if weights_name in var_to_shape_map:
            keras_params.append(reader.get_tensor(weights_name))
        if bias_name in var_to_shape_map:
            keras_params.append(reader.get_tensor(bias_name))
        layer.set_weights(keras_params)


def inspect_ckpt(checkpoint_prefix):
    print_tensors_in_checkpoint_file(checkpoint_prefix, tensor_name='', all_tensors=False, all_tensor_names=True)

def compile_model(model, cfg):
    '''
    model: keras model
    config: dict of config values
    '''
    learning_rate = cfg['hyperparameters']['learning_rate']
    decay = cfg['hyperparameters']['decay']
    loss_name = cfg['model']['loss']
    optimizer_name = cfg['model']['optimizer']
    if loss_name == 'mean_squared_error':
        loss = losses.mean_squared_error
    if optimizer_name == 'Adam':
        optimizer = Adam(lr=learning_rate, decay=decay) #Adam is introducing a *= warning. FYI
    model.compile(loss=loss, optimizer=optimizer)

if __name__ == '__main__':
    # download_ckpt_to_dir("sagemaker-us-east-1-359098845427", "", '.')
    # download_ckpt_to_dir("sagemaker-us-east-1-359098845427", "bsi-training-dense-2018-05-03-17-03-40-446/checkpoints", 'tf_ckpts')
    # x = ['model.ckpt-100.data', 'model.ckpt-200.data', 'model.ckpt-1000.data']
    # print most_recent_ckpt_from_list(x)
    pass