from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow

def keras_load_ckpt(checkpoint_file, model):
    '''
    checkpoint_file: path to a tensorflow ckpt.  Should be a filename prefix.
        Requires .data, .index, .meta files
    model: a keras model into which the weights are loaded
    '''
    # model.summary()
    # print_tensors_in_checkpoint_file(checkpoint_file, 'conv1/kernel', False, all_tensor_names=False)
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    for layer in model.layers:
        if layer.count_params() == 0:
            continue
        # print layer.name
        # print layer.count_params()

        layer_weights = reader.get_tensor(layer.name + '/kernel')
        layer_biases = reader.get_tensor(layer.name + '/bias')
        keras_weights = [layer_weights, layer_biases]
        layer.set_weights(keras_weights)
        

def inspect_ckpt(checkpoint_file):
    print_tensors_in_checkpoint_file(checkpoint_file, tensor_name='', all_tensors=False, all_tensor_names=True)

if __name__ == '__main__':
    from vistools.train_helpers import get_model_and_load_weights
    from vistools.config import ConfigManager


    cm = ConfigManager('dense_config_sage.json')
    cfg = cm.get_json()
    model = get_model_and_load_weights(cfg)
    
    ckpt_name = './tf_ckpts/bsi-training-dense-2018-05-03-17-03-40-446/checkpoints/model.ckpt-989'
    keras_load_ckpt(ckpt_name, model)
