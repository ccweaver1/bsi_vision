import unittest
import numpy as np
import json
import tensorflow as tf
from vistools.data import Data
from vistools.train_helpers import get_model_and_load_weights, compile_model
from vistools.config import ConfigManager

class CodeTests(unittest.TestCase):

    @staticmethod
    def load_compiled_model(config):
        model = get_model_and_load_weights(config)
        compile_model(model, config)
        return model
    
    @staticmethod
    def get_target_shape_from_config(config):
        target_shape = tuple(config.get('model', 'target_shape'))
        if len(target_shape) == 2:
            target_shape += (3,)
        return target_shape

    @staticmethod
    def get_array_from_tensor(tensors):
        if not isinstance(tensors, list):
            tensors = [tensors]
        ret = []
        sess = tf.Session()
        with sess.as_default():
            for t in tensors:
                ret.append(t.eval())
        if len(ret) == 1:
            ret = ret[0]
        return ret


    def setUp(self):
        data_path = 'train_practice'
        config_manager = ConfigManager('resnet50_pool_config_v1.json')
        self.config = config_manager.get_json()
        self.assertIsNotNone(self.config)
        self.data = Data(data_path, self.config, 'train')
        self.assertIsNotNone(self.data)

    def test_create_training_data_object(self):
        self.assertGreater(len(self.data.train_paths) + len(self.data.test_paths), 0) 

    def test_get_data_for_sagemaker(self):
        input_layer_name = 'input_1'
        X, y = self.data.get_data_for_sagemaker('train', input_layer_name)
        for i, key in enumerate(X):
            x_arr, y_arr = self.get_array_from_tensor([X[key], y])
            num_x_samples = x_arr.shape[0]
            num_y_samples = y_arr.shape[0]
            self.assertEqual(num_x_samples, num_y_samples)
            target_shape = self.get_target_shape_from_config(self.config)
            self.assertTupleEqual(x_arr.shape[1:], target_shape)
            self.assertEqual(y_arr.shape[1], 9)

    def test_model_loading(self):
        model = self.load_compiled_model(self.config)
        target_shape = self.get_target_shape_from_config(self.config)
        input_layer = model.layers[0]
        input_shape = input_layer.input_shape
        self.assertTupleEqual(input_shape[1:], target_shape)

        output_layer = model.layers[-1]
        output_shape = output_layer.output_shape
        self.assertEqual(output_shape[1], 9)
        
    def test_predictions(self):
        model = self.load_compiled_model(self.config)
        model_input_name = model.layers[0].name
        X, y = self.data.get_data_for_sagemaker('train', model_input_name)
        


if __name__ == '__main__':
    unittest.main()