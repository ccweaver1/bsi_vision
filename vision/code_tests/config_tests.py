import unittest
import numpy as np
import json
from vistools.config import ConfigManager


test_config = 'test_config.json'
class ConfigTests(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def create_config_instance(config_filename, folder='configs/', always_write=False):
        return ConfigManager(config_filename, folder=folder, always_write=always_write)

    def test_create_config(self):
        c = self.create_config_instance(test_config)
        self.assertIsNotNone(c)
    
    def test_read_config_vars(self):
        c = self.create_config_instance(test_config)
        self.assertIsInstance(c.get('hyperparameters'), dict)
        self.assertIsInstance(c.get('model', 'model_name'), unicode)

    def test_put(self):
        c = self.create_config_instance(test_config)  
        c.put('x', 'blah')
        self.assertEqual(c.get('x'), 'blah')

    def test_put_then_write(self):
        c = self.create_config_instance(test_config)  
        c.put('x', 'blah')
        c.write()
        new_c = self.create_config_instance(test_config)
        self.assertEqual(new_c.get('x'), 'blah')   

    def test_put_and_write(self):
        c = self.create_config_instance(test_config, always_write=True)
        c.put('y', 'yah')
        new_c = self.create_config_instance(test_config)
        self.assertEqual(new_c.get('y'), 'yah')

    def test_put_and_write_non_string(self):
        c = self.create_config_instance(test_config, always_write=True)
        c.put('a_number', 10)
        new_c = self.create_config_instance(test_config)
        self.assertEqual(new_c.get('a_number'), 10)

    def test_put_dict(self):
        c = self.create_config_instance(test_config)
        new_config_dict = {'lr': 1.00, 'train': True}
        c.put('z', new_config_dict)
        self.assertDictEqual(c.get('z'), new_config_dict)

    def test_update_nested_element(self):
        c = self.create_config_instance(test_config)
        new_config_dict = {'lr': 20, 'test': False}
        c.put('just', new_config_dict)
        self.assertEqual(c.get('just', 'lr'), 20)
        c.put('just', 'lr', 30)
        self.assertEqual(c.get('just', 'lr'), 30)
        c.write()

    def test_edit_json_obj_directly(self):
        c = self.create_config_instance(test_config)
        j = c.get_json()
        j['y'] = 'nah'
        self.assertEqual(j['y'], c.get('y'))   
    
    def test_put_new_val_under_new_key(self):
        c = self.create_config_instance(test_config)
        c.put('new_key', 'another_new_key', 'new val')
        self.assertEqual(c.get('new_key', 'another_new_key'), 'new val')

    def test_load_from_dict(self):
        c = self.create_config_instance(test_config)
        new_config_info = {'run': True, 'yolo': 'absolutely'}
        c.load_from_dict(new_config_info)
        self.assertEqual(c.get('run'), True)     

if __name__ == '__main__':
    unittest.main()