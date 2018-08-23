import json
import os

class ConfigManager(object):
    '''
    A manager class for config files.  Config files are written in json.  They are
    loaded, read and written to by training and testing code.
    '''
    
    def __init__(self, config_filename, folder='configs/', always_write=True):
        if '/' in config_filename:
            self.file_path = config_filename
        else:
            self.file_path = os.path.join(folder, config_filename)
        self.always_write = always_write
        with open(self.file_path, 'r') as f:
            self.json = json.load(f)


    def get(self, *keys):
        ret = self.json
        for k in keys:
            ret = ret.get(k, {})
        return ret

    def put(self, *args):
        if len(args) < 2:
            raise ValueError('Expected 1 or more keys followed by value.')
        keys = args[:-1]
        val = args[-1]
        temp = self.json
        for k in keys[:-1]:
            check_for_k = temp.get(k, {})
            if not check_for_k:
                temp[k] = {}
                temp = temp.get(k)
            else:
                temp = check_for_k

        temp[keys[-1]] = val
        if self.always_write:
            self.write()
    
    def write(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.json, f)
        
    def get_json(self):
        return self.json
    
    def load_from_dict(self, d):
        s = json.dumps(d)
        self.json = json.loads(s)

    def write_copy(self, new_filename):
        with open(new_filename, 'w') as f:
            json.dump(self.json, f)

if __name__ == '__main__':
    c = ConfigManager('resnet50_pool_config_sage.json')
    print c.get('hyperparameters')